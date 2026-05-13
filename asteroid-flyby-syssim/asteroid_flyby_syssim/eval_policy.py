"""Evaluation script for trained RL policies.

This script loads a trained policy and runs it in the asteroid tracking
environment, collecting metrics and optionally rendering videos.

Usage:
    python -m asteroid_flyby_syssim.eval_policy \
        --checkpoint outputs/rl_training/checkpoints/policy_step_100000.pt \
        --config configs/rl_training_config.toml \
        --num-episodes 10 \
        --render
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .train_ppo import TrainArgs, ViTGyroPolicy, load_config, create_environment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_episode(
    env,
    policy_net: ViTGyroPolicy,
    device: torch.device,
    deterministic: bool = True,
    render_mode: Optional[str] = None,
) -> dict:
    """Run a single episode with the policy.
    
    Parameters
    ----------
    env
        Gymnasium environment.
    policy_net : ViTGyroPolicy
        Policy network.
    device : torch.device
        Computation device.
    deterministic : bool
        Use deterministic policy (argmax) vs. sampling.
    render_mode : str, optional
        Render mode (e.g., "rgb_array").
    
    Returns
    -------
    episode_data : dict
        Episode metrics:
        - episode_return: Total discounted return
        - episode_length: Number of steps
        - time_to_acquisition: Steps until asteroid first visible
        - mean_tracking_error: Mean centering error when visible
        - visibility_rate: Fraction of steps with asteroid visible
        - num_visible_steps: Total steps asteroid was visible
    """
    obs, info = env.reset()
    
    episode_return = 0.0
    episode_length = 0
    time_to_acquisition = None
    visible_steps = []
    error_angles = []
    frames = []
    
    done = False
    gamma = 0.99  # Discount factor
    
    while not done:
        # Convert observation to torch tensors
        image = torch.from_numpy(obs["image"]).float().to(device)
        image = image.unsqueeze(0)  # Add batch dimension: [1, H, W, 3]
        if image.max() > 1.0:
            image = image / 255.0
        image = image.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        gyro_history = torch.from_numpy(obs["gyro_history"]).float().to(device)
        gyro_history = gyro_history.unsqueeze(0)  # [1, history_len, 3]
        
        # Get action from policy
        action_mean, _ = policy_net(image, gyro_history)
        
        if deterministic:
            action = torch.tanh(action_mean).detach().cpu().numpy()[0]
        else:
            # Sample from distribution with learned std
            std = torch.exp(policy_net.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = torch.tanh(dist.sample()).detach().cpu().numpy()[0]
        
        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Accumulate metrics
        episode_return += reward * (gamma ** episode_length)
        episode_length += 1
        
        # Track visibility and error
        if step_info.get("asteroid_visible", False):
            visible_steps.append(episode_length)
            error_angles.append(step_info.get("error_angle_deg", 180.0))
            if time_to_acquisition is None:
                time_to_acquisition = episode_length
        
        # Optionally render
        if render_mode == "rgb_array":
            # Try to get rendered frame from env
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Compute metrics
    data = {
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "time_to_acquisition": int(time_to_acquisition) if time_to_acquisition else episode_length,
        "num_visible_steps": int(len(visible_steps)),
        "visibility_rate": float(len(visible_steps) / episode_length) if episode_length > 0 else 0.0,
        "mean_tracking_error_deg": float(np.mean(error_angles)) if error_angles else 180.0,
        "max_tracking_error_deg": float(np.max(error_angles)) if error_angles else 180.0,
        "min_tracking_error_deg": float(np.min(error_angles)) if error_angles else 180.0,
    }
    
    if frames:
        data["frames"] = frames
    
    return data


def evaluate(
    checkpoint_path: Path,
    config: TrainArgs,
    num_episodes: int = 10,
    output_dir: Optional[Path] = None,
    deterministic: bool = True,
    render: bool = False,
):
    """Evaluate a trained policy.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file.
    config : TrainArgs
        Training arguments.
    num_episodes : int
        Number of evaluation episodes.
    output_dir : Path, optional
        Output directory for results CSV. If None, no CSV saved.
    deterministic : bool
        Use deterministic policy.
    render : bool
        Save render videos (if env supports it).
    """
    # Setup device
    device = torch.device(config.device.device)
    
    # Load policy
    logger.info(f"Loading policy from {checkpoint_path}...")
    policy_net = ViTGyroPolicy(
        image_size=config.rl_environment.camera_height,
        vit_model=config.network.vit_model,
        vit_pretrained=config.network.vit_pretrained,
        vit_freeze_depth=config.network.vit_freeze_depth,
        temporal_attention_heads=config.network.temporal_attention_heads,
        temporal_attention_dim=config.network.temporal_attention_dim,
        hidden_dim=config.network.hidden_dim,
        action_dim=3,
    ).to(device)
    policy_net.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_state"])
    logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
    
    # Create environment
    logger.info("Creating environment...")
    env = create_environment(config)
    
    # Run episodes
    logger.info(f"Running {num_episodes} evaluation episodes...")
    episode_data_list = []
    
    for i in tqdm(range(num_episodes), desc="Evaluating"):
        render_mode = "rgb_array" if render else None
        episode_data = run_episode(
            env,
            policy_net,
            device,
            deterministic=deterministic,
            render_mode=render_mode,
        )
        episode_data["episode_num"] = i + 1
        episode_data_list.append(episode_data)
        
        logger.info(
            f"Episode {i+1}: return={episode_data['episode_return']:.2f}, "
            f"length={episode_data['episode_length']}, "
            f"visibility={episode_data['visibility_rate']:.1%}"
        )
        
        # Optionally save rendered frames
        if render and "frames" in episode_data and output_dir:
            video_path = output_dir / f"episode_{i+1:03d}.mp4"
            save_video(episode_data["frames"], str(video_path), fps=30)
    
    # Aggregate statistics
    returns = [d["episode_return"] for d in episode_data_list]
    lengths = [d["episode_length"] for d in episode_data_list]
    vis_rates = [d["visibility_rate"] for d in episode_data_list]
    errors = [d["mean_tracking_error_deg"] for d in episode_data_list]
    
    summary = {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_visibility": float(np.mean(vis_rates)),
        "mean_error_deg": float(np.mean(errors)),
    }
    
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    for key, val in summary.items():
        logger.info(f"{key:20s}: {val:10.4f}")
    logger.info("=" * 50)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "eval_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_data_list[0].keys())
            writer.writeheader()
            for data in episode_data_list:
                # Remove frames if present (can't write to CSV)
                data_copy = {k: v for k, v in data.items() if k != "frames"}
                writer.writerow(data_copy)
        logger.info(f"Results saved to {csv_path}")
        
        # Save summary JSON
        import json
        summary_path = output_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
    
    env.close()


def save_video(frames: list, output_path: str, fps: int = 30):
    """Save frames as video.
    
    Parameters
    ----------
    frames : list
        List of numpy arrays [H, W, 3] in uint8 [0, 255].
    output_path : str
        Output video path (e.g., .mp4 or .gif).
    fps : int
        Frames per second.
    """
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps)
        logger.info(f"Video saved to {output_path}")
    except ImportError:
        logger.warning("imageio not installed, cannot save video.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL policy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_training_config.toml"),
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render and save episode videos.",
    )
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return
    
    config = load_config(args.config)
    
    # Evaluate
    evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        deterministic=True,
        render=args.render,
    )


if __name__ == "__main__":
    main()
