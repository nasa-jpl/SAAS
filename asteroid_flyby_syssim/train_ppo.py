"""PPO training script for asteroid visual acquisition and tracking.

This script trains a Vision Transformer-based policy using PPO to autonomously
control spacecraft attitude for acquiring and tracking an asteroid using camera
observations and gyroscope measurements.

Usage:
    python -m asteroid_flyby_syssim.train_ppo \
        --config configs/rl_training_config.toml \
        --output-dir outputs/rl_training
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import toml
import torch
import torch.nn as nn
import tyro
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.utils.tensorboard import SummaryWriter
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

try:
    import timm
except ImportError as e:
    print(f"Error: timm not installed. Install with: pip install timm")
    sys.exit(1)

try:
    import timm
except ImportError as e:
    print(f"Error: timm not installed. Install with: pip install timm")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTGyroPolicy(nn.Module):
    """Policy network combining Vision Transformer and gyroscope history.
    
    Architecture:
    - ViT backbone for image processing
    - Temporal attention over gyroscope history
    - Shared features → actor and critic heads
    """
    
    def __init__(
        self,
        image_size: int = 128,
        vit_model: str = "vit_tiny",
        vit_pretrained: bool = True,
        vit_freeze_depth: int = 6,
        temporal_attention_heads: int = 4,
        temporal_attention_dim: int = 64,
        hidden_dim: int = 256,
        action_dim: int = 3,
        action_std_init: float = 0.5,
    ):
        """Initialize policy network.
        
        Parameters
        ----------
        image_size : int
            Input image size (assumes square images).
        vit_model : str
            timm ViT model name.
        vit_pretrained : bool
            Use pretrained ImageNet weights.
        vit_freeze_depth : int
            Number of transformer layers to freeze from ViT.
        temporal_attention_heads : int
            Number of attention heads for gyro temporal processing.
        temporal_attention_dim : int
            Feature dimension for temporal attention.
        hidden_dim : int
            Hidden dimension for actor/critic MLPs.
        action_dim : int
            Output action dimension.
        """
        super().__init__()
        
        # Vision Transformer for image encoding
        try:
            self.vit = timm.create_model(
                vit_model,
                pretrained=vit_pretrained,
                num_classes=hidden_dim,
                img_size=image_size,
            )
        except Exception as exc:
            fallback_model = "vit_base_patch16_224"
            logger.warning(
                f"Failed to create ViT model '{vit_model}': {exc}. "
                f"Falling back to '{fallback_model}'."
            )
            self.vit = timm.create_model(
                fallback_model,
                pretrained=vit_pretrained,
                num_classes=hidden_dim,
                img_size=image_size,
            )
        
        # Freeze early ViT layers if specified
        vit_blocks = self.vit.blocks
        num_blocks = len(vit_blocks)
        for i in range(min(vit_freeze_depth, num_blocks)):
            for param in vit_blocks[i].parameters():
                param.requires_grad = False
        
        # Temporal attention for gyroscope history
        # Project gyro history into the attention embedding dimension.
        self.gyro_input_proj = nn.Linear(3, temporal_attention_dim)
        self.gyro_temporal_attn = nn.MultiheadAttention(
            embed_dim=temporal_attention_dim,
            num_heads=temporal_attention_heads,
            batch_first=True,
            dtype=torch.float32,
        )
        
        # Feature fusion
        vit_output_dim = hidden_dim
        total_feature_dim = vit_output_dim + temporal_attention_dim
        
        # Shared backbone
        self.shared_net = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * action_std_init)
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, image: torch.Tensor, gyro_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for policy.
        
        Parameters
        ----------
        image : Tensor
            Image batch [batch, 3, 128, 128] float [0, 1].
        gyro_history : Tensor
            Gyro history [batch, history_len, 3].
        
        Returns
        -------
        action_mean : Tensor
            Mean action [batch, action_dim].
        value : Tensor
            Value estimate [batch, 1].
        """
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # ViT encoding
        vit_features = self.vit(image)  # [batch, hidden_dim]
        
        # Temporal processing of gyro
        gyro_tokens = self.gyro_input_proj(gyro_history)
        gyro_attn_out, _ = self.gyro_temporal_attn(
            gyro_tokens, gyro_tokens, gyro_tokens
        )  # [batch, history_len, temporal_attention_dim]
        gyro_features = gyro_attn_out.mean(dim=1)  # Average over time
        
        # Fuse features
        fused = torch.cat([vit_features, gyro_features], dim=-1)
        shared = self.shared_net(fused)
        
        # Actor
        action_mean = self.actor_mean(shared)
        
        # Critic
        value = self.critic(shared)
        
        return action_mean, value


class TorchRLPPOAdapter(nn.Module):
    """Wrap ViTGyroPolicy for TorchRL TensorDict inputs."""

    def __init__(self, base_policy: ViTGyroPolicy):
        super().__init__()
        self.base_policy = base_policy

    def forward(self, observation: TensorDict) -> dict[str, torch.Tensor]:
        image = observation["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        gyro_history = observation["gyro_history"].to(dtype=torch.float32)
        action_mean, _ = self.base_policy(image, gyro_history)
        action_scale = self.base_policy.log_std.exp().view(1, -1).expand_as(action_mean)
        return {"loc": action_mean, "scale": action_scale}


class TorchRLValueAdapter(nn.Module):
    """Wrap ViTGyroPolicy value head for TorchRL TensorDict inputs."""

    def __init__(self, base_policy: ViTGyroPolicy):
        super().__init__()
        self.base_policy = base_policy

    def forward(self, observation: TensorDict) -> dict[str, torch.Tensor]:
        image = observation["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        gyro_history = observation["gyro_history"].to(dtype=torch.float32)
        _, value = self.base_policy(image, gyro_history)
        return {"state_value": value}


def _stack_observations(observations: list[dict], device: torch.device) -> TensorDict:
    """Stack a list of Gym observations into a nested TorchRL TensorDict."""
    images = []
    gyros = []
    for obs in observations:
        image = torch.as_tensor(obs["image"], device=device)
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        images.append(image)
        gyros.append(torch.as_tensor(obs["gyro_history"], device=device, dtype=torch.float32))

    image_batch = torch.stack(images)
    gyro_batch = torch.stack(gyros)
    return TensorDict(
        {
            "observation": TensorDict(
                {
                    "image": image_batch,
                    "gyro_history": gyro_batch,
                },
                batch_size=[len(images)],
            )
        },
        batch_size=[len(images)],
    )


def _build_ppo_modules(
    policy_net: ViTGyroPolicy,
    action_dim: int,
    device: torch.device,
) -> tuple[ProbabilisticActor, TensorDictModule]:
    policy_adapter = TorchRLPPOAdapter(policy_net).to(device)
    value_adapter = TorchRLValueAdapter(policy_net).to(device)

    policy_td_module = TensorDictModule(
        policy_adapter,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_td_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
    )

    critic_module = TensorDictModule(
        value_adapter,
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic_module


@dataclass
class TrainArgs:
    """Command line arguments for the PPO training script."""

    config: Path = Path("configs/rl_training_config.toml")
    output_dir: Path = Path("outputs/rl_training")
    checkpoint: Optional[Path] = None


def load_config(config_path: Path) -> dict:
    """Load TOML configuration file.
    
    Parameters
    ----------
    config_path : Path
        Path to .toml config file.
    
    Returns
    -------
    config : dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = toml.load(f)
    return config


def create_environment(config: dict) -> gym.Env:
    """Create RL environment from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    env : gym.Env
        Asteroid tracking environment.
    """
    from .flyby_sim import FlybyRunConfig
    from .rl_env import AsteroidTrackingEnv, RLEnvironmentConfig
    
    # Build FlybyRunConfig from config
    flyby_cfg = FlybyRunConfig(
        sim=FlybyRunConfig.SimulationConfig(
            sim_dt=config["simulation"]["sim_dt"],
            start_date_utc=config["simulation"]["start_date_utc"],
            duration_seconds=config["simulation"]["duration_seconds"],
        ),
        asteroid=FlybyRunConfig.AsteroidConfig(
            asteroid=config["asteroid"]["asteroid"],
            gravity_lmax=config["asteroid"]["gravity_lmax"],
        ),
        # ... (fill in other configs as needed)
    )
    
    # Build RLEnvironmentConfig
    rl_cfg = RLEnvironmentConfig(
        camera_width=config["rl_environment"]["camera_width"],
        camera_height=config["rl_environment"]["camera_height"],
        camera_fov_deg=config["rl_environment"]["camera_fov_deg"],
        max_steps=config["rl_environment"]["max_steps"],
        max_steps_without_asteroid=config["rl_environment"]["max_steps_without_asteroid"],
        asteroid_visibility_threshold=config["rl_environment"]["asteroid_visibility_threshold"],
        torque_scale_nm=config["rl_environment"]["torque_scale_nm"],
        gyro_history_length=config["rl_environment"]["gyro_history_length"],
        render_at_frequency=config["rl_environment"]["render_at_frequency"],
    )
    
    env = AsteroidTrackingEnv(flyby_config=flyby_cfg, rl_config=rl_cfg)
    return env


def train(
    config: dict,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
):
    """Main training loop.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_dir : Path
        Output directory for logs and checkpoints.
    checkpoint_path : Path, optional
        Path to checkpoint to resume from.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tensorboard logging
    log_dir = output_dir / config["logging"]["tensorboard_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    
    # Save config
    config_path = output_dir / "config.toml"
    with open(config_path, "w") as f:
        toml.dump(config, f)
    logger.info(f"Config saved to {config_path}")
    
    # Setup device
    device = torch.device(config["device"]["device"])
    logger.info(f"Training on device: {device}")
    
    if config["reproducibility"].get("deterministic_torch", False):
        try:
            torch.use_deterministic_algorithms(True)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception as exc:
            logger.warning(f"Could not enable fully deterministic PyTorch mode: {exc}")
    
    # Set seeds
    seed = config["reproducibility"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environments
    num_envs = max(1, int(config["training"]["num_envs"]))
    logger.info(f"Creating {num_envs} environment(s)...")
    envs = [create_environment(config) for _ in range(num_envs)]
    current_obs = [env.reset()[0] for env in envs]
    episode_returns = [0.0 for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    recent_episode_returns: list[float] = []
    recent_episode_lengths: list[int] = []
    
    # Create networks
    logger.info("Creating policy network...")
    policy_net = ViTGyroPolicy(
        image_size=config["rl_environment"]["camera_height"],
        vit_model=config["network"]["vit_model"],
        vit_pretrained=config["network"]["vit_pretrained"],
        vit_freeze_depth=config["network"]["vit_freeze_depth"],
        temporal_attention_heads=config["network"]["temporal_attention_heads"],
        temporal_attention_dim=config["network"]["temporal_attention_dim"],
        hidden_dim=config["network"]["hidden_dim"],
        action_dim=3,
        action_std_init=config["network"].get("action_std_init", 0.5),
    ).to(device)
    
    # PPO modules
    actor, critic = _build_ppo_modules(policy_net, action_dim=3, device=device)
    ppo_loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=config["training"]["ppo_clip_ratio"],
        entropy_coeff=config["training"]["entropy_coeff"],
        critic_coeff=config["training"]["value_coeff"],
        normalize_advantage=True,
    )
    ppo_loss.make_value_estimator(
        ValueEstimators.GAE,
        gamma=config["training"]["gamma"],
        lmbda=config["training"]["gae_lambda"],
        deactivate_vmap=True,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=config["training"]["learning_rate"],
    )
    
    # Resume from checkpoint if requested
    if checkpoint_path is not None:
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            policy_net.load_state_dict(checkpoint["policy_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            step_count = int(checkpoint.get("step", step_count))
            logger.info(f"Resumed checkpoint from {checkpoint_path} at step {step_count}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Training loop
    step_count = 0
    episode_count = 0
    best_return = -np.inf
    steps_without_improvement = 0
    
    logger.info("Starting training...")
    
    try:
        while step_count < config["training"]["max_steps"]:
            rollout_steps = int(config["training"]["steps_per_rollout"])
            batch_actions = []
            batch_action_log_probs = []
            batch_rewards = []
            batch_dones = []
            batch_terminated = []
            batch_observations = []
            batch_next_observations = []
            batch_next_rewards = []
            
            for _ in range(rollout_steps):
                observation_td = _stack_observations(current_obs, device)
                dist = actor.get_dist(observation_td)
                actions = dist.rsample().clamp(-1.0, 1.0)
                action_log_probs = dist.log_prob(actions).unsqueeze(-1)
                
                next_obs_list = []
                rewards = []
                dones = []
                terminated_flags = []
                next_rewards = []
                
                for env_idx, env in enumerate(envs):
                    action = actions[env_idx].cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    
                    batch_observations.append(current_obs[env_idx])
                    batch_next_observations.append(obs)
                    batch_actions.append(actions[env_idx].cpu())
                    batch_action_log_probs.append(action_log_probs[env_idx].cpu())
                    batch_next_rewards.append(torch.tensor([reward], device=device, dtype=torch.float32))
                    batch_rewards.append(torch.tensor([reward], device=device, dtype=torch.float32))
                    batch_dones.append(torch.tensor([done], device=device, dtype=torch.bool))
                    batch_terminated.append(torch.tensor([terminated], device=device, dtype=torch.bool))
                    
                    episode_returns[env_idx] += float(reward)
                    episode_lengths[env_idx] += 1
                    
                    if done:
                        recent_episode_returns.append(episode_returns[env_idx])
                        recent_episode_lengths.append(episode_lengths[env_idx])
                        episode_count += 1
                        if episode_returns[env_idx] > best_return:
                            best_return = episode_returns[env_idx]
                            steps_without_improvement = 0
                        else:
                            steps_without_improvement += 1
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        obs, _ = env.reset()
                    
                    current_obs[env_idx] = obs
                
            num_transitions = len(batch_actions)
            if num_transitions == 0:
                break
            
            batch = TensorDict(
                {
                    "observation": TensorDict(
                        {
                            "image": torch.stack([
                                torch.as_tensor(o["image"], device=device).permute(2, 0, 1)
                                if torch.as_tensor(o["image"]).ndim == 3
                                else torch.as_tensor(o["image"], device=device)
                                for o in batch_observations
                            ]),
                            "gyro_history": torch.stack(
                                [torch.as_tensor(o["gyro_history"], device=device, dtype=torch.float32) for o in batch_observations]
                            ),
                        },
                        batch_size=[num_transitions],
                    ),
                    "action": torch.stack(batch_actions),
                    "action_log_prob": torch.stack(batch_action_log_probs),
                    "done": torch.stack(batch_dones),
                    "terminated": torch.stack(batch_terminated),
                    "next": TensorDict(
                        {
                            "observation": TensorDict(
                                {
                                    "image": torch.stack([
                                        torch.as_tensor(o["image"], device=device).permute(2, 0, 1)
                                        if torch.as_tensor(o["image"]).ndim == 3
                                        else torch.as_tensor(o["image"], device=device)
                                        for o in batch_next_observations
                                    ]),
                                    "gyro_history": torch.stack(
                                        [torch.as_tensor(o["gyro_history"], device=device, dtype=torch.float32) for o in batch_next_observations]
                                    ),
                                },
                                batch_size=[num_transitions],
                            ),
                            "reward": torch.stack(batch_next_rewards),
                            "done": torch.stack(batch_dones),
                            "terminated": torch.stack(batch_terminated),
                        },
                        batch_size=[num_transitions],
                    ),
                },
                batch_size=[num_transitions],
            )
            
            total_loss = 0.0
            shuffled_indices = torch.randperm(num_transitions, device=device)
            for epoch in range(int(config["training"]["num_epochs"])):
                for start in range(0, num_transitions, int(config["training"]["batch_size"])):
                    batch_idx = shuffled_indices[start : start + int(config["training"]["batch_size"])]
                    minibatch = batch.view(-1)[batch_idx]
                    loss_out = ppo_loss(minibatch)
                    loss_values = [loss_out["loss_objective"].mean()]
                    if "loss_entropy" in loss_out:
                        loss_values.append(loss_out["loss_entropy"].mean())
                    if "loss_critic" in loss_out:
                        loss_values.append(loss_out["loss_critic"].mean())
                    loss = sum(loss_values)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config["training"]["grad_clip_norm"])
                    optimizer.step()
                    total_loss += loss.item()
            
            step_count += num_transitions
            avg_return = float(np.mean(recent_episode_returns)) if recent_episode_returns else 0.0
            avg_length = float(np.mean(recent_episode_lengths)) if recent_episode_lengths else 0.0
            
            if step_count % config["logging"]["log_frequency"] == 0:
                writer.add_scalar("train/step", step_count, step_count)
                writer.add_scalar("train/ppo/loss", total_loss / max(1, int(config["training"]["num_epochs"]) * (num_transitions // int(config["training"]["batch_size"]))), step_count)
                writer.add_scalar("train/episode/return", avg_return, step_count)
                writer.add_scalar("train/episode/length", avg_length, step_count)
                logger.info(
                    f"Step {step_count}/{config['training']['max_steps']} | avg_return={avg_return:.3f} | avg_length={avg_length:.1f}"
                )
            
            if step_count % config["training"]["checkpoint_frequency"] == 0:
                checkpoint_path = output_dir / "checkpoints" / f"policy_step_{step_count}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "step": step_count,
                    "policy_state": policy_net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    
    logger.info("Training complete!")
    writer.close()


def main() -> int:
    """Main entry point."""
    args = tyro.cli(TrainArgs)
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    train(config, args.output_dir, checkpoint_path=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
