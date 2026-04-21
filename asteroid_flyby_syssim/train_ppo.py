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

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import toml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

try:
    import torchrl
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
    from torchrl.objectives import ClipPPOLoss
except ImportError as e:
    print(f"Error: torchrl not installed. Install with: pip install pytorch-rl")
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
        self.vit = timm.create_model(
            vit_model,
            pretrained=vit_pretrained,
            num_classes=hidden_dim,  # Output feature dimension
            img_size=image_size,
        )
        
        # Freeze early ViT layers if specified
        vit_blocks = self.vit.blocks
        num_blocks = len(vit_blocks)
        for i in range(min(vit_freeze_depth, num_blocks)):
            for param in vit_blocks[i].parameters():
                param.requires_grad = False
        
        # Temporal attention for gyroscope history
        # Input: [batch, gyro_history_len, 3] → [batch, temporal_attention_dim]
        self.gyro_temporal_attn = nn.MultiheadAttention(
            embed_dim=3,  # Gyro is 3D
            num_heads=temporal_attention_heads,
            batch_first=True,
            dtype=torch.float32,
        )
        self.gyro_projection = nn.Linear(3, temporal_attention_dim)
        
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
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
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
        gyro_attn_out, _ = self.gyro_temporal_attn(
            gyro_history, gyro_history, gyro_history
        )  # [batch, history_len, 3]
        gyro_features = gyro_attn_out.mean(dim=1)  # Average over time
        gyro_features = self.gyro_projection(gyro_features)  # [batch, temporal_dim]
        
        # Fuse features
        fused = torch.cat([vit_features, gyro_features], dim=-1)
        shared = self.shared_net(fused)
        
        # Actor
        action_mean = self.actor_mean(shared)
        
        # Critic
        value = self.critic(shared)
        
        return action_mean, value


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
    
    # Set seeds
    seed = config["reproducibility"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    logger.info("Creating environment...")
    env = create_environment(config)
    
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
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=config["training"]["learning_rate"],
    )
    
    # Training loop
    step_count = 0
    episode_count = 0
    best_return = -np.inf
    steps_without_improvement = 0
    
    logger.info("Starting training...")
    
    try:
        while step_count < config["training"]["max_steps"]:
            # Collect experience (placeholder)
            # In real implementation, this would collect from parallel environments
            
            # PPO update (placeholder)
            # In real implementation, this would:
            # 1. Collect trajectory data
            # 2. Compute advantages with GAE
            # 3. Do PPO clipped objective updates
            
            # Logging (placeholder)
            if step_count % config["logging"]["log_frequency"] == 0:
                writer.add_scalar("train/step", step_count, step_count)
                logger.info(f"Step {step_count}/{config['training']['max_steps']}")
            
            # Checkpointing
            if step_count % config["training"]["checkpoint_frequency"] == 0:
                checkpoint_path = output_dir / "checkpoints" / f"policy_step_{step_count}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "step": step_count,
                    "policy_state": policy_net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            step_count += 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    
    logger.info("Training complete!")
    writer.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train RL policy for asteroid visual acquisition and tracking.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_training_config.toml"),
        help="Path to configuration TOML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rl_training"),
        help="Output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Run training
    train(config, args.output_dir, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
