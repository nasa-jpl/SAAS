"""PPO training script for asteroid visual acquisition and tracking.

This script trains a Vision Transformer-based policy using PPO to autonomously
control spacecraft attitude for acquiring and tracking an asteroid using camera
observations and gyroscope measurements.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import importlib

try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:
    tomllib = importlib.import_module("tomli")

import gymnasium as gym
import numpy as np
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelEnvExecutor:
    """Execute environment steps in parallel using thread pool.
    
    This executor manages concurrent environment stepping to improve throughput.
    Each environment runs in a separate thread, allowing efficient parallelization
    of I/O-bound simulation operations.
    """
    
    def __init__(self, envs: list, max_workers: Optional[int] = None):
        """Initialize parallel executor.
        
        Parameters
        ----------
        envs : list
            List of gymnasium environments to manage.
        max_workers : int, optional
            Maximum number of worker threads. If None, uses len(envs).
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.max_workers = max_workers or len(envs)
        self._lock = threading.Lock()
        
    def step_all(self, actions: np.ndarray) -> tuple[list, list, list, list, list]:
        """Step all environments in parallel.
        
        Parameters
        ----------
        actions : np.ndarray
            Actions for each environment [num_envs, action_dim].
        
        Returns
        -------
        observations : list
            List of observations from each environment.
        rewards : list
            List of rewards from each environment.
        terminateds : list
            List of terminated flags.
        truncateds : list
            List of truncated flags.
        infos : list
            List of info dicts.
        """
        def step_env(env_idx: int, action: np.ndarray):
            env = self.envs[env_idx]
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                return env_idx, (obs, reward, terminated, truncated, info)
            except Exception as e:
                logger.error(f"Error stepping environment {env_idx}: {e}")
                raise
        
        results = [None] * self.num_envs
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for env_idx in range(self.num_envs):
                future = executor.submit(step_env, env_idx, actions[env_idx])
                futures[future] = env_idx
            
            for future in as_completed(futures):
                env_idx, result = future.result()
                results[env_idx] = result
        
        # Unpack results
        observations = [r[0] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]
        
        return observations, rewards, terminateds, truncateds, infos


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
        vit_model: str = "vit_tiny_patch16_224",
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

        # Accept common shorthand aliases for timm model IDs.
        vit_aliases = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }
        resolved_vit_model = vit_aliases.get(vit_model, vit_model)
        
        # Vision Transformer for image encoding
        try:
            self.vit = timm.create_model(
                resolved_vit_model,
                pretrained=vit_pretrained,
                num_classes=hidden_dim,
                img_size=image_size,
            )
        except Exception as exc:
            fallback_model = "vit_tiny_patch16_224"
            logger.warning(
                f"Failed to create ViT model '{resolved_vit_model}': {exc}. "
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
    image_batch = _stack_image_batch(observations, device)
    gyro_batch = _stack_gyro_batch(observations, device)
    batch_len = len(observations)
    return TensorDict(
        {
            "observation": TensorDict(
                {
                    "image": image_batch,
                    "gyro_history": gyro_batch,
                },
                batch_size=[batch_len],
            )
        },
        batch_size=[batch_len],
    )


def _stack_image_batch(observations: list[dict], device: torch.device) -> torch.Tensor:
    images = []
    for obs in observations:
        image = torch.as_tensor(obs["image"], device=device)
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        images.append(image)
    return torch.stack(images)


def _stack_gyro_batch(observations: list[dict], device: torch.device) -> torch.Tensor:
    return torch.stack(
        [torch.as_tensor(obs["gyro_history"], device=device, dtype=torch.float32) for obs in observations]
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
class SimulationArgs:
    sim_dt: float = 0.01
    start_date_utc: str = "2024-01-01T00:00:00"
    duration_seconds: float = 3600.0


@dataclass
class AsteroidArgs:
    asteroid: str = "Ceres"
    gravity_lmax: int = 10


@dataclass
class FlybyArgs:
    periapsis_radius_m: float = 3000.0
    external_angle_deg: float = 120.0
    true_anomaly0_deg: float = -90.0
    inbound_ra_deg: float = 0.0
    inbound_dec_deg: float = 0.0
    bplane_angle_deg: float = 0.0


@dataclass
class SpacecraftArgs:
    inertia_kgm2: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    boresight_body: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class ControllerArgs:
    kp: float = 0.5
    kd: float = 2.0
    ki: float = 0.01
    integral_limit: float = 0.1


@dataclass
class ReactionWheelArgs:
    max_momentum_nms: float = 10.0
    max_torque_nm: float = 1.0
    max_speed_rps: float = 628.3
    inertia_kgm2: float = 0.01
    friction_viscous: float = 0.001
    friction_coulomb: float = 0.001
    command_lag_tau: float = 0.01


@dataclass
class GyroscopeArgs:
    bias_rad_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale_errors: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    white_noise_std_rad_s: float = 1e-4
    bias_random_walk_std_rad_s2: float = 1e-6
    sample_rate_hz: float = 100.0


@dataclass
class OutputArgs:
    output_dir: str = "outputs/rl_training"
    run_name: str = "ast_track_01"
    render_video: bool = False
    camera_width: int = 128
    camera_height: int = 128
    camera_fov_deg: float = 50.0
    render_fps: float = 10.0
    spp: int = 2
    use_integrator_mask: bool = False


@dataclass
class RLEnvironmentArgs:
    camera_width: int = 128
    camera_height: int = 128
    camera_fov_deg: float = 50.0
    max_steps: int = 1000
    max_steps_without_asteroid: int = 100
    asteroid_visibility_threshold: float = 0.01
    torque_scale_nm: float = 1.0
    gyro_history_length: int = 4
    render_at_frequency: Optional[float] = None
    randomize_on_reset: bool = True
    periapsis_radius_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    external_angle_offset_deg_range: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    true_anomaly0_offset_deg_range: list[float] = field(default_factory=lambda: [-45.0, 45.0])
    inbound_ra_offset_deg_range: list[float] = field(default_factory=lambda: [-30.0, 30.0])
    inbound_dec_offset_deg_range: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    bplane_angle_offset_deg_range: list[float] = field(default_factory=lambda: [-30.0, 30.0])


@dataclass
class NetworkArgs:
    vit_model: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = True
    vit_freeze_depth: int = 6
    temporal_attention_heads: int = 4
    temporal_attention_dim: int = 64
    hidden_dim: int = 256
    action_std_init: float = 0.5


@dataclass
class TrainingArgs:
    algorithm: str = "PPO"
    num_envs: int = 8
    steps_per_rollout: int = 512
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    grad_clip_norm: float = 0.5
    ppo_clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    max_steps: int = 1_000_000
    checkpoint_frequency: int = 10_000
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.95


@dataclass
class LoggingArgs:
    tensorboard_log_dir: str = "outputs/rl_training/logs"
    log_frequency: int = 100
    save_video_frequency: Optional[int] = None
    log_episode_return: bool = True
    log_episode_length: bool = True
    log_policy_loss: bool = True
    log_value_loss: bool = True
    log_entropy: bool = True
    log_asteroid_visibility: bool = True
    log_mean_error_angle: bool = True
    log_gradient_norm: bool = True


@dataclass
class EvaluationArgs:
    num_eval_episodes: int = 10
    eval_frequency: int = 50_000
    deterministic: bool = True
    render_video: bool = True
    video_dir: str = "outputs/rl_training/eval_videos"


@dataclass
class DeviceArgs:
    device: str = "cuda"
    mixed_precision: bool = False


@dataclass
class ReproducibilityArgs:
    seed: int = 42
    deterministic_torch: bool = True


@dataclass
class TrainArgs:
    """Command line arguments for PPO training."""

    output_dir: Path = Path("outputs/rl_training")
    checkpoint: Optional[Path] = None
    simulation: SimulationArgs = field(default_factory=SimulationArgs)
    asteroid: AsteroidArgs = field(default_factory=AsteroidArgs)
    flyby: FlybyArgs = field(default_factory=FlybyArgs)
    spacecraft: SpacecraftArgs = field(default_factory=SpacecraftArgs)
    controller: ControllerArgs = field(default_factory=ControllerArgs)
    reaction_wheel: ReactionWheelArgs = field(default_factory=ReactionWheelArgs)
    gyroscope: GyroscopeArgs = field(default_factory=GyroscopeArgs)
    output: OutputArgs = field(default_factory=OutputArgs)
    rl_environment: RLEnvironmentArgs = field(default_factory=RLEnvironmentArgs)
    network: NetworkArgs = field(default_factory=NetworkArgs)
    training: TrainingArgs = field(default_factory=TrainingArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    evaluation: EvaluationArgs = field(default_factory=EvaluationArgs)
    device: DeviceArgs = field(default_factory=DeviceArgs)
    reproducibility: ReproducibilityArgs = field(default_factory=ReproducibilityArgs)

def create_environment(args: TrainArgs) -> gym.Env:
    """Create RL environment from config.
    
    Parameters
    ----------
    args : TrainArgs
        Training arguments.
    
    Returns
    -------
    env : gym.Env
        Asteroid tracking environment.
    """
    from .asteroid_camera import NodeAsteroidCamera
    from .flyby_sim import (
        AsteroidConfig,
        ControllerConfig,
        FlybyConfig,
        FlybyRunConfig,
        OutputConfig,
        ReactionWheelConfig,
        SimulationConfig,
        SpacecraftConfig,
    )
    from .rl_env import AsteroidTrackingEnv, RLEnvironmentConfig

    supported_asteroids = set(NodeAsteroidCamera.ASTEROID_SHAPE_DATASETS.keys())
    asteroid_name = str(args.asteroid.asteroid)
    if asteroid_name not in supported_asteroids:
        supported_display = ", ".join(sorted(supported_asteroids))
        raise ValueError(
            f"Unsupported asteroid '{asteroid_name}'. Supported values: {supported_display}."
        )

    render_frequency = args.rl_environment.render_at_frequency
    if render_frequency is None:
        render_frequency = args.output.render_fps

    rw_max_momentum = float(args.reaction_wheel.max_momentum_nms)
    rw_max_torque = float(args.reaction_wheel.max_torque_nm)
    rw_max_speed = float(args.reaction_wheel.max_speed_rps)
    rw_inertia = float(args.reaction_wheel.inertia_kgm2)

    flyby_cfg = FlybyRunConfig(
        sim=SimulationConfig(
            dt_s=args.simulation.sim_dt,
            start_date_utc=args.simulation.start_date_utc,
            duration_s=args.simulation.duration_seconds,
        ),
        asteroid=AsteroidConfig(
            asteroid=args.asteroid.asteroid,
            gravity_lmax=args.asteroid.gravity_lmax,
        ),
        flyby=FlybyConfig(
            periapsis_radius_m=args.flyby.periapsis_radius_m,
            external_angle_deg=args.flyby.external_angle_deg,
            true_anomaly0_deg=args.flyby.true_anomaly0_deg,
            inbound_ra_deg=args.flyby.inbound_ra_deg,
            inbound_dec_deg=args.flyby.inbound_dec_deg,
            bplane_angle_deg=args.flyby.bplane_angle_deg,
        ),
        spacecraft=SpacecraftConfig(
            inertia_kgm2=tuple(args.spacecraft.inertia_kgm2),
            boresight_body=tuple(args.spacecraft.boresight_body),
        ),
        controller=ControllerConfig(
            kp=args.controller.kp,
            kd=args.controller.kd,
            ki=args.controller.ki,
            integral_limit=args.controller.integral_limit,
        ),
        rwa=ReactionWheelConfig(
            wheel_inertia_kgm2=(rw_inertia, rw_inertia, rw_inertia),
            torque_max_nm=(rw_max_torque, rw_max_torque, rw_max_torque),
            wheel_speed_max_rads=(rw_max_speed, rw_max_speed, rw_max_speed),
            momentum_max_nms=(rw_max_momentum, rw_max_momentum, rw_max_momentum),
            command_lag_tau_s=args.reaction_wheel.command_lag_tau,
            viscous_friction_nms=args.reaction_wheel.friction_viscous,
            coulomb_friction_nm=args.reaction_wheel.friction_coulomb,
        ),
        output=OutputConfig(
            output_dir=args.output.output_dir,
            run_name=args.output.run_name,
            render_video=args.output.render_video,
            camera_width=args.output.camera_width,
            camera_height=args.output.camera_height,
            camera_fov_deg=args.output.camera_fov_deg,
            render_fps=render_frequency,
            spp=args.output.spp,
            use_integrator_mask=args.output.use_integrator_mask,
        ),
    )
    
    # Build RLEnvironmentConfig
    rl_cfg = RLEnvironmentConfig(
        camera_width=args.rl_environment.camera_width,
        camera_height=args.rl_environment.camera_height,
        camera_fov_deg=args.rl_environment.camera_fov_deg,
        max_steps=args.rl_environment.max_steps,
        max_steps_without_asteroid=args.rl_environment.max_steps_without_asteroid,
        asteroid_visibility_threshold=args.rl_environment.asteroid_visibility_threshold,
        torque_scale_nm=args.rl_environment.torque_scale_nm,
        gyro_history_length=args.rl_environment.gyro_history_length,
        render_at_frequency=args.rl_environment.render_at_frequency,
        randomize_on_reset=args.rl_environment.randomize_on_reset,
        periapsis_radius_scale_range=tuple(args.rl_environment.periapsis_radius_scale_range),
        external_angle_offset_deg_range=tuple(args.rl_environment.external_angle_offset_deg_range),
        true_anomaly0_offset_deg_range=tuple(args.rl_environment.true_anomaly0_offset_deg_range),
        inbound_ra_offset_deg_range=tuple(args.rl_environment.inbound_ra_offset_deg_range),
        inbound_dec_offset_deg_range=tuple(args.rl_environment.inbound_dec_offset_deg_range),
        bplane_angle_offset_deg_range=tuple(args.rl_environment.bplane_angle_offset_deg_range),
    )
    
    env = AsteroidTrackingEnv(flyby_config=flyby_cfg, rl_config=rl_cfg)
    return env


def train(
    args: TrainArgs,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
):
    """Main training loop.
    
    Parameters
    ----------
    args : TrainArgs
        Training arguments.
    output_dir : Path
        Directory to save outputs and logs.
    checkpoint_path : Optional[Path]
        Path to checkpoint to resume from, if any.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tensorboard logging
    log_dir = output_dir / args.logging.tensorboard_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    
    # Save resolved configuration as an artifact for reproducibility.
    config_path = output_dir / "config_snapshot.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(repr(args))
    logger.info(f"Resolved config snapshot saved to {config_path}")
    
    # Setup device
    device = torch.device(args.device.device)
    logger.info(f"Training on device: {device}")
    
    if args.reproducibility.deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception as exc:
            logger.warning(f"Could not enable fully deterministic PyTorch mode: {exc}")
    
    # Set seeds
    seed = args.reproducibility.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environments
    num_envs = max(1, int(args.training.num_envs))
    logger.info(f"Creating {num_envs} environment(s)...")
    envs = [create_environment(args) for _ in range(num_envs)]
    current_obs = [env.reset()[0] for env in envs]
    episode_returns = [0.0 for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    recent_episode_returns: list[float] = []
    recent_episode_lengths: list[int] = []
    
    # Create parallel environment executor
    env_executor = ParallelEnvExecutor(envs, max_workers=num_envs)
    
    # Create networks
    logger.info("Creating policy network...")
    policy_net = ViTGyroPolicy(
        image_size=args.rl_environment.camera_height,
        vit_model=args.network.vit_model,
        vit_pretrained=args.network.vit_pretrained,
        vit_freeze_depth=args.network.vit_freeze_depth,
        temporal_attention_heads=args.network.temporal_attention_heads,
        temporal_attention_dim=args.network.temporal_attention_dim,
        hidden_dim=args.network.hidden_dim,
        action_dim=3,
        action_std_init=args.network.action_std_init,
    ).to(device)
    
    # PPO modules
    actor, critic = _build_ppo_modules(policy_net, action_dim=3, device=device)
    ppo_loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=args.training.ppo_clip_ratio,
        entropy_coeff=args.training.entropy_coeff,
        critic_coeff=args.training.value_coeff,
        normalize_advantage=True,
    )
    ppo_loss.make_value_estimator(
        ValueEstimators.GAE,
        gamma=args.training.gamma,
        lmbda=args.training.gae_lambda,
        deactivate_vmap=True,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=args.training.learning_rate,
    )

    step_count = 0
    
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
    episode_count = 0
    best_return = -np.inf
    steps_without_improvement = 0
    
    logger.info("Starting training...")
    
    try:
        while step_count < args.training.max_steps:
            rollout_steps = int(args.training.steps_per_rollout)
            batch_actions = []
            batch_action_log_probs = []
            batch_dones = []
            batch_terminated = []
            batch_observations = []
            batch_next_observations = []
            batch_next_rewards = []
            
            for _ in range(rollout_steps):
                observation_td = _stack_observations(current_obs, device)
                with torch.no_grad():
                    dist = actor.get_dist(observation_td)
                    actions = dist.rsample()
                    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                    action_log_probs = dist.log_prob(actions).unsqueeze(-1)
                
                # Step all environments in parallel
                actions_np = actions.detach().cpu().numpy()
                observations, rewards, terminateds, truncateds, infos = env_executor.step_all(actions_np)
                
                for env_idx in range(num_envs):
                    obs = observations[env_idx]
                    reward = rewards[env_idx]
                    terminated = terminateds[env_idx]
                    truncated = truncateds[env_idx]
                    done = bool(terminated or truncated)
                    
                    batch_observations.append(current_obs[env_idx])
                    batch_next_observations.append(obs)
                    batch_actions.append(actions[env_idx].detach())
                    batch_action_log_probs.append(action_log_probs[env_idx].detach())
                    batch_next_rewards.append(torch.tensor([reward], device=device, dtype=torch.float32))
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
                        obs, _ = envs[env_idx].reset()
                    
                    current_obs[env_idx] = obs
                
            num_transitions = len(batch_actions)
            if num_transitions == 0:
                break
            
            batch = TensorDict(
                {
                    "observation": TensorDict(
                        {
                            "image": _stack_image_batch(batch_observations, device),
                            "gyro_history": _stack_gyro_batch(batch_observations, device),
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
                                    "image": _stack_image_batch(batch_next_observations, device),
                                    "gyro_history": _stack_gyro_batch(batch_next_observations, device),
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
            for epoch in range(int(args.training.num_epochs)):
                for start in range(0, num_transitions, int(args.training.batch_size)):
                    batch_idx = shuffled_indices[start : start + int(args.training.batch_size)]
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
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.training.grad_clip_norm)
                    optimizer.step()
                    total_loss += loss.item()
            
            step_count += num_transitions
            avg_return = float(np.mean(recent_episode_returns)) if recent_episode_returns else 0.0
            avg_length = float(np.mean(recent_episode_lengths)) if recent_episode_lengths else 0.0
            
            if step_count % args.logging.log_frequency == 0:
                writer.add_scalar("train/step", step_count, step_count)
                writer.add_scalar("train/ppo/loss", total_loss / max(1, int(args.training.num_epochs) * (num_transitions // int(args.training.batch_size))), step_count)
                writer.add_scalar("train/episode/return", avg_return, step_count)
                writer.add_scalar("train/episode/length", avg_length, step_count)
                logger.info(
                    f"Step {step_count}/{args.training.max_steps} | avg_return={avg_return:.3f} | avg_length={avg_length:.1f}"
                )
            
            if step_count % args.training.checkpoint_frequency == 0:
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
    train(args, args.output_dir, checkpoint_path=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
