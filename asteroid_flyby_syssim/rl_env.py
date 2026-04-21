"""Gymnasium environment wrapper for asteroid tracking control task.

The environment provides:
- Observations: Camera frames (128×128 RGB) + gyroscope measurements (4-step history)
- Actions: 3D normalized torque commands [-1, 1]
- Rewards: Negative for asteroid not in view, negative based on centering error when visible
- Termination: Asteroid out of view for >100 steps or time limit exceeded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from syssim.core import NodeSystem

from .asteroid_camera import NodeAsteroidCamera, NodeFrameCollector
from .flyby_sim import build_flyby_rl_system, FlybyRunConfig, FlybyArtifacts
from .gyroscope import GyroscopeConfig


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _body_axis_to_inertial(q_bi_wxyz: np.ndarray, axis_body: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(_quat_wxyz_to_xyzw(q_bi_wxyz))
    return rotation.apply(np.asarray(axis_body, dtype=float))


def _centering_error_deg(
    q_bi_wxyz: np.ndarray,
    position_sc_i_m: np.ndarray,
    boresight_body: np.ndarray,
) -> float:
    boresight_i = _body_axis_to_inertial(q_bi_wxyz, boresight_body)
    target_i = -np.asarray(position_sc_i_m, dtype=float)

    boresight_norm = np.linalg.norm(boresight_i)
    target_norm = np.linalg.norm(target_i)
    if boresight_norm < 1e-9 or target_norm < 1e-9:
        return 180.0

    cos_angle = np.clip(np.dot(boresight_i, target_i) / (boresight_norm * target_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass
class RLEnvironmentConfig:
    """Configuration for RL environment.
    
    Attributes
    ----------
    camera_width : int
        Camera frame width in pixels.
    camera_height : int
        Camera frame height in pixels.
    camera_fov_deg : float
        Camera field of view in degrees.
    max_steps : int
        Maximum steps per episode.
    max_steps_without_asteroid : int
        Maximum consecutive steps with asteroid out of view before termination.
    asteroid_visibility_threshold : float
        Minimum fraction of asteroid pixels in frame to consider "visible".
    torque_scale_nm : float
        Maximum torque command magnitude [N⋅m].
    gyro_history_length : int
        Number of gyro measurements to stack in observation.
    render_at_frequency : int
        Render camera image at this frequency (Hz). If None, render every step.
        E.g., render_at_frequency=10 means render every 0.1 seconds.
    """
    
    camera_width: int = 128
    camera_height: int = 128
    camera_fov_deg: float = 50.0
    max_steps: int = 1000
    max_steps_without_asteroid: int = 100
    asteroid_visibility_threshold: float = 0.01  # 1% of frame
    torque_scale_nm: float = 1.0
    gyro_history_length: int = 4
    render_at_frequency: Optional[int] = None


class AsteroidTrackingEnv(gym.Env):
    """Gymnasium environment for asteroid visual acquisition and tracking.
    
    The agent observes rendered camera images and gyroscope measurements,
    and outputs 3-axis torque commands to control spacecraft attitude.
    
    The reward function incentivizes acquiring the asteroid visually and
    then keeping it centered in the camera frame.
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        flyby_config: FlybyRunConfig,
        rl_config: RLEnvironmentConfig = None,
        seed: Optional[int] = None,
    ):
        """Initialize RL environment.
        
        Parameters
        ----------
        flyby_config : FlybyRunConfig
            Base flyby simulation configuration.
        rl_config : RLEnvironmentConfig, optional
            RL-specific configuration. If None, uses defaults.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.flyby_config = flyby_config
        self.rl_config = rl_config or RLEnvironmentConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        
        # Observation space: Dict with "image" and "gyro_history"
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.rl_config.camera_height, self.rl_config.camera_width, 3),
                dtype=np.uint8,
            ),
            "gyro_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.rl_config.gyro_history_length, 3),
                dtype=np.float32,
            ),
        })
        
        # Action space: 3D normalized torque [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        
        # Simulation state
        self._system: Optional[NodeSystem] = None
        self._artifacts: Optional[FlybyArtifacts] = None
        self._step_count: int = 0
        self._steps_without_asteroid: int = 0
        self._gyro_history: list[np.ndarray] = []
        self._last_frame: Optional[np.ndarray] = None
        self._asteroid_visible_count: int = 0
        self._error_angle_history: list[float] = []

        self._rl_action_input: Optional[Any] = None
        self._gyro_output: Optional[Any] = None
        self._camera_image_output: Optional[Any] = None
        self._camera_mask_output: Optional[Any] = None
        self._camera_visible_output: Optional[Any] = None
        self._attitude_output: Optional[Any] = None
        self._position_output: Optional[Any] = None
        
        # For reward computation
        self._info = {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """Reset environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional reset options (unused for now).
        
        Returns
        -------
        observation : dict
            Initial observation dict with "image" and "gyro_history".
        info : dict
            Info dict with metadata.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._system is not None:
            try:
                self._system.finalize()
            except Exception:
                pass

        # Rebuild simulation with the RL-specific system builder.
        self._system, self._artifacts = build_flyby_rl_system(self.flyby_config)

        # Initialize simulation nodes and populate the first observation.
        self._system.initialize()

        self._rl_action_input = self._system.get_node("rl_action_input")
        self._gyro_output = self._system.get_node("rl_gyro_output")
        self._camera_image_output = self._system.get_node("rl_camera_image_output")
        self._camera_mask_output = self._system.get_node("rl_camera_mask_output")
        self._camera_visible_output = self._system.get_node("rl_camera_visible_output")
        self._attitude_output = self._system.get_node("rl_attitude_output")
        self._position_output = self._system.get_node("rl_position_output")

        if self._rl_action_input is not None:
            self._rl_action_input.value = np.zeros(3, dtype=float)

        self._system.step(0.0)
        
        # Reset tracking variables
        self._step_count = 0
        self._steps_without_asteroid = 0
        self._gyro_history = []
        self._error_angle_history = []
        self._asteroid_visible_count = 0
        self._info = {}
        
        # Initialize gyro history with zeros
        for _ in range(self.rl_config.gyro_history_length):
            self._gyro_history.append(np.zeros(3, dtype=np.float32))
        
        # Get initial observation
        obs = self._get_observation()
        return obs, self._info
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step.
        
        Parameters
        ----------
        action : ndarray
            Normalized torque command [-1, 1] in body frame, shape (3,).
        
        Returns
        -------
        observation : dict
            Observation dict.
        reward : float
            Step reward.
        terminated : bool
            True if episode terminated (time limit or asteroid lost).
        truncated : bool
            False (not used).
        info : dict
            Additional info: asteroid_visible, error_angle_deg, etc.
        """
        # Convert normalized action to physical torque [N⋅m]
        tau_cmd = np.asarray(action, dtype=float) * self.rl_config.torque_scale_nm

        if self._rl_action_input is not None:
            self._rl_action_input.value = tau_cmd

        # Advance the simulation by one timestep.
        self._system.step(self.flyby_config.sim.sim_dt)

        # Get current observation
        obs = self._get_observation()
        
        # Compute reward
        reward, info = self._compute_reward()
        self._info = info
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False
        
        self._step_count += 1
        
        return obs, reward, terminated, truncated, self._info
    
    def _get_observation(self) -> dict:
        """Get current observation from simulation.
        
        Returns
        -------
        obs : dict
            Dict with "image" (uint8 [0, 255]) and "gyro_history" (float32).
        """
        gyro_meas = None
        if self._gyro_output is not None:
            gyro_meas = self._gyro_output.value
        if gyro_meas is None:
            gyro_meas = np.zeros(3, dtype=np.float32)
        else:
            gyro_meas = np.asarray(gyro_meas, dtype=np.float32)
        
        # Update gyro history (FIFO)
        self._gyro_history.append(gyro_meas)
        if len(self._gyro_history) > self.rl_config.gyro_history_length:
            self._gyro_history.pop(0)
        
        # Pad gyro history if not full yet
        while len(self._gyro_history) < self.rl_config.gyro_history_length:
            self._gyro_history.insert(0, np.zeros(3, dtype=np.float32))
        
        gyro_history_arr = np.array(self._gyro_history, dtype=np.float32)
        
        # Get camera image
        image = self._get_camera_image()
        if image is None:
            image = np.zeros(
                (self.rl_config.camera_height, self.rl_config.camera_width, 3),
                dtype=np.uint8
            )
        
        obs = {
            "image": image,
            "gyro_history": gyro_history_arr,
        }
        return obs
    
    def _get_camera_image(self) -> Optional[np.ndarray]:
        """Get rendered camera image from simulation.
        
        Returns
        -------
        image : ndarray, optional
            RGB image [height, width, 3] in uint8 [0, 255], or None if not available.
        """
        image = None
        if self._camera_image_output is not None:
            image = self._camera_image_output.value
        if image is not None:
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            return image
        
        return None
    
    def _compute_reward(self) -> tuple[float, dict]:
        """Compute step reward and info dict.
        
        Reward structure:
        - If asteroid not visible: -1.0
        - If visible: -0.1 × error_angle_degrees + 0.05 (visibility bonus)
        
        Returns
        -------
        reward : float
            Step reward.
        info : dict
            Dict with "asteroid_visible", "error_angle_deg", etc.
        """
        asteroid_visible, error_angle_deg = self._compute_asteroid_visibility()
        
        info = {
            "asteroid_visible": asteroid_visible,
            "error_angle_deg": error_angle_deg,
        }
        
        if not asteroid_visible:
            reward = -1.0
            self._steps_without_asteroid += 1
        else:
            reward = -0.1 * error_angle_deg + 0.05
            self._steps_without_asteroid = 0
            self._asteroid_visible_count += 1
        
        self._error_angle_history.append(error_angle_deg)
        
        return reward, info
    
    def _compute_asteroid_visibility(self) -> tuple[bool, float]:
        """Compute if asteroid is visible and centering error angle.
        
        Returns
        -------
        visible : bool
            True if asteroid is in camera frame above threshold.
        error_angle_deg : float
            Angle between camera boresight and true asteroid center [degrees].
            Returns 180.0 if not visible.
        """
        if self._system is None:
            return False, 180.0

        position = None
        attitude = None
        if self._position_output is not None:
            position = self._position_output.value
        if self._attitude_output is not None:
            attitude = self._attitude_output.value
        if position is None or attitude is None:
            return False, 180.0

        error_angle_deg = _centering_error_deg(
            q_bi_wxyz=attitude,
            position_sc_i_m=np.asarray(position, dtype=float),
            boresight_body=np.asarray(self.flyby_config.spacecraft.boresight_body, dtype=float),
        )

        asteroid_visible = None
        visibility_fraction = None

        if self._camera_mask_output is not None:
            asteroid_mask = self._camera_mask_output.value
            if asteroid_mask is not None:
                mask = np.asarray(asteroid_mask)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                if mask.dtype.kind in {"u", "i"}:
                    visibility_fraction = float(mask.mean() / 255.0)
                else:
                    visibility_fraction = float(np.clip(mask, 0.0, 1.0).mean())
                asteroid_visible = visibility_fraction >= self.rl_config.asteroid_visibility_threshold

        if asteroid_visible is None and self._camera_visible_output is not None:
            visible_value = self._camera_visible_output.value
            if visible_value is not None:
                asteroid_visible = bool(visible_value)

        if asteroid_visible is None:
            asteroid_visible = error_angle_deg <= (self.flyby_config.output.camera_fov_deg * 0.5)

        self._info["asteroid_visibility_fraction"] = visibility_fraction
        return asteroid_visible, error_angle_deg
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate.
        
        Returns
        -------
        terminated : bool
            True if episode reached terminal condition.
        """
        # Terminate if max steps reached
        if self._step_count >= self.rl_config.max_steps:
            return True
        
        # Terminate if asteroid lost for too long
        if self._steps_without_asteroid >= self.rl_config.max_steps_without_asteroid:
            return True
        
        return False
    
    def render(self):
        """Render environment (not implemented for RL training)."""
        pass
    
    def close(self):
        """Close environment and clean up."""
        if self._system is not None:
            # Finalize simulation
            try:
                self._system.finalize()
            except Exception:
                pass
            finally:
                self._system = None
                self._artifacts = None
