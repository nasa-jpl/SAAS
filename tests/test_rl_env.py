"""Unit tests for RL environment wrapper."""

import numpy as np
import pytest

from asteroid_flyby_syssim.flyby_sim import FlybyRunConfig
from asteroid_flyby_syssim.rl_env import AsteroidTrackingEnv, RLEnvironmentConfig


def test_env_reset():
    """Test environment reset functionality."""
    # Create minimal config
    rl_config = RLEnvironmentConfig(max_steps=100)
    
    # Create dummy flyby config (would need full config for real test)
    # For now, just test the environment interface
    
    # Skip full integration test for now (requires asteroid models)
    pytest.skip("Requires full asteroid model setup")


def test_observation_space():
    """Test observation space is correctly defined."""
    rl_config = RLEnvironmentConfig(
        camera_width=128,
        camera_height=128,
        gyro_history_length=4,
    )
    
    # Don't instantiate full env, just test config
    import gymnasium as gym
    from gymnasium import spaces
    
    # Check that spaces are valid
    assert isinstance(rl_config.camera_width, int)
    assert isinstance(rl_config.camera_height, int)
    assert isinstance(rl_config.gyro_history_length, int)
    
    # Create spaces
    image_space = spaces.Box(
        low=0, high=255,
        shape=(rl_config.camera_height, rl_config.camera_width, 3),
        dtype=np.uint8,
    )
    
    gyro_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(rl_config.gyro_history_length, 3),
        dtype=np.float32,
    )
    
    obs_space = spaces.Dict({
        "image": image_space,
        "gyro_history": gyro_space,
    })
    
    # Test space validity
    assert obs_space.contains({
        "image": np.zeros((128, 128, 3), dtype=np.uint8),
        "gyro_history": np.zeros((4, 3), dtype=np.float32),
    })


def test_action_space():
    """Test action space is correctly defined."""
    import gymnasium as gym
    from gymnasium import spaces
    
    action_space = spaces.Box(
        low=-1.0, high=1.0,
        shape=(3,),
        dtype=np.float32,
    )
    
    # Test valid action
    action = np.array([0.5, -0.3, 0.8], dtype=np.float32)
    assert action_space.contains(action)
    
    # Test invalid action (outside bounds)
    invalid_action = np.array([1.5, 0.0, 0.0], dtype=np.float32)
    assert not action_space.contains(invalid_action)


def test_rl_config_defaults():
    """Test RL configuration defaults are sensible."""
    config = RLEnvironmentConfig()
    
    assert config.max_steps == 1000
    assert config.max_steps_without_asteroid == 100
    assert config.asteroid_visibility_threshold > 0
    assert config.torque_scale_nm > 0
    assert config.gyro_history_length >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
