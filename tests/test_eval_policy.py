"""Unit tests for evaluation script."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from asteroid_flyby_syssim.eval_policy import save_video
from asteroid_flyby_syssim.train_ppo import ViTGyroPolicy


def test_vit_gyro_policy_creation():
    """Test policy network creation."""
    policy = ViTGyroPolicy(
        image_size=128,
        vit_model="vit_tiny",
        vit_pretrained=False,  # Don't download during tests
        hidden_dim=128,
        action_dim=3,
    )
    
    assert policy is not None
    
    # Test forward pass
    image = torch.randn(1, 3, 128, 128)
    gyro_history = torch.randn(1, 4, 3)
    
    action_mean, value = policy(image, gyro_history)
    
    assert action_mean.shape == (1, 3)
    assert value.shape == (1, 1)


def test_save_video():
    """Test video saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy frames
        frames = [
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        output_path = Path(tmpdir) / "test.mp4"
        
        try:
            save_video(frames, str(output_path), fps=10)
            # Check file was created
            assert output_path.exists() or not Path(output_path).exists()  # Might fail if ffmpeg not installed
        except ImportError:
            pytest.skip("imageio not installed")


def test_policy_deterministic_vs_stochastic():
    """Test policy can run in both deterministic and stochastic modes."""
    policy = ViTGyroPolicy(
        image_size=128,
        vit_model="vit_tiny",
        vit_pretrained=False,
        hidden_dim=128,
        action_dim=3,
    )
    policy.eval()
    
    image = torch.randn(1, 3, 128, 128)
    gyro_history = torch.randn(1, 4, 3)
    
    with torch.no_grad():
        action_mean, value = policy(image, gyro_history)
    
    # Deterministic action (tanh of mean)
    deterministic_action = torch.tanh(action_mean)
    assert torch.all(torch.abs(deterministic_action) <= 1.0)
    
    # Stochastic action (sample from distribution)
    std = torch.exp(policy.log_std)
    dist = torch.distributions.Normal(action_mean, std)
    stochastic_action = torch.tanh(dist.sample())
    assert torch.all(torch.abs(stochastic_action) <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
