# Quick Start: RL Asteroid Tracking Training

This is a 5-minute quick start guide to get training your first policy.

## Prerequisites

1. Clone/navigate to asteroid-flyby-syssim:
```bash
cd /home/j/Code/sync/saas/asteroid-flyby-syssim
```

2. Install package with RL dependencies:
```bash
pip install -e .
```

Verify installation:
```bash
python -c "import torch; import gymnasium; import timm; print('✓ All dependencies OK')"
```

## Step 1: Check Configuration

The training config is ready at:
```
configs/rl_training_config.toml
```

Key settings (you can customize):
```toml
[training]
max_steps = 1_000_000        # Total training steps (adjust for time budget)
num_envs = 8                 # Parallel environments (reduce to 4 if GPU memory limited)
learning_rate = 1e-4         # Adam learning rate

[rl_environment]
max_steps = 1000             # Max steps per episode
asteroid_visibility_threshold = 0.01  # How much of asteroid must be visible
```

## Step 2: Start Training

```bash
# Start training (will take 8-12 hours on RTX 3080)
python -m asteroid_flyby_syssim.train_ppo \
    --config configs/rl_training_config.toml \
    --output-dir outputs/rl_training

# Output structure created:
# outputs/rl_training/
# ├── config.toml
# ├── logs/                    (TensorBoard logs)
# │   ├── events.out.tfevents.* 
# └── checkpoints/
#     ├── policy_step_10000.pt
#     ├── policy_step_20000.pt
#     └── ...
```

## Step 3: Monitor Training (in another terminal)

```bash
tensorboard --logdir outputs/rl_training/logs
# Then open browser: http://localhost:6006
```

Watch these metrics improve:
- **train/return** - Episode return (should increase)
- **train/loss_actor** - Policy loss (should decrease)
- **train/metrics/visibility** - Asteroid visibility % (should increase)

## Step 4: Evaluate Trained Policy

After training completes (or when checkpoint reaches good performance):

```bash
# Evaluate best checkpoint
python -m asteroid_flyby_syssim.eval_policy \
    --checkpoint outputs/rl_training/checkpoints/policy_step_100000.pt \
    --config configs/rl_training_config.toml \
    --num-episodes 10 \
    --output-dir outputs/rl_results

# Results:
# outputs/rl_results/
# ├── eval_results.csv              (per-episode metrics)
# └── eval_summary.json             (aggregate stats)
```

View results:
```bash
cat outputs/rl_results/eval_summary.json
```

Expected output (after good training):
```json
{
  "num_episodes": 10,
  "mean_return": -45.5,
  "std_return": 12.3,
  "mean_visibility": 0.62,
  "mean_error_deg": 8.5
}
```

## Step 5: Optional - Render Evaluation Videos

```bash
python -m asteroid_flyby_syssim.eval_policy \
    --checkpoint outputs/rl_training/checkpoints/policy_step_100000.pt \
    --config configs/rl_training_config.toml \
    --num-episodes 3 \
    --render \
    --output-dir outputs/rl_results

# Videos saved to:
# outputs/rl_results/
# ├── episode_001.mp4
# ├── episode_002.mp4
# └── episode_003.mp4
```

## Common Customizations

### Reduce Training Time (for quick test)

Edit `configs/rl_training_config.toml`:
```toml
[training]
max_steps = 100_000              # From 1_000_000 (10x faster)
checkpoint_frequency = 10_000    # More frequent saves

[simulation]
duration_seconds = 600.0         # Shorter episodes (10 min instead of 1 hr)
```

### Use GPU Memory Efficiently

```toml
[training]
num_envs = 4                     # Fewer parallel (from 8)
batch_size = 16                  # Smaller batch (from 32)

[network]
vit_model = "vit_tiny"          # Already minimal, can't reduce further
hidden_dim = 128                 # Smaller MLP (from 256)
```

### Run on CPU (for testing)

```toml
[device]
device = "cpu"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: torchrl` | `pip install pytorch-rl` |
| `CUDA out of memory` | Reduce `num_envs` to 4 or 2 |
| Training very slow | Run on GPU not CPU; use TensorFlow's NVIDIA GPU test |
| No TensorBoard output | Check `ls outputs/rl_training/logs/` - dir should exist |

## File Structure Created

After training:
```
outputs/
├── rl_training/                 (main output directory)
│   ├── config.toml              (saved config)
│   ├── logs/                    (TensorBoard logs)
│   │   └── events.out.tfevents.*
│   └── checkpoints/
│       ├── policy_step_10000.pt
│       ├── policy_step_20000.pt
│       └── ... (every 10k steps)
│
└── rl_results/                  (evaluation output)
    ├── eval_results.csv
    ├── eval_summary.json
    └── episode_*.mp4 (if --render)
```

## What's Happening Under the Hood

1. **Simulation**: Spacecraft with gyro performs hyperbolic asteroid flyby
2. **Camera**: Renders 128×128 RGB images (Mitsuba 3 ray tracer)
3. **Observations**: Image + 4-frame gyro history
4. **Policy**: Vision Transformer + temporal attention
5. **Actions**: 3D torque commands to spacecraft
6. **Reward**: +0.05 for visibility, -0.1 × centering_error
7. **Algorithm**: PPO with 8 parallel environments

## Next Steps

- [ ] Train policy to convergence (~12 hours)
- [ ] Evaluate learned policy on held-out trajectories
- [ ] Compare with ground-truth guidance controller
- [ ] Analyze failure cases (asteroid lost, oscillations)
- [ ] Ablation: ViT vs. CNN, with/without gyro history
- [ ] Domain randomization (different asteroids, trajectories)

## More Info

See `RL_TRAINING_GUIDE.md` for detailed documentation:
- Architecture details
- Configuration reference
- Debugging tips
- References

---

**Happy training! 🚀**
