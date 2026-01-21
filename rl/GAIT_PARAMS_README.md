# Gait Parameter RL for Speed Optimization

This implements **Approach 1: Adaptive Gait Parameters** to speed up MPC walking through learned parameter tuning.

## Overview

The gait parameter policy dynamically adjusts 4 key parameters to balance speed and stability:
- **Stride length** (0.8-1.5× base: 0.096m - 0.18m)
- **Single support time** (0.6-1.0× base: 0.57s - 0.95s)
- **Double support time** (0.5-1.0× base: 0.75s - 1.5s)
- **ZMP safety margin** (0.5-1.5× base: 0.005m - 0.015m)

## Training

Train a gait parameter policy with CEM:

```bash
python v2/main_mpc.py --train-residual-cem --no-gui \
    --cem-iters 15 --cem-pop 24 --cem-elite 0.25 \
    --cem-steps 800 --cem-action-dim 4 --cem-policy-mode gait \
    --cem-output v2/rl/gait_speed_policy.npz
```

**Recommended hyperparameters:**
- Iterations: 15-20 (more exploration needed than weight tuning)
- Population: 24-32 (larger search space)
- Elite fraction: 0.25
- Steps: 600-1000 (enough to measure speed gains)
- Action dim: 4 (stride, ss_time, ds_time, margin scales)

## Evaluation

Test the trained gait policy:

```bash
# Baseline (conservative MPC)
python v2/main_mpc.py --no-gui --max-steps 1000 --print-every 200

# With gait parameter policy
python v2/main_mpc.py --no-gui --max-steps 1000 --print-every 200 \
    --policy v2/rl/gait_speed_policy.npz --policy-mode gait \
    --gait-update-steps 10
```

## Observation Space (17D)

The policy receives:
1. **State (6D)**: CoM position, velocity, DCM
2. **ZMP bounds (4D)**: Current support polygon limits
3. **Phase (1D)**: Progress through current gait phase
4. **Walking state (5D)**: One-hot encoded (INIT, DS, SS_LEFT, SS_RIGHT, END)
5. **Stability metrics (3D)**:
   - DCM tracking error magnitude
   - Average forward speed (recent window)
   - ZMP violation rate (recent window)

## Reward Function

Optimized for speed with safety constraints:
```python
R = 50.0 * avg_speed              # Primary: maximize speed
  + 100.0 * zmp_compliance        # Constraint: stay in support polygon
  - 100.0 * max_zmp_violation     # Heavy penalty for violations
  - 20.0 * avg_dcm_error          # Stability requirement
  - 10.0 * forward_vel_mae        # Smooth velocity tracking
  - 0.5 * CoT                     # Minor: energy efficiency
  - 200.0 * fell                  # Critical: no falls
```

## Expected Results

- **Speed improvement**: 2-3× baseline (0.05 → 0.12-0.15 m/s)
- **Training time**: 2-4 hours (15-20 iterations × 24 pop × ~30s)
- **Safety**: High (MPC still enforces all constraints)
- **Stability**: Comparable ZMP compliance (~99%)

## How It Works

1. **Policy acts at 10Hz** (every 10 sim steps)
2. **Outputs 4 normalized values** [-1, 1] mapped to parameter ranges
3. **Parameters update immediately**:
   - Stride affects future step planning
   - Timing updates state machine transitions
   - Margin updates MPC constraints
4. **MPC tracks the adapted gait** at 100Hz

## Advantages

✅ Sample efficient (parameters change slowly)  
✅ Safe (MPC enforces physics constraints)  
✅ Interpretable (4 intuitive parameters)  
✅ Sim2real friendly (parameter transfer is robust)  
✅ Fast training (linear policy, simple action space)

## Comparison to Other Modes

| Mode | Action Dim | Updates | Primary Goal |
|------|-----------|---------|--------------|
| `residual` | 2D | 100Hz | ZMP adjustment |
| `weights` | 3D | 20 steps | MPC tuning |
| **`gait`** | **4D** | **10 steps** | **Speed** |

## Next Steps

If gait parameters plateau (~0.15 m/s):
1. Try **Approach 2: Learned Foot Placement** for 0.15-0.20+ m/s
2. Add terrain/disturbance adaptation
3. Explore running gaits (requires contact modeling)
