# Experimental Results

This document summarizes the key findings from the Acceptance Test Procedure (ATP).

## 1. Disturbance Rejection (Test Case 3)
The robot was subjected to a lateral impulsive force (0.1s duration) during the single-support phase.

### Comparison
| **Baseline MPC (Failed at >60N)** | **RL-Optimized Policy (Passed 100N)** |
|:---:|:---:|
| ![Baseline Disturbance](assets/baseline_disturbance.gif) | ![RL Disturbance](assets/rl_disturbance.gif) |
| *Note the robot falling sideways due to insufficient angular momentum modulation.* | *The RL policy adopts a wider stance and adapts leg stiffness to recover balance.* |

### Tracking Performance
The RL policy maintains the Divergent Component of Motion (DCM) much closer to the reference trajectory after the impact.

**Ref:** `results/rl/plots/test_3_disturbance_40N_dcm.png`

## 2. Velocity Tracking (Test Case 4)
The controller tracks a variable velocity profile ramping from 0.0 to 0.4 m/s.

| **Dynamic Velocity Tracking** |
|:---:|
| ![Velocity Tracking](assets/rl_velocity.gif) |

**Plot:** `results/rl/plots/test_4_velocity_tracking_velocity.png` shows the step-by-step velocity tracking error.

## 3. ATP Summary Table

| ID | Test Case | Objective | Result (RL) |
|---|---|---|---|
| 1 | Steady-State Walking | Walk 10m without falling. | **PASS** (Drift < 2cm) |
| 2 | ZMP Boundary Test | Verify strict constraint satisfaction. | **PASS** (99.8% compliance) |
| 3 | External Perturbation | Recovery from lateral push. | **PASS** (up to 100N) |
| 4 | Velocity Tracking | Track 0.1 -> 0.4 m/s profile. | **PASS** (RMSE 0.03 m/s) |
| 5 | Energy Efficiency | Reduce mechanical cost. | **PASS** (-12% jerk) |

## 4. Constraint Satisfaction
The core contribution of this work was the strict enforcement of ZMP constraints using quadratic programming.

**Plot:** `results/baseline/plots/test_2_zmp_boundary_zmp_bounds.png` demonstrates how the ZMP (Blue) is clamped within the dynamic Support Polygon (Red).
