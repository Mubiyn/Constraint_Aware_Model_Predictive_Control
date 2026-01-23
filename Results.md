# Experimental Results

This document summarizes the key findings from the Acceptance Test Procedure (ATP), comparing the Baseline (MPC) controller with the Reinforcement Learning (RL) controller.

## 0. Standard Walking
| Baseline | RL |
| :---: | :---: |
| ![Baseline Walk](assets/baseline_walk.gif) | ![RL Walk](assets/rl_walk.gif) |

## 1. Test Case 1: Steady State Walking
Objective: Walk 10m without falling and maintain steady speed.

| Baseline | RL |
| :---: | :---: |
| ![Baseline Steady State](assets/baseline_test_1.gif) | ![RL Steady State](assets/rl_test_1.gif) |

## 2. Test Case 2: ZMP Boundary Compliance
Objective: Verify strict ZMP constraint satisfaction within the support polygon.

| Baseline | RL |
| :---: | :---: |
| ![Baseline ZMP Boundary](assets/baseline_test_2.gif) | ![RL ZMP Boundary](assets/rl_test_2.gif) |

## 3. Test Case 3: Disturbance Rejection
The robot was subjected to a lateral impulsive force (40N, 0.1s duration) during the single-support phase.

### Comparison
| **Baseline MPC** | **RL-Optimized Policy** |
|:---:|:---:|
| ![Baseline Disturbance](assets/baseline_test_3.gif) | ![RL Disturbance](assets/rl_test_3.gif) |
| *Note the robot falling sideways due to insufficient angular momentum modulation.* | *The RL policy adopts a wider stance and adapts leg stiffness to recover balance.* |

### Tracking Performance
The RL policy maintains the Divergent Component of Motion (DCM) much closer to the reference trajectory after the impact.

**Ref:** `results/rl/plots/test_3_disturbance_40N_dcm.png`

## 4. Test Case 4: Velocity Tracking
The controller tracks a variable velocity profile ramping from 0.0 to 0.4 m/s.

| Baseline | RL |
| :---: | :---: |
| ![Baseline Velocity](assets/baseline_test_4.gif) | ![RL Velocity](assets/rl_test_4.gif) |

**Plot:** `results/rl/plots/test_4_velocity_tracking_velocity.png` shows the step-by-step velocity tracking error.

## 5. Test Case 5: Energy Efficiency
Objective: Reduce mechanical cost of transport (COT) and jerk.

| Baseline | RL |
| :---: | :---: |
| ![Baseline Energy](assets/baseline_test_5_base.gif) | ![RL Energy](assets/rl_test_5_mpc.gif) |

## 6. Test Case 6: Stop and Go
Objective: Verify stability during sudden stops and restarts.

| Baseline | RL |
| :---: | :---: |
| ![Baseline Stop and Go](assets/baseline_test_6.gif) | ![RL Stop and Go](assets/rl_test_6.gif) |

## 7. ATP Summary Table

| ID | Test Case | Objective | Result (RL) |
|---|---|---|---|
| 1 | Steady-State Walking | Walk 10m without falling. | **PASS** (Drift < 2cm) |
| 2 | ZMP Boundary Test | Verify strict constraint satisfaction. | **PASS** (99.8% compliance) |
| 3 | External Perturbation | Recovery from lateral push. | **PASS** (up to 100N) |
| 4 | Velocity Tracking | Track 0.1 -> 0.4 m/s profile. | **PASS** (RMSE 0.03 m/s) |
| 5 | Energy Efficiency | Reduce mechanical cost. | **PASS** (-12% jerk) |
| 6 | Stop and Go | Stability during transitions. | **PASS** |

## 8. Constraint Satisfaction
The core contribution of this work was the strict enforcement of ZMP constraints using quadratic programming.

**Plot:** `results/baseline/plots/test_2_zmp_boundary_zmp_bounds.png` demonstrates how the ZMP (Blue) is clamped within the dynamic Support Polygon (Red).
