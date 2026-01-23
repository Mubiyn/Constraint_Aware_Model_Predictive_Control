# Acceptance Test Plan: Constraint-Aware Model Predictive Control and Reinforcement Learning for Bipedal Locomotion

## 1. Introduction
Humanoid walking is a fundamentally unstable process requiring the generation of dynamically consistent trajectories for the Center of Mass (CoM) relative to the Zero-Moment Point (ZMP). Traditional Preview Control (the current baseline) operates as a batch process that utilizes a fixed ZMP reference to generate CoM trajectories. However, this approach lacks the ability to handle real-time physical constraints—such as actuator limits and footstep boundaries—leading to instability under external perturbations or uneven terrain.

This project implements a Model Predictive Control (MPC) framework to replace the baseline tracker. Unlike Preview Control, the proposed MPC performs iterative numerical optimization during runtime to satisfy multi-objective costs while respecting hard physical constraints. Furthermore, if the MPC phase achieves its primary benchmarks, the project will explore a Hybrid DRL–MPC approach to optimize the control policy against nonlinearities and terrain uncertainties that are difficult to model analytically.

## 2. System Description

### 2.1 Plant Model: Linear Inverted Pendulum (LIPM)
The robot is modeled as a concentrated mass at the CoM $(x_c, y_c)$ supported by massless legs at a constant height $z_c$. The dynamics are described by the second-order linear differential equations:
$$\ddot{x}_c = \frac{g}{z_c}(x_c - x_z)$$
$$\ddot{y}_c = \frac{g}{z_c}(y_c - y_z)$$
where $(x_z, y_z)$ denotes the ZMP position and $g$ is the gravitational acceleration.

### 2.2 Simulation Environment
- **Platform**: PyBullet Physics Engine.
- **Robot Model**: Talos Humanoid (URDF).
- **Control Frequency**: 100–200 Hz (Optimization cycles of 10–15 ms).
- **Language**: Python 3.10+ (using OSQP or CVXPY solvers).

## 3. Theoretical Basis

### 3.1 Baseline: ZMP Preview Control
The baseline minimizes a quadratic cost function over a finite horizon:
$$J = \sum_{k=0}^{N} (Q_e e_k^2 + x_k^T Q_x x_k + R \Delta u_k^2)$$
where $e_k$ is the ZMP tracking error and $\Delta u_k$ is the control input (CoM Jerk). This is a "feedback + integral + preview" law that follows a predefined footstep sequence but cannot adjust to unexpected disturbances.

### 3.2 Optimization: Model Predictive Control (MPC)
The proposed solution reformulates the problem as a Quadratic Program (QP) solved at each time step. The discrete-time state-space model for CoM jerk $u$ is formulated as:
$$\hat{x}_{k+1} = A \hat{x}_k + B u_k$$
$$p_{x,k} = C \hat{x}_k$$
With sampling time $T$, the matrices are defined as:
$$A = \begin{bmatrix} 1 & T & T^2/2 \\ 0 & 1 & T \\ 0 & 0 & 1 \end{bmatrix}, B = \begin{bmatrix} T^3/6 \\ T^2/2 \\ T \end{bmatrix}, C = \begin{bmatrix} 1 & 0 & -z_c/g \end{bmatrix}$$

### 3.3 Constraints and Reward Shaping
The MPC explicitly enforces the Balance Criterion:
$$p_{x,min} \leq p_{x,k} \leq p_{x,max}$$
where $p_{x,min}$ and $p_{x,max}$ define the support polygon (the foot area). The cost function is optimized to reduce CoM velocity fluctuations, which improves stability at higher walking speeds.

## 4. Test Environment Parameters
| Parameter | Value |
| :--- | :--- |
| CoM Height ($z_c$) | 0.8 meters |
| Sampling Time ($T$) | 0.005 - 0.01 seconds |
| Prediction Horizon ($N$) | 1.6 seconds (approx. 2 steps) |
| Gravity ($g$) | 9.81 $m/s^2$ |
| Solver | OSQP (sparse QP solver) |

## 5. Test Objectives
The ATP verifies that the controller:
- **Ensures Stability**: Maintains the ZMP within the support polygon at all times.
- **Reduces Energy Consumption**: Minimizes CoM jerk and velocity fluctuations.
- **Enhances Robustness**: Successfully rejects external lateral and sagittal disturbances.
- **Improves Motion Quality**: Eliminates discontinuity in CoM acceleration.

## 6. Acceptance Metrics
- **ZMP Deviation**: RMS error between actual ZMP and the center of the support foot.
- **Success Rate**: Percentage of walking cycles completed without falling (Base height $z < z_{threshold}$).
- **Cost of Transport (CoT)**: Calculated as $C_{et} = \frac{Energy}{Weight \times Distance}$.
- **Disturbance Threshold**: Maximum impulsive force (Newtons) the robot can withstand without losing balance.
- **Computation Time**: Mean time per optimization cycle (must be $< 15$ ms).

## 7. Test Cases
| ID | Test Case | Objective | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| 1 | Steady-State Walking | Verify basic locomotion. | Successful 10-meter walk without falling or foot slippage. |
| 2 | ZMP Boundary Test | Verify constraint enforcement. | ZMP never exits the foot polygon boundaries, even at max speed. |
| 3 | External Perturbation | Test robustness. | Recovery from a torso push of $\geq 40N$ for 0.1s during single-support phase. |
| 4 | Velocity Tracking | Measure tracking accuracy. | MAE of forward base velocity $\leq 0.02$ m/s. |
| 5 | Energy Efficiency | Compare to baseline. | $\geq 15\%$ reduction in mechanical cost of transport compared to Preview Control. |
| 6 | Stop-and-Go | Verify terminal constraints. | Robot successfully transitions to a full stop with $v = 0$ at the final foothold. |

## 8. Advanced Optimization Extension: Reinforcement Learning
If the MPC controller satisfies all "High" priority test cases, the project will extend into Residual Reinforcement Learning.

### 8.1 Hybrid Structure
The control input will be augmented by a learned residual:
$$u(t) = u_{mpc}(t) + u_{rl}(t)$$
The RL policy will be trained using Proximal Policy Optimization (PPO) in PyBullet.

### 8.2 Learning Objectives
- **Adaptive Weight Tuning**: The RL agent will learn to dynamically adjust the $Q$ and $R$ matrices of the MPC cost function based on perceived terrain noise.
- **Nonlinear Compensation**: The RL component will account for rotational dynamics and "soft contact" effects (foot-terrain interpenetration) that are neglected in the standard LIPM.

## 9. Test Outputs
- **Time-Series Logs**: CSV files containing CoM/ZMP trajectories, joint torques, and solver status.
- **Stability Plots**: Overlay of ZMP position vs. foot support polygons.
- **Energy Reports**: Comparative analysis of current draw and power consumption.
- **Visual Validation**: Video recording of PyBullet simulation rollouts.

## 10. Acceptance Criteria
The optimized mechatronic system is accepted if:
- The robot maintains stability under $2 \times$ higher disturbance forces than the baseline.
- ZMP constraints are strictly satisfied ($100\%$ compliance).
- Trajectory tracking MAE is reduced by at least $50\%$ relative to Preview Control.
- Real-time execution is confirmed at $\geq 100$ Hz.

## 11. Test Results (MPC Implementation)

### 11.1 Implementation Summary
The MPC controller was implemented using a **DCM (Divergent Component of Motion)** based approach for improved stability:

- **DCM Dynamics**: $\xi = x + \dot{x}/\omega$ where $\omega = \sqrt{g/z_c}$
- **Control Law**: $p_{zmp} = \xi - \frac{1}{\omega}(\xi_{ref} - \xi)$
- **MPC Formulation**: QP minimizing DCM tracking error subject to ZMP constraints

### 11.2 Test Case Results

| ID | Test Case | Result | Measured Value |
| :--- | :--- | :--- | :--- |
| 1 | Steady-State Walking | **PASS** | 10m walk completed (distance: 10.07m, fell: False) |
| 2 | ZMP Boundary Test | **PARTIAL** | ZMP compliance: 98.2% (max violation: 0.0575m) |
| 3 | External Perturbation | **PASS** | Recovered from 40N lateral push (fell: False) |
| 4 | Velocity Tracking | **FAIL** | Forward velocity MAE: 0.0327 m/s (target ≤ 0.02 m/s), avg v=0.0425 m/s |
| 5 | Energy Efficiency | **FAIL** | CoT MPC=1.0101, Baseline=1.0101 (reduction: 0.0%) |
| 6 | Stop-and-Go | **PASS** | Robot successfully stops at final position with stable CoM |

### 11.3 Key Metrics

| Metric | Target | Achieved | Status |
| :--- | :--- | :--- | :--- |
| Real-time Execution | ≥100 Hz | 100 Hz (~3.9ms avg solve time) | **PASS** |
| Disturbance Rejection | ≥40N push | 40N push survived (fell: False) | **PASS** |
| ZMP Compliance | 100% | 98.2% | **PARTIAL** |
| MPC Solve Time | <15ms | ~3.9ms avg, 30.5ms max | **PARTIAL** |

### 11.4 Detailed Disturbance Test Results

Tested with a 40N lateral push during the walking sequence:

```
40N Push:
  - Outcome: fell=False
  - Notes: System remained stable and completed the test sequence.
```

### 11.5 Known Limitations

1. **ZMP Constraint Compliance**: The ~98% compliance rate indicates the MPC occasionally produces ZMP commands outside the support polygon. This is compensated by DCM feedback control clipping the ZMP to bounds. Improvement strategies:
   - Tighten support polygon margins
   - Increase QP constraint penalties
   - Use soft constraints with slack variables

2. **Forward Velocity Tracking**: The robot exhibits slight lateral oscillation during walking, which is characteristic of DCM-based control. This could be reduced with:
   - Better DCM reference trajectory generation
   - Angular momentum regulation
   - Ankle torque optimization

### 11.6 Reinforcement Learning Extension Results (Gait Parameter Tuning)

This extension evaluates a lightweight RL-style optimization (CEM over a linear policy) that adjusts gait parameters (stride/timing/margins) at runtime while keeping the underlying DCM-MPC controller unchanged.

**Policy artifact**: `v2/rl/gait_cem_v2.npz`

**Evaluation (10-step walk, no GUI, eval mode)**:

| Controller | Distance (m) | Avg Speed (m/s) | ZMP Compliance (%) | Avg DCM Error (m) | Fell |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline MPC | 1.192 | 0.041 | 98.2 | 0.0218 | False |
| MPC + Gait Policy | 1.199 | 0.050 | 96.1 | 0.0233 | False |

**Interpretation**: The gait policy increases average speed for the 10-step sequence, at the cost of a modest reduction in ZMP constraint compliance.

## 12. Conclusion
This ATP provides a rigorous framework for validating an advanced MPC controller in a non-MATLAB environment. By focusing on numerical optimization and hard physical constraints, with a potential extension into Reinforcement Learning for adaptive robustness, this research ensures a measurable improvement in the stability and efficiency of humanoid locomotion over classical batch-processed methods.
