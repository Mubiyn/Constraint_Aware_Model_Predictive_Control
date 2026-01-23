# Section 1: Introduction and System Overview

## 1.1 Project Scope and Objectives
This project focuses on the Design and Optimization of Mechatronic Systems, specifically targeting **Constraint-Aware Model Predictive Control (MPC)** and **Reinforcement Learning (RL)** for bipedal locomotion. 

The primary objective was to replace a traditional Preview Control baseline with a robust, real-time MPC framework capable of handling physical constraints (ZMP boundaries, actuator limits) and external disturbances. Furthermore, the project aimed to augment this controller with learning-based approaches to optimize gait parameters and residual control policies.

We have successfully implemented:
1.  **LIPM-based MPC**: A real-time Quadratic Programming (QP) controller.
2.  **RL Integration**: A Cross-Entropy Method (CEM) optimizer for gait parameters and residual policy training.
3.  **ATP Suite**: A comprehensive Acceptance Test Procedure suite to validate stability, robustness, and efficiency.

## 1.2 Simulation Environment
The development and testing were conducted entirely within a high-fidelity physics simulation.

*   **Engine**: [PyBullet](https://pybullet.org) (v3.2+)
    *   Selected for its faster-than-real-time physics capabilities key for RL training.
    *   **Time Step**: 240Hz (physics), downsampled to ~10ms for control logic.
    *   **Solver**: Numerical constraint solver (SI = 200 iterations).

*   **Kinematics & Dynamics**: [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
    *   Used for rigid body algorithms (RNEA, Forward Kinematics).
    *   Provides Jacobians and frame placements efficiently.

## 1.3 Robot Platform: TALOS
The target platform is the **Talos** humanoid robot (Pal Robotics).

*   **Model**: Full-body URDF (v2).
*   **Key Characteristics**:
    *   **Mass**: ~95kg.
    *   **Height**: ~1.75m.
    *   **Actuation**: Torque-controlled joints (modeled via position/velocity loops in sim).
    *   **Sensing**: Joint encoders, IMU (base orientation), and Force/Torque sensors (simulated via contact points).

### 1.4 Modeling Assumptions & Limitations
The control architecture relies on the **Linear Inverted Pendulum Mode (LIPM)**, which introduces specific limitations:
*   **Constant CoM Height**: The controller assumes the Center of Mass stays at a fixed height ($z_c \approx 0.88m$), ignoring vertical dynamics.
*   **Zero Angular Momentum**: The model assumes the robot does not rotate its torso to generate momentum (no "windmilling" arms).
*   **Massless Legs**: Leg mass is neglected in the simplified dynamic model, though fully simulated in the physics engine.

These assumptions create a "Sim-to-Model" gap. While the QP solver optimizes the simplified model, the RL component helps bridge this gap by learning to compensate for the unmodeled full-body dynamics.
