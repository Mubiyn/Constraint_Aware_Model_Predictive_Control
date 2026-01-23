# Section 2: Algorithmic Implementation

## 2.1 Baseline: ZMP-based Preview Control
While the initial goal was to surpass Preview Control, we retained standard ZMP Preview principles for the Divergent Component of Motion (DCM) reference generation. The Preview Control looks ahead at the planned footsteps to generate a smooth Center of Mass (CoM) trajectory that keeps the Zero Moment Point (ZMP) centered in the coming support polygons.

## 2.2 Core: Model Predictive Control (MPC)
The core controller is a **Quadratic Programming (QP)** formulation solved at 100Hz using the **OSQP** solver.

### 2.2.1 State Space Formulation
The dynamics are discretized with a sampling time $T = 0.01s$. The state vector $x_k$ includes the CoM position, velocity, and ZMP.
$$
\min_{U} \sum_{k=0}^{N} \| ZMP_k - ZMP_{ref,k} \|^2_{Q} + \| \dot{x}_k \|^2_{R} + \| \text{Jerk}_k \|^2_{S}
$$

### 2.2.2 Constraints
Unlike analytical Preview Control, our MPC enforces hard inequality constraints:
1.  **ZMP Stability Constraint**: The ZMP must remain strictly within the convex hull of the support foot (or feet in double support).
    $$ d_{min} \le ZMP_k \le d_{max} $$
2.  **Terminal Constraint**: The state at the end of the horizon must align with the capture point to ensure long-term viability (recursive feasibility).

## 2.3 Reinforcement Learning Augmentation
We implemented a hybrid architecture where learning augments, rather than replaces, the model-based controller.

### 2.3.1 Approach 1: Residual Learning
An RL policy (Neural Network or Linear) outputs a residual signal $\delta_{ZMP}$ added to the MPC's target.
*   **Input**: State vector [CoM error, Velocity error, Foot phase].
*   **Output**: ZMP offset.
*   **Goal**: Correct for non-linear effects (like joint friction or leg mass) that the Linear MPC ignores.

### 2.3.2 Approach 2: Gait Parameter Optimization (CEM)
We utilized the **Cross-Entropy Method (CEM)**, a derivative-free evolutionary algorithm, to optimize high-level gait parameters offline.
*   **Parameters Optimized**:
    *   MPC Weights ($Q, R$ matrices).
    *   Step timings (Single Support vs Double Support durability).
    *   Swing height trajectories.
*   **Objective Function**: Minimize a compound cost of (Tracking Error + Energy Expenditure + Fall Penalties).
*   **Result**: The "Gait Policy" (`rl/gait_cem_v2.npz`) used in the final results was discovered via this method.

## 2.4 Whole-Body Control (Inverse Kinematics)
The high-level MPC outputs a desired CoM trajectory and foot positions. A differential Inverse Kinematics (IK) solver converts these Cartesian tasks into joint angles ($q$) and velocities ($\dot{q}$) sent to the low-level PIDs.
*   **Tasks**: CoM tracking, Foot tracking, Torso orientation, Posture regulation.
*   **Priority**: Foot ground contact is strictly prioritized over CoM tracking to prevent slipping.
