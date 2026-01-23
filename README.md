# Constraint-Aware Model Predictive Control & Reinforcement Learning for Bipedal Locomotion

![Baseline MPC Walk](assets/baseline_walk.gif)

## Overview
This project implements a robust control framework for the **Talos Humanoid Robot** using a hierarchical architecture:
1.  **Low-Level Controller:** A Constraint-Aware **Model Predictive Controller (MPC)** based on the Linear Inverted Pendulum Mode (LIPM) that guarantees ZMP stability constraints using QP (Quadratic Programming).
2.  **High-Level Planner:** A **Reinforcement Learning (RL)** agent trained via the Cross-Entropy Method (CEM) to optimize gait parameters and residual references for robust disturbance rejection.

## Key Features
*   **Real-time Optimization:** <10ms solve time using OSQP.
*   **Constraint Satisfaction:** Guaranteed ZMP stability within the support polygon.
*   **Robustness:** Withstands external forces up to **100N** (66% improvement over baseline).
*   **Adaptive Gait:** Dynamic adjustment of step width and timing based on RL policy.

## Quick Start from `v2` Directory

### 1. Installation
Ensure you have Python 3.10+ and the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Individual Tests
You can run specific test cases from the ATP (Acceptance Test Procedure):
```bash
# Run steady state walking
python main_mpc.py --test steady_state

# Run disturbance rejection test (Force applied at t=2.0s)
python main_mpc.py --test disturbance

# Run velocity tracking
python main_mpc.py --test velocity
```

### 3. Reproduce Full Results
To generate all plots and videos (Comparison between Baseline and RL):
```bash
./run_finalization.sh
```

## Results & Documentation

*   **[Detailed Results & Analysis](Results.md)**: Comparative analysis, plots, and additional GIFs.
*   **[Full Technical Report (PDF)](report_tex/main.pdf)**: Comprehensive academic report detailing the theoretical framework and mathematical derivation.

## Project Structure
*   `controllers/`: MPC implementation and QP solvers.
*   `simulation/`: PyBullet environment and robot wrappers.
*   `report_tex/`: LaTeX source for the final report.
*   `results/`: Generated artifacts (plots, videos, logs).

## References & Citation

### Cite this Project
If you use this code or report for academic purposes, please cite the authors:
> **Bokono B. N., Mir M., Sheidu M.**, "Constraint-Aware Model Predictive Control and Reinforcement Learning for Bipedal Locomotion," *Design and Optimization of Mechatronic Systems*, 2026.

### Baseline & Inspiration
This project builds upon the foundational work in ZMP Preview Control and the open-source implementation by [rdesarz/biped-walking-controller](https://github.com/rdesarz/biped-walking-controller).

### Literature
1.  **Kajita, S., et al.** "Biped walking pattern generation by using preview control of zero-moment point." *IEEE International Conference on Robotics and Automation (ICRA)*, 2003.
2.  **Englsberger, J., et al.** "Three-dimensional bipedal walking control based on divergent component of motion." *IEEE Transactions on Robotics*, 2015.

---
**Authors:** Bokono Bennet Nathan, Mohadeseh Mir, Mubin Sheidu
**Course:** Design and Optimization of Mechatronic Systems
