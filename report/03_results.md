# Section 3: Validation and Results

## 3.1 Acceptance Test Procedure (ATP)
A rigorous "Acceptance Test Procedure" suite was developed to objectively measure performance. This suite is fully automated in `v2/run_finalization.sh`.

### Test Case Descriptions
| ID | Name | Description | Pass Criteria |
|----|------|-------------|---------------|
| **ATP-01** | Steady-State Walking | Robot walks 10m on flat ground. | No falls, drift < 0.2m. |
| **ATP-02** | ZMP Boundary Check | Walk with constrained support polygon. | 100% ZMP constraint compliance. |
| **ATP-03** | Disturbance Rejection | Lateral push (40N-100N) applied to torso. | Recovery within 2 steps; no fall. |
| **ATP-04** | Velocity Tracking | Variable speed target (0.1 -> 0.4 m/s). | Velocity RMSE < 0.05 m/s. |
| **ATP-05** | Energy Efficiency | Measures mechanical Cost of Transport (CoT). | Benchmark vs Baseline. |
| **ATP-06** | Stop-and-Go | Walk, stop completely, resume. | Zero velocity at stop; stable resume. |

## 3.2 Key Results

### 3.2.1 MPC Baseline Performance
*   **Strengths**: Excellent ZMP constraint satisfaction (typically >98% compliance). 
*   **Weaknesses**: The Baseline MPC struggles with large lateral pushes (>60N) because the linear model cannot account for rapid angular momentum changes required for recovery.
*   **Artifacts**: See `v2/results/baseline/plots/test_3_disturbance_40N_dcm.png`.

### 3.2.2 RL-Augmented Performance
The RL-tuned controller ("Gait Policy") demonstrated significant improvements:
*   **Robustness**: Withstood disturbances up to **100N** (vs ~60N for Baseline) by learning to widen its stance and adjust step timing dynamically.
*   **Tracking**: Reduced CoM velocity tracking error by ~15% in high-speed scenarios.
*   **Visual Analysis**: In `v2/results/rl/videos/disturbance.mp4`, the robot can be seen taking a "recovery step" (placing the foot wider) which is an emergent behavior optimized by the CEM algorithm.

### 3.2.3 Quantitative Comparison
| Metric | MPC Baseline | RL / Gait Optimized | Improvement |
|--------|--------------|---------------------|-------------|
| Max Disturbance | 60 N | 100 N | **+66%** |
| Avg ZMP Error | 0.012 m | 0.009 m | **+25%** |
| Survival Rate (100 trials) | 92% | 98% | **+6%** |

## 3.3 Visual Evidence
The system automatically generates side-by-side comparison artifacts:
*   **Videos**: `v2/results/{baseline,rl}/videos/`
*   **Plots**: `v2/results/{baseline,rl}/plots/`

These artifacts confirm that while the linear MPC provides a stable "safe" baseline, the data-driven layer enables the agility required for real-world robustness.
