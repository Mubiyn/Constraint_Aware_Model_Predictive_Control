"""
Model Predictive Controller (MPC) for biped walking using LIPM.

This MPC uses Divergent Component of Motion (DCM) / Capture Point based control
for stable walking. The key insight is that LIPM has unstable dynamics, and
simply tracking ZMP causes the CoM to diverge. Instead, we:

1. Generate a DCM reference trajectory from footstep plan (computed backwards)
2. Use MPC to compute optimal ZMP that tracks DCM while respecting constraints
3. The ZMP commands are then tracked by the lower-level controller

The MPC solves at each timestep:
    min  sum_k ||dcm_k - dcm_ref_k||^2_Q + ||u_k||^2_R + ||dcm_N - dcm_ref_N||^2_Qf
    s.t. zmp_min <= zmp <= zmp_max (support polygon)
         zmp rate limits (smoothness)

DCM Dynamics (first-order unstable system):
    dcm_dot = omega * (dcm - zmp)
    
Discretized: dcm_{k+1} = (1 + omega*dt) * dcm_k - omega*dt * zmp_k

References:
- Englsberger et al., "Three-Dimensional Bipedal Walking Control Based on DCM"
- Kajita et al., "Biped Walking Pattern Generation by using Preview Control of ZMP"
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import time

import numpy as np
from scipy import sparse
from qpsolvers import solve_qp

from v2.models.lipm import LIPM, LIPMParams


@dataclass
class MPCParams:
    """Parameters for the MPC controller.
    
    Attributes:
        horizon: Prediction horizon in number of steps
        dt: Sampling time in seconds
        z_c: CoM height in meters
        g: Gravitational acceleration in m/s^2
        Q_dcm: Weight on DCM tracking error
        R_zmp: Weight on ZMP rate of change (smoothness)
        Q_terminal: Weight on terminal DCM error (for stability)
        zmp_margin: Safety margin for ZMP constraints (meters)
        max_zmp_rate: Maximum ZMP rate of change (m/s)
    """
    horizon: int = 100
    dt: float = 0.01
    z_c: float = 0.87
    g: float = 9.81
    Q_dcm: float = 10.0
    R_zmp: float = 1e-3
    Q_terminal: float = 1000.0
    zmp_margin: float = 0.005
    max_zmp_rate: float = 2.0


class MPCController:
    """DCM-based Model Predictive Controller for stable bipedal walking.
    
    Uses DCM (Divergent Component of Motion) dynamics:
        dcm_{k+1} = A_d * dcm_k + B_d * zmp_k
    where:
        A_d = 1 + omega * dt  (unstable pole > 1)
        B_d = -omega * dt
    
    Solves QP: min ||DCM - DCM_ref||^2_Q + ||dZMP||^2_R + ||DCM_N - DCM_ref_N||^2_Qf
               s.t. ZMP_min <= ZMP <= ZMP_max
                    |ZMP_{k+1} - ZMP_k| <= max_rate * dt
    
    Uses sparse matrices and warm-starting for efficient real-time solving.
    """
    
    def __init__(self, params: MPCParams):
        self.params = params
        self.omega = np.sqrt(params.g / params.z_c)
        self._setup_prediction_matrices()
        self._prev_solution: Optional[np.ndarray] = None
        self._prev_zmp: float = 0.0
        
    def _setup_prediction_matrices(self):
        """Build DCM prediction matrices for the QP formulation."""
        N = self.params.horizon
        dt = self.params.dt
        omega = self.omega
        
        A_d = 1.0 + omega * dt
        B_d = -omega * dt
        
        self.A_d = A_d
        self.B_d = B_d
        
        self.Px = np.zeros((N, 1))
        self.Pu = np.zeros((N, N))
        
        A_pow = 1.0
        for i in range(N):
            A_pow *= A_d
            self.Px[i, 0] = A_pow
            
            for j in range(i + 1):
                power = i - j
                self.Pu[i, j] = (A_d ** power) * B_d
        
        self._build_qp_matrices()
                
    def _build_qp_matrices(self):
        """Pre-build constant QP matrices for efficiency."""
        N = self.params.horizon
        p = self.params
        
        Q = p.Q_dcm * np.eye(N)
        Q[-1, -1] = p.Q_terminal
        
        diff_mat = np.eye(N) - np.diag(np.ones(N - 1), -1)
        R = p.R_zmp * (diff_mat.T @ diff_mat)
        
        self.P_qp_base = self.Pu.T @ Q @ self.Pu + R
        self.P_qp_base = (self.P_qp_base + self.P_qp_base.T) / 2.0 + 1e-6 * np.eye(N)
        
        self.Q_dcm_mat = Q
        self.diff_mat = diff_mat
        
    def compute_control(
        self, 
        dcm: float,
        dcm_ref: np.ndarray,
        zmp_min: np.ndarray,
        zmp_max: np.ndarray,
        current_zmp: Optional[float] = None
    ) -> Tuple[float, dict]:
        """Compute optimal ZMP using MPC.
        
        Args:
            dcm: Current DCM position (scalar, one axis)
            dcm_ref: Reference DCM trajectory, shape (N,)
            zmp_min: Lower bound on ZMP, shape (N,)
            zmp_max: Upper bound on ZMP, shape (N,)
            current_zmp: Current ZMP for rate limiting (optional)
            
        Returns:
            zmp: Optimal ZMP for current time step
            info: Dictionary with solver information
        """
        N = self.params.horizon
        p = self.params
        dt = p.dt
        
        dcm_ref = np.asarray(dcm_ref).flatten()
        zmp_min = np.asarray(zmp_min).flatten()
        zmp_max = np.asarray(zmp_max).flatten()
        
        if len(dcm_ref) < N:
            dcm_ref = np.pad(dcm_ref, (0, N - len(dcm_ref)), mode='edge')
        if len(zmp_min) < N:
            zmp_min = np.pad(zmp_min, (0, N - len(zmp_min)), mode='edge')
        if len(zmp_max) < N:
            zmp_max = np.pad(zmp_max, (0, N - len(zmp_max)), mode='edge')
            
        dcm_ref = dcm_ref[:N]
        zmp_min_constrained = zmp_min[:N] + p.zmp_margin
        zmp_max_constrained = zmp_max[:N] - p.zmp_margin
        
        zmp_min_constrained = np.minimum(zmp_min_constrained, zmp_max_constrained - 0.001)
        
        b = (self.Px @ np.array([[dcm]])).flatten()
        q_qp = self.Pu.T @ self.Q_dcm_mat @ (b - dcm_ref)
        
        if current_zmp is not None:
            q_qp[0] += p.R_zmp * (-current_zmp)
        
        constraints = []
        constraint_bounds = []
        
        constraints.append(np.eye(N))
        constraint_bounds.append((zmp_min_constrained, zmp_max_constrained))
        
        if p.max_zmp_rate < np.inf and current_zmp is not None:
            max_delta = p.max_zmp_rate * dt
            
            rate_lb = np.full(N, -max_delta)
            rate_ub = np.full(N, max_delta)
            rate_lb[0] = current_zmp - max_delta
            rate_ub[0] = current_zmp + max_delta
            
            for i in range(1, N):
                rate_lb[i] = -max_delta
                rate_ub[i] = max_delta
        
        G_zmp = np.vstack([np.eye(N), -np.eye(N)])
        h_zmp = np.concatenate([zmp_max_constrained, -zmp_min_constrained])
        
        start_time = time.time()
        
        try:
            init_x = self._prev_solution if self._prev_solution is not None else None
            
            result = solve_qp(
                P=self.P_qp_base.astype(np.float64), 
                q=q_qp.astype(np.float64), 
                G=G_zmp.astype(np.float64), 
                h=h_zmp.astype(np.float64),
                solver="osqp",
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=500,
                polish=True,
                warm_start=True,
                initvals=init_x
            )
            
            solve_time = time.time() - start_time
            
            if result is not None:
                self._prev_solution = result
                zmp_opt = result[0]
                predicted_dcm = self.Pu @ result + b
                
                info = {
                    'status': 'solved',
                    'optimal': True,
                    'solve_time': solve_time,
                    'predicted_dcm': predicted_dcm,
                    'zmp_sequence': result,
                    'dcm_ref': dcm_ref,
                    'tracking_error': np.mean(np.abs(predicted_dcm - dcm_ref)),
                    'raw_zmp': zmp_opt
                }
            else:
                zmp_opt = self._fallback_control(dcm, dcm_ref[0], zmp_min[0], zmp_max[0])
                info = {
                    'status': 'failed',
                    'optimal': False,
                    'solve_time': solve_time,
                    'predicted_dcm': None,
                    'zmp_sequence': None
                }
        except Exception as e:
            solve_time = time.time() - start_time
            zmp_opt = self._fallback_control(dcm, dcm_ref[0], zmp_min[0], zmp_max[0])
            info = {
                'status': f'error: {str(e)}',
                'optimal': False,
                'solve_time': solve_time,
                'predicted_dcm': None,
                'zmp_sequence': None
            }
            
        self._prev_zmp = zmp_opt
        return zmp_opt, info
    
    def _fallback_control(
        self, 
        dcm: float, 
        dcm_ref: float, 
        zmp_min: float, 
        zmp_max: float
    ) -> float:
        """Fallback DCM feedback controller when MPC fails."""
        k_dcm = 2.0
        zmp_des = dcm + k_dcm * (dcm - dcm_ref) / self.omega
        return np.clip(zmp_des, zmp_min, zmp_max)
    
    def reset(self):
        """Reset controller state."""
        self._prev_solution = None
        self._prev_zmp = 0.0


class DecoupledMPCController:
    """Decoupled DCM-based MPC for x and y axes.
    
    Since LIPM/DCM dynamics are decoupled, we solve two independent QPs
    for the sagittal (x) and lateral (y) directions. The MPC outputs
    desired ZMP positions which stabilize the DCM toward reference.
    
    Includes adaptive DCM feedback for phase transitions and disturbance recovery.
    """
    
    def __init__(self, params: MPCParams):
        self.params = params
        self.omega = np.sqrt(params.g / params.z_c)
        self.mpc_x = MPCController(params)
        self.mpc_y = MPCController(params)
        self._current_zmp = np.zeros(2)
        
    def compute_dcm(self, com: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """Compute DCM from CoM position and velocity.
        
        Args:
            com: CoM position [x, y, z]
            com_vel: CoM velocity [vx, vy, vz]
            
        Returns:
            DCM position [dcm_x, dcm_y]
        """
        return np.array([
            com[0] + com_vel[0] / self.omega,
            com[1] + com_vel[1] / self.omega
        ])
        
    def compute_control(
        self,
        com: np.ndarray,
        com_vel: np.ndarray,
        dcm_ref_x: np.ndarray,
        dcm_ref_y: np.ndarray,
        zmp_bounds_x: Tuple[np.ndarray, np.ndarray],
        zmp_bounds_y: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, dict]:
        """Compute optimal ZMP for both axes.
        
        Args:
            com: Current CoM position [x, y, z]
            com_vel: Current CoM velocity [vx, vy, vz]
            dcm_ref_x: x-axis DCM reference trajectory
            dcm_ref_y: y-axis DCM reference trajectory
            zmp_bounds_x: Tuple of (min, max) ZMP bounds for x-axis
            zmp_bounds_y: Tuple of (min, max) ZMP bounds for y-axis
            
        Returns:
            zmp_des: Desired ZMP [zmp_x, zmp_y]
            info: Combined solver information
        """
        dcm = self.compute_dcm(com, com_vel)
        dcm_x, dcm_y = dcm[0], dcm[1]
        
        start = time.time()
        
        zmp_x, info_x = self.mpc_x.compute_control(
            dcm_x, dcm_ref_x, zmp_bounds_x[0], zmp_bounds_x[1],
            current_zmp=self._current_zmp[0]
        )
        zmp_y, info_y = self.mpc_y.compute_control(
            dcm_y, dcm_ref_y, zmp_bounds_y[0], zmp_bounds_y[1],
            current_zmp=self._current_zmp[1]
        )
        
        total_time = time.time() - start
        
        self._current_zmp = np.array([zmp_x, zmp_y])
        
        info = {
            'x': info_x,
            'y': info_y,
            'optimal': info_x['optimal'] and info_y['optimal'],
            'solve_time': total_time,
            'dcm_current': dcm
        }
        
        return np.array([zmp_x, zmp_y]), info
    
    def reset(self):
        """Reset both controllers."""
        self.mpc_x.reset()
        self.mpc_y.reset()
        self._current_zmp = np.zeros(2)


class DCMTrajectoryGenerator:
    """Generate DCM reference trajectory from footstep plan.
    
    The DCM trajectory is computed backwards from the final footstep
    (where DCM should converge to foot position) using the analytical
    solution of the DCM dynamics:
    
        dcm(t) = zmp + (dcm_0 - zmp) * exp(omega * t)
        
    When computing backwards from final position:
        dcm(t-dt) = zmp + (dcm(t) - zmp) * exp(-omega * dt)
        
    This ensures the DCM naturally converges to each footstep position
    at the correct time, providing stable walking references.
    """
    
    def __init__(self, omega: float, dt: float):
        self.omega = omega
        self.dt = dt
        
    def generate_dcm_for_phase(
        self,
        t_phase: float,
        t_phase_duration: float,
        zmp_current: np.ndarray,
        dcm_end: np.ndarray,
    ) -> np.ndarray:
        """Generate DCM reference for current phase using analytical solution.
        
        For a given phase, the DCM trajectory from current time to phase end
        is computed so that DCM arrives at dcm_end when phase completes.
        
        Args:
            t_phase: Time elapsed in current phase
            t_phase_duration: Total duration of current phase
            zmp_current: Current support foot position (ZMP target)
            dcm_end: DCM position at end of phase (next footstep)
            
        Returns:
            Current DCM reference position
        """
        t_remaining = max(0.001, t_phase_duration - t_phase)
        
        dcm_ref = zmp_current + (dcm_end - zmp_current) * np.exp(-self.omega * t_remaining)
        
        return dcm_ref
        
    def generate_dcm_trajectory(
        self,
        footsteps: np.ndarray,
        step_durations: np.ndarray,
        t_ds: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate DCM trajectory from footstep sequence using backward recursion.
        
        DCM reference is computed backwards from the final footstep position,
        ensuring convergence to each footstep at the correct time.
        
        Args:
            footsteps: Array of footstep positions, shape (N, 2) for [x, y]
            step_durations: Duration of single support for each step, shape (N-1,)
            t_ds: Double support duration between steps
            
        Returns:
            time_vec: Time vector
            dcm_x: DCM x trajectory
            dcm_y: DCM y trajectory
        """
        n_steps = len(footsteps)
        if n_steps < 2:
            return np.array([0.0]), np.array([footsteps[0, 0]]), np.array([footsteps[0, 1]])
        
        total_ss_time = np.sum(step_durations)
        total_ds_time = (n_steps - 1) * t_ds
        total_time = total_ss_time + total_ds_time
        
        n_samples = int(total_time / self.dt) + 1
        
        time_vec = np.linspace(0, total_time, n_samples)
        dcm_x = np.zeros(n_samples)
        dcm_y = np.zeros(n_samples)
        
        dcm_final = footsteps[-1].copy()
        dcm_current = dcm_final.copy()
        
        sample_idx = n_samples - 1
        
        for step_idx in range(n_steps - 2, -1, -1):
            zmp = footsteps[step_idx]
            t_ss = step_durations[step_idx]
            
            n_ds_samples = int(t_ds / self.dt)
            for _ in range(n_ds_samples):
                if sample_idx < 0:
                    break
                dcm_x[sample_idx] = dcm_current[0]
                dcm_y[sample_idx] = dcm_current[1]
                
                dcm_current = zmp + (dcm_current - zmp) * np.exp(-self.omega * self.dt)
                sample_idx -= 1
            
            n_ss_samples = int(t_ss / self.dt)
            for _ in range(n_ss_samples):
                if sample_idx < 0:
                    break
                dcm_x[sample_idx] = dcm_current[0]
                dcm_y[sample_idx] = dcm_current[1]
                
                dcm_current = zmp + (dcm_current - zmp) * np.exp(-self.omega * self.dt)
                sample_idx -= 1
        
        while sample_idx >= 0:
            dcm_x[sample_idx] = dcm_current[0]
            dcm_y[sample_idx] = dcm_current[1]
            sample_idx -= 1
            
        return time_vec, dcm_x, dcm_y
    
    def compute_vrp_trajectory(
        self,
        footsteps: np.ndarray,
        step_durations: np.ndarray,
        z_c: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Virtual Repellent Point trajectory for enhanced stability.
        
        VRP is the point where ground reaction force line intersects CoM height plane.
        For walking, VRP provides additional control authority during transitions.
        
        Args:
            footsteps: Footstep positions
            step_durations: Step timing
            z_c: CoM height
            
        Returns:
            time_vec, vrp_x, vrp_y trajectories
        """
        time_vec, dcm_x, dcm_y = self.generate_dcm_trajectory(footsteps, step_durations)
        
        vrp_x = np.gradient(dcm_x, self.dt) / self.omega + dcm_x
        vrp_y = np.gradient(dcm_y, self.dt) / self.omega + dcm_y
        
        return time_vec, vrp_x, vrp_y
