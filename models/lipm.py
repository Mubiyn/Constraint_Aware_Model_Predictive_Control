"""
Linear Inverted Pendulum Model (LIPM) for biped walking.

The LIPM simplifies the humanoid robot dynamics by assuming:
- Mass concentrated at CoM
- Massless legs
- Constant CoM height
- No angular momentum about CoM

This results in decoupled linear dynamics:
    x_ddot = (g/z_c) * (x_c - x_z)
    y_ddot = (g/z_c) * (y_c - y_z)

For MPC formulation, we use jerk (u = x_dddot) as control input.

Key Concept - Divergent Component of Motion (DCM) / Capture Point:
    dcm = x + x_dot / omega
    
The DCM represents where the robot needs to step to come to rest.
DCM dynamics: dcm_dot = omega * (dcm - zmp)

For stable walking, we track the DCM rather than just ZMP.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LIPMParams:
    """Parameters for the Linear Inverted Pendulum Model.
    
    Attributes:
        z_c: CoM height in meters
        g: Gravitational acceleration in m/s^2
        dt: Sampling time in seconds
    """
    z_c: float = 0.8
    g: float = 9.81
    dt: float = 0.01
    
    @property
    def omega(self) -> float:
        """Natural frequency of the pendulum."""
        return np.sqrt(self.g / self.z_c)


class LIPM:
    """Linear Inverted Pendulum Model with jerk control input.
    
    State vector: [x, x_dot, x_ddot] (position, velocity, acceleration)
    Control input: u = x_dddot (jerk)
    Output: ZMP position p_x = x - (z_c/g) * x_ddot
    
    Discrete-time state-space model:
        x_{k+1} = A * x_k + B * u_k
        p_k = C * x_k
    """
    
    def __init__(self, params: LIPMParams):
        self.params = params
        self._build_matrices()
        
    def _build_matrices(self):
        """Build discrete-time state-space matrices."""
        T = self.params.dt
        z_c = self.params.z_c
        g = self.params.g
        
        # State transition matrix (3x3)
        self.A = np.array([
            [1.0, T, T**2 / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0]
        ])
        
        # Input matrix (3x1)
        self.B = np.array([
            [T**3 / 6.0],
            [T**2 / 2.0],
            [T]
        ])
        
        # Output matrix for ZMP (1x3)
        # p_x = x - (z_c/g) * x_ddot
        self.C = np.array([[1.0, 0.0, -z_c / g]])
        
        # Direct feedthrough (1x1)
        self.D = np.array([[0.0]])
        
    def predict(self, x: np.ndarray, u: float) -> np.ndarray:
        """Predict next state given current state and control input.
        
        Args:
            x: Current state [x, x_dot, x_ddot], shape (3,)
            u: Control input (jerk), scalar
            
        Returns:
            Next state, shape (3,)
        """
        return self.A @ x + self.B.flatten() * u
    
    def get_zmp(self, x: np.ndarray) -> float:
        """Compute ZMP position from state.
        
        Args:
            x: State [x, x_dot, x_ddot], shape (3,)
            
        Returns:
            ZMP position, scalar
        """
        return float(self.C @ x)
    
    def build_prediction_matrices(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build prediction matrices for MPC over horizon N.
        
        The predicted states over N steps can be written as:
            X = Sx * x0 + Su * U
            
        where X = [x_1, x_2, ..., x_N] and U = [u_0, u_1, ..., u_{N-1}]
        
        Args:
            N: Prediction horizon
            
        Returns:
            Sx: State propagation matrix (3N x 3)
            Su: Control influence matrix (3N x N)
        """
        A, B = self.A, self.B
        n_x = 3
        n_u = 1
        
        Sx = np.zeros((n_x * N, n_x))
        Su = np.zeros((n_x * N, n_u * N))
        
        A_pow = np.eye(n_x)
        for i in range(N):
            A_pow = A_pow @ A
            Sx[i*n_x:(i+1)*n_x, :] = A_pow
            
            for j in range(i + 1):
                power = i - j
                A_pow_j = np.linalg.matrix_power(A, power)
                Su[i*n_x:(i+1)*n_x, j*n_u:(j+1)*n_u] = A_pow_j @ B
                
        return Sx, Su
    
    def build_zmp_prediction_matrices(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build ZMP prediction matrices for MPC.
        
        Predicted ZMP positions: P = Px * x0 + Pu * U
        
        Args:
            N: Prediction horizon
            
        Returns:
            Px: ZMP state propagation matrix (N x 3)
            Pu: ZMP control influence matrix (N x N)
        """
        Sx, Su = self.build_prediction_matrices(N)
        n_x = 3
        
        # Stack C matrices for each prediction step
        C_bar = np.kron(np.eye(N), self.C)  # (N x 3N)
        
        Px = C_bar @ Sx  # (N x 3)
        Pu = C_bar @ Su  # (N x N)
        
        return Px, Pu
    
    def get_com_from_state(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Extract CoM position, velocity, acceleration from state.
        
        Args:
            x: State vector [x, x_dot, x_ddot]
            
        Returns:
            Tuple of (position, velocity, acceleration)
        """
        return x[0], x[1], x[2]


class DecoupledLIPM:
    """Decoupled LIPM for x and y axes.
    
    Since the dynamics are linear and decoupled, we can handle
    x and y axes independently.
    """
    
    def __init__(self, params: LIPMParams):
        self.params = params
        self.lipm_x = LIPM(params)
        self.lipm_y = LIPM(params)
        
    def predict(self, state_x: np.ndarray, state_y: np.ndarray, 
                u_x: float, u_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next states for both axes.
        
        Args:
            state_x: x-axis state [x, x_dot, x_ddot]
            state_y: y-axis state [y, y_dot, y_ddot]
            u_x: x-axis jerk
            u_y: y-axis jerk
            
        Returns:
            Tuple of (next_state_x, next_state_y)
        """
        return self.lipm_x.predict(state_x, u_x), self.lipm_y.predict(state_y, u_y)
    
    def get_zmp(self, state_x: np.ndarray, state_y: np.ndarray) -> Tuple[float, float]:
        """Get ZMP position for both axes.
        
        Args:
            state_x: x-axis state
            state_y: y-axis state
            
        Returns:
            Tuple of (zmp_x, zmp_y)
        """
        return self.lipm_x.get_zmp(state_x), self.lipm_y.get_zmp(state_y)
    
    def get_com_position(self, state_x: np.ndarray, state_y: np.ndarray) -> np.ndarray:
        """Get 3D CoM position.
        
        Args:
            state_x: x-axis state
            state_y: y-axis state
            
        Returns:
            CoM position [x, y, z]
        """
        return np.array([state_x[0], state_y[0], self.params.z_c])
    
    def get_com_velocity(self, state_x: np.ndarray, state_y: np.ndarray) -> np.ndarray:
        """Get 3D CoM velocity.
        
        Args:
            state_x: x-axis state
            state_y: y-axis state
            
        Returns:
            CoM velocity [x_dot, y_dot, z_dot]
        """
        return np.array([state_x[1], state_y[1], 0.0])
    
    def get_dcm(self, state_x: np.ndarray, state_y: np.ndarray) -> np.ndarray:
        """Compute Divergent Component of Motion (Capture Point).
        
        DCM = CoM + CoM_velocity / omega
        
        Args:
            state_x: x-axis state
            state_y: y-axis state
            
        Returns:
            DCM position [dcm_x, dcm_y]
        """
        omega = self.params.omega
        dcm_x = state_x[0] + state_x[1] / omega
        dcm_y = state_y[0] + state_y[1] / omega
        return np.array([dcm_x, dcm_y])
    
    def compute_stable_zmp(
        self, 
        state_x: np.ndarray, 
        state_y: np.ndarray,
        dcm_ref: np.ndarray,
        k_dcm: float = 2.0
    ) -> np.ndarray:
        """Compute ZMP that drives DCM toward reference using feedback control.
        
        From DCM dynamics: dcm_dot = omega * (dcm - zmp)
        For DCM tracking: dcm_dot_des = k * (dcm_ref - dcm)
        Therefore: zmp = dcm - dcm_dot_des / omega
                      = dcm - k * (dcm_ref - dcm) / omega
                      = dcm * (1 + k/omega) - k * dcm_ref / omega
        
        Alternatively (more intuitive):
            zmp = dcm + (dcm - dcm_ref) * k / omega
        
        Args:
            state_x: x-axis state
            state_y: y-axis state
            dcm_ref: Reference DCM position [dcm_x_ref, dcm_y_ref]
            k_dcm: DCM feedback gain (higher = more aggressive tracking)
            
        Returns:
            Desired ZMP position [zmp_x, zmp_y]
        """
        omega = self.params.omega
        dcm = self.get_dcm(state_x, state_y)
        zmp = dcm + (dcm - dcm_ref) * k_dcm / omega
        return zmp
