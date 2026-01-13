"""
Foot trajectory generation and support polygon utilities.

Provides:
- Quintic Bezier swing foot trajectories with zero vel/acc at endpoints
- Support polygon computation for ZMP constraints
- Step sequence generation for walking patterns
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np
from shapely import Point, Polygon, affinity


class Foot(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class FootParams:
    """Foot geometry parameters for support polygon computation.
    
    Attributes:
        length: Foot length in meters (heel to toe)
        width: Foot width in meters
        offset_x: Offset from ankle to foot center in x
        offset_y: Offset from ankle to foot center in y
    """
    length: float = 0.20
    width: float = 0.10
    offset_x: float = 0.0
    offset_y: float = 0.0


def _bezier_quintic(P: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate quintic Bezier curve at parameter values s.
    
    Args:
        P: Control points, shape (6, 3)
        s: Parameter values in [0, 1], shape (N,)
        
    Returns:
        Curve samples, shape (N, 3)
    """
    B = np.array([
        math.comb(5, i) * ((1.0 - s) ** (5 - i)) * (s ** i) 
        for i in range(6)
    ])
    return B.T @ P


class FootTrajectoryGenerator:
    """Generate smooth swing foot trajectories using quintic Bezier curves.
    
    The trajectory has zero velocity and acceleration at both endpoints,
    suitable for smooth foot landing and takeoff.
    """
    
    def __init__(self, foot_height: float = 0.03):
        """Initialize generator with desired foot clearance.
        
        Args:
            foot_height: Maximum height of swing foot above ground
        """
        self.foot_height = foot_height
        self._calibrate_alpha()
        
    def _calibrate_alpha(self):
        """Find alpha parameter that achieves desired apex height."""
        P = np.zeros((6, 3))
        P[0] = P[1] = P[2] = np.array([0.0, 0.0, 0.0])
        P[3] = P[4] = P[5] = np.array([0.3, 0.0, 0.0])
        
        self.alpha = 0.0
        for alpha in np.linspace(0.0, 2.0 * self.foot_height, 3000):
            P[1][2] = alpha / 2.0
            P[2][2] = alpha
            P[3][2] = alpha
            P[4][2] = alpha / 2.0
            
            apex = _bezier_quintic(P, np.array([0.5]))
            if abs(apex[0, 2] - self.foot_height) < 1e-3:
                self.alpha = alpha
                break
                
    def generate(
        self, 
        p_start: np.ndarray, 
        p_end: np.ndarray, 
        s: np.ndarray
    ) -> np.ndarray:
        """Generate swing foot trajectory between two positions.
        
        Args:
            p_start: Start position [x, y, z]
            p_end: End position [x, y, z]
            s: Phase values in [0, 1], shape (N,)
            
        Returns:
            Trajectory points, shape (N, 3)
        """
        P = np.zeros((6, 3))
        P[0] = P[1] = P[2] = p_start.copy()
        P[3] = P[4] = P[5] = p_end.copy()
        
        P[1][2] = p_start[2] + self.alpha / 2.0
        P[2][2] = p_start[2] + self.alpha
        P[3][2] = p_end[2] + self.alpha
        P[4][2] = p_end[2] + self.alpha / 2.0
        
        return _bezier_quintic(P, s)


def compute_steps_sequence(
    rf_initial: np.ndarray,
    lf_initial: np.ndarray,
    n_steps: int,
    stride_length: float,
    lateral_offset: float = None
) -> Tuple[np.ndarray, List[Foot]]:
    """Generate a straight-line walking step sequence.
    
    Args:
        rf_initial: Initial right foot position [x, y, z]
        lf_initial: Initial left foot position [x, y, z]
        n_steps: Number of steps to generate
        stride_length: Length of each step in meters
        lateral_offset: Lateral distance between feet (default: from initial)
        
    Returns:
        steps_pose: Array of step positions, shape (n_steps + 2, 3)
        steps_foot: List of Foot enums indicating which foot each step belongs to
    """
    if lateral_offset is None:
        lateral_offset = abs(lf_initial[1] - rf_initial[1]) / 2.0
        
    steps_pose = np.zeros((n_steps + 2, 3))
    steps_foot = []
    
    steps_pose[0] = rf_initial.copy()
    steps_foot.append(Foot.RIGHT)
    
    for i in range(1, n_steps + 1):
        is_right = (i % 2 == 0)
        sign = -1.0 if is_right else 1.0
        
        steps_pose[i] = np.array([
            rf_initial[0] + i * stride_length,
            sign * lateral_offset,
            rf_initial[2]
        ])
        steps_foot.append(Foot.RIGHT if is_right else Foot.LEFT)
        
    final_x = rf_initial[0] + n_steps * stride_length
    if n_steps % 2 == 0:
        steps_pose[n_steps + 1] = np.array([final_x, lateral_offset, rf_initial[2]])
        steps_foot.append(Foot.LEFT)
    else:
        steps_pose[n_steps + 1] = np.array([final_x, -lateral_offset, rf_initial[2]])
        steps_foot.append(Foot.RIGHT)
        
    return steps_pose, steps_foot


def compute_support_polygon(
    foot_position: np.ndarray,
    foot_yaw: float,
    foot_params: FootParams
) -> Polygon:
    """Compute support polygon for a single foot.
    
    Args:
        foot_position: Foot center position [x, y, z]
        foot_yaw: Foot orientation in radians
        foot_params: Foot geometry parameters
        
    Returns:
        Shapely Polygon representing the support region
    """
    half_l = foot_params.length / 2.0
    half_w = foot_params.width / 2.0
    
    corners = [
        (foot_params.offset_x + half_l, foot_params.offset_y + half_w),
        (foot_params.offset_x + half_l, foot_params.offset_y - half_w),
        (foot_params.offset_x - half_l, foot_params.offset_y - half_w),
        (foot_params.offset_x - half_l, foot_params.offset_y + half_w),
    ]
    
    polygon = Polygon(corners)
    polygon = affinity.rotate(polygon, foot_yaw, use_radians=True, origin=(0, 0))
    polygon = affinity.translate(polygon, foot_position[0], foot_position[1])
    
    return polygon


def compute_zmp_bounds(
    support_polygon: Polygon
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get axis-aligned bounding box of support polygon for ZMP constraints.
    
    Args:
        support_polygon: Shapely Polygon
        
    Returns:
        x_bounds: (x_min, x_max)
        y_bounds: (y_min, y_max)
    """
    bounds = support_polygon.bounds
    return (bounds[0], bounds[2]), (bounds[1], bounds[3])


def get_double_support_polygon(
    lf_position: np.ndarray,
    rf_position: np.ndarray,
    lf_yaw: float,
    rf_yaw: float,
    foot_params: FootParams
) -> Polygon:
    """Compute support polygon during double support.
    
    Args:
        lf_position: Left foot position
        rf_position: Right foot position
        lf_yaw: Left foot yaw
        rf_yaw: Right foot yaw
        foot_params: Foot geometry
        
    Returns:
        Combined support polygon (convex hull of both feet)
    """
    lf_poly = compute_support_polygon(lf_position, lf_yaw, foot_params)
    rf_poly = compute_support_polygon(rf_position, rf_yaw, foot_params)
    return lf_poly.union(rf_poly).convex_hull
