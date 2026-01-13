"""
Inverse kinematics solver for whole-body control.

Uses a QP formulation to track CoM, foot, and torso tasks while
respecting kinematic constraints. The stance foot is constrained
to remain fixed while tracking other tasks.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pinocchio as pin
from qpsolvers import solve_qp


@dataclass
class IKParams:
    """Parameters for inverse kinematics solver.
    
    Attributes:
        model: Pinocchio model
        data: Pinocchio data
        fixed_foot_frame: Frame ID of stance foot
        moving_foot_frame: Frame ID of swing foot
        torso_frame: Frame ID of torso
        w_com: Weight on CoM tracking task
        w_foot: Weight on swing foot task
        w_torso: Weight on torso orientation task
        mu: Damping coefficient for regularization
        dt: Time step
        locked_joints: Optional list of joints to lock
    """
    model: pin.Model
    data: pin.Data
    fixed_foot_frame: int
    moving_foot_frame: int
    torso_frame: int
    w_com: float = 1.0
    w_foot: float = 1.0
    w_torso: float = 0.1
    mu: float = 1e-4
    dt: float = 0.01
    locked_joints: Optional[List[int]] = None


def _se3_error_and_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    frame_id: int,
    target_pose: pin.SE3
) -> tuple:
    """Compute 6D pose error and task Jacobian.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Current configuration
        frame_id: Target frame ID
        target_pose: Desired frame pose
        
    Returns:
        error: 6D pose error (angular, linear)
        jacobian: 6xnv task Jacobian
    """
    current_pose = data.oMf[frame_id]
    error_pose = current_pose.actInv(target_pose)
    error = pin.log(error_pose).vector
    
    jacobian = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)
    jlog = pin.Jlog6(error_pose)
    
    return error, jlog @ jacobian


def solve_inverse_kinematics(
    q: np.ndarray,
    com_target: np.ndarray,
    fixed_foot_pose: pin.SE3,
    moving_foot_pose: pin.SE3,
    torso_pose: pin.SE3,
    params: IKParams
) -> tuple:
    """Solve inverse kinematics with stance foot constraint.
    
    Minimizes:
        w_com * ||J_com * dq - e_com||^2
        + w_foot * ||J_foot * dq - e_foot||^2
        + w_torso * ||J_torso * dq - e_torso||^2
        + mu * ||dq||^2
        
    Subject to:
        J_fixed * dq = e_fixed  (stance foot constraint)
    
    Args:
        q: Current configuration
        com_target: Desired CoM position
        fixed_foot_pose: Desired stance foot pose
        moving_foot_pose: Desired swing foot pose
        torso_pose: Desired torso pose
        params: IK solver parameters
        
    Returns:
        q_next: Updated configuration
        dq: Velocity solution
    """
    model, data = params.model, params.data
    nv = model.nv
    
    locked_indices = set()
    if params.locked_joints:
        for j in params.locked_joints:
            if 0 <= j < model.njoints:
                i0 = model.idx_vs[j]
                i1 = model.idx_vs[j + 1] if j + 1 < model.njoints else nv
                locked_indices.update(range(i0, i1))
            elif 0 <= j < nv:
                locked_indices.add(j)
    active_indices = np.array(sorted(set(range(nv)) - locked_indices), dtype=int)
    
    def reduce_jacobian(J):
        return J[:, active_indices] if J is not None else None
    
    pin.computeCentroidalMap(model, data, q)
    com = pin.centerOfMass(model, data, q)
    J_com = pin.jacobianCenterOfMass(model, data, q)
    e_com = com_target - com
    
    e_fixed, J_fixed = _se3_error_and_jacobian(
        model, data, q, params.fixed_foot_frame, fixed_foot_pose
    )
    
    e_moving, J_moving = _se3_error_and_jacobian(
        model, data, q, params.moving_foot_frame, moving_foot_pose
    )
    
    e_torso_full, J_torso_full = _se3_error_and_jacobian(
        model, data, q, params.torso_frame, torso_pose
    )
    S_angular = np.zeros((3, 6))
    S_angular[0, 3] = S_angular[1, 4] = S_angular[2, 5] = 1.0
    e_torso = S_angular @ e_torso_full
    J_torso = S_angular @ J_torso_full
    
    J_com_r = reduce_jacobian(J_com)
    J_fixed_r = reduce_jacobian(J_fixed)
    J_moving_r = reduce_jacobian(J_moving)
    J_torso_r = reduce_jacobian(J_torso)
    n_active = len(active_indices)
    
    H = (
        params.w_com * (J_com_r.T @ J_com_r)
        + params.w_foot * (J_moving_r.T @ J_moving_r)
        + params.w_torso * (J_torso_r.T @ J_torso_r)
        + params.mu * np.eye(n_active)
    )
    
    g = (
        -params.w_com * (J_com_r.T @ e_com)
        - params.w_foot * (J_moving_r.T @ e_moving)
        - params.w_torso * (J_torso_r.T @ e_torso)
    )
    
    H = 0.5 * (H + H.T)
    
    dq_reduced = solve_qp(
        P=H, q=g,
        A=J_fixed_r, b=e_fixed,
        solver="osqp"
    )
    
    dq = np.zeros(nv)
    if dq_reduced is not None:
        dq[active_indices] = dq_reduced
    
    q_next = pin.integrate(model, q, dq)
    
    return q_next, dq
