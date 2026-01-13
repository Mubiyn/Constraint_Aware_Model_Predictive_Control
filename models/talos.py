"""
Talos humanoid robot model wrapper using Pinocchio.

Provides a clean interface to load the Talos URDF and access
frame IDs, joint configurations, and kinematics computations.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin


def _set_joint(q: np.ndarray, model: pin.Model, joint_name: str, val: float) -> None:
    """Set a single joint value in configuration vector."""
    jid = model.getJointId(joint_name)
    if jid > 0 and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val


def _q_from_base_pose(q: np.ndarray, oMb: pin.SE3) -> np.ndarray:
    """Update configuration vector with base pose."""
    R = oMb.rotation
    p = oMb.translation
    quat = pin.Quaternion(R)
    q_out = q.copy()
    q_out[:3] = p
    q_out[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
    return q_out


class TalosModel:
    """Talos humanoid robot model using Pinocchio.
    
    Loads the Talos URDF and provides access to kinematic computations,
    frame IDs, and joint configuration utilities.
    """
    
    def __init__(self, urdf_path: Path, reduced: bool = False):
        """Initialize Talos model.
        
        Args:
            urdf_path: Path to the talos_data directory
            reduced: If True, lock upper body joints to reduce DOF
        """
        self.urdf_path = urdf_path
        self.reduced = reduced
        
        full_urdf = urdf_path / "urdf" / "talos_full_v2.urdf"
        if not full_urdf.exists():
            full_urdf = urdf_path / "urdf" / "talos_full.urdf"
        
        package_dirs = [str(urdf_path.parent)]
            
        self.full_model, self.full_col_model, self.full_vis_model = pin.buildModelsFromUrdf(
            str(full_urdf), package_dirs, pin.JointModelFreeFlyer()
        )
        
        q = pin.neutral(self.full_model)
        
        _set_joint(q, self.full_model, "leg_left_4_joint", 0.0)
        _set_joint(q, self.full_model, "leg_right_4_joint", 0.0)
        _set_joint(q, self.full_model, "arm_right_4_joint", -1.5)
        _set_joint(q, self.full_model, "arm_left_4_joint", -1.5)
        
        if self.reduced:
            joints_to_lock = list(self._get_locked_joints_idx())
            self.model, self.geom = pin.buildReducedModel(
                self.full_model, self.full_col_model, joints_to_lock, q
            )
            _, self.vis = pin.buildReducedModel(
                self.full_model, self.full_vis_model, joints_to_lock, q
            )
        else:
            self.model = self.full_model
            self.geom = self.full_col_model
            self.vis = self.full_vis_model
            
        self.data = self.model.createData()
        
        self.left_foot_id = self.model.getFrameId("left_sole_link")
        self.right_foot_id = self.model.getFrameId("right_sole_link")
        self.torso_id = self.model.getFrameId("torso_1_link")
        
    def get_default_configuration(self) -> np.ndarray:
        """Get default standing configuration."""
        q = pin.neutral(self.model)
        
        if not self.reduced:
            _set_joint(q, self.model, "leg_left_4_joint", 0.0)
            _set_joint(q, self.model, "leg_right_4_joint", 0.0)
            _set_joint(q, self.model, "arm_right_4_joint", -1.5)
            _set_joint(q, self.model, "arm_left_4_joint", -1.5)
            
        _set_joint(q, self.model, "leg_left_1_joint", 0.0)
        _set_joint(q, self.model, "leg_left_2_joint", 0.0)
        _set_joint(q, self.model, "leg_left_3_joint", -0.5)
        _set_joint(q, self.model, "leg_left_4_joint", 1.0)
        _set_joint(q, self.model, "leg_left_5_joint", -0.6)
        
        _set_joint(q, self.model, "leg_right_1_joint", 0.0)
        _set_joint(q, self.model, "leg_right_2_joint", 0.0)
        _set_joint(q, self.model, "leg_right_3_joint", -0.5)
        _set_joint(q, self.model, "leg_right_4_joint", 1.0)
        _set_joint(q, self.model, "leg_right_5_joint", -0.6)
        
        self.update_kinematics(q)
        return q
    
    def update_kinematics(self, q: np.ndarray) -> None:
        """Update forward kinematics for the current configuration."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
    def get_frame_pose(self, frame_id: int) -> pin.SE3:
        """Get the world pose of a frame."""
        return self.data.oMf[frame_id].copy()
    
    def get_left_foot_pose(self) -> pin.SE3:
        """Get left foot world pose."""
        return self.get_frame_pose(self.left_foot_id)
    
    def get_right_foot_pose(self) -> pin.SE3:
        """Get right foot world pose."""
        return self.get_frame_pose(self.right_foot_id)
    
    def get_com_position(self, q: np.ndarray) -> np.ndarray:
        """Compute center of mass position."""
        return pin.centerOfMass(self.model, self.data, q)
    
    def get_joint_id(self, name: str) -> Optional[int]:
        """Get joint index in q vector from joint name."""
        jid = self.model.getJointId(name)
        n_joints = len(self.model.joints)
        return self.model.joints[jid].idx_q if jid < n_joints else None
    
    def update_configuration_with_base(self, q: np.ndarray, oMb: pin.SE3) -> np.ndarray:
        """Update configuration with new base pose."""
        return _q_from_base_pose(q, oMb)
    
    @staticmethod
    def _get_locked_joints_idx():
        """Get indices of joints to lock in reduced model."""
        return range(14, 46)
    
    def get_locked_joints_idx(self):
        """Get indices of joints to lock (arms, hands, head)."""
        return list(self._get_locked_joints_idx())
    
    @staticmethod
    def get_rf_link_name() -> str:
        return "leg_right_6_link"
    
    @staticmethod
    def get_lf_link_name() -> str:
        return "leg_left_6_link"
    
    @property
    def nq(self) -> int:
        """Number of configuration variables."""
        return self.model.nq
    
    @property
    def nv(self) -> int:
        """Number of velocity variables."""
        return self.model.nv
