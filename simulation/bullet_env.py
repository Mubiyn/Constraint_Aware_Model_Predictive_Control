"""
PyBullet simulation environment for bipedal walking.

Provides a wrapper around PyBullet for loading the Talos robot,
applying controls, and reading sensor data.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pybullet as pb
import pybullet_data


def _build_joint_maps(robot_id: int, model) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build mappings between PyBullet and Pinocchio joint indices.
    
    Args:
        robot_id: PyBullet body ID
        model: Pinocchio model
        
    Returns:
        pos_map: PyBullet joint ID -> Pinocchio q index
        vel_map: PyBullet joint ID -> Pinocchio v index
    """
    pos_map = {}
    vel_map = {}
    
    name_to_bullet = {}
    for jid in range(pb.getNumJoints(robot_id)):
        info = pb.getJointInfo(robot_id, jid)
        name = info[1].decode()
        jtype = info[2]
        if jtype != pb.JOINT_FIXED:
            name_to_bullet[name] = jid
            
    for j in range(1, model.njoints):
        if model.joints[j].nv == 0:
            continue
        name = model.names[j]
        if name in name_to_bullet:
            bid = name_to_bullet[name]
            pos_map[bid] = model.joints[j].idx_q
            vel_map[bid] = model.idx_vs[j]
            
    return pos_map, vel_map


def _link_index(body_id: int, link_name: str) -> Optional[int]:
    """Get PyBullet link index from link name."""
    for i in range(pb.getNumJoints(body_id)):
        info = pb.getJointInfo(body_id, i)
        if info[12].decode() == link_name:
            return i
    return None


class BulletSimulator:
    """PyBullet simulation environment for Talos walking.
    
    Handles physics simulation, robot control, and sensor reading.
    """
    
    def __init__(
        self,
        urdf_path: Path,
        model,
        dt: float = 1.0/240.0,
        gui: bool = True,
        solver_iterations: int = 200
    ):
        """Initialize PyBullet simulation.
        
        Args:
            urdf_path: Path to robot URDF file
            model: Pinocchio model for joint mapping
            dt: Simulation time step
            gui: Whether to launch GUI
            solver_iterations: Number of physics solver iterations
        """
        self.dt = dt
        
        self.cid = pb.connect(
            pb.GUI if gui else pb.DIRECT,
            options="--window_title=MPC Walking --width=1920 --height=1080"
        )
        
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(dt)
        pb.setRealTimeSimulation(0)
        
        pb.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=solver_iterations,
            numSubSteps=1,
            useSplitImpulse=1,
            splitImpulsePenetrationThreshold=0.01,
            contactSlop=0.005,
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.05
        )
        
        self.plane_id = pb.loadURDF("plane.urdf")
        
        self.robot_id = pb.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            flags=pb.URDF_MERGE_FIXED_LINKS
        )
        
        self.pos_map, self.vel_map = _build_joint_maps(self.robot_id, model.model)
        
        self.rf_link_id = _link_index(self.robot_id, model.get_rf_link_name())
        self.lf_link_id = _link_index(self.robot_id, model.get_lf_link_name())
        
        self.log_id = None
        
    @property
    def pb(self):
        """Access to pybullet module."""
        return pb
        
    def step(self):
        """Advance simulation by one time step."""
        pb.stepSimulation()
        
    def reset_configuration(self, q: np.ndarray):
        """Set robot configuration from Pinocchio q vector.
        
        Args:
            q: Configuration vector with base pose (7) + joints
        """
        p_base = q[:3].tolist()
        q_base = q[3:7].tolist()
        
        _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(self.robot_id, -1)
        p_com, q_com = pb.multiplyTransforms(p_base, q_base, p_li, q_li)
        
        pb.resetBasePositionAndOrientation(self.robot_id, p_com, q_com)
        
        for jid, qid in self.pos_map.items():
            pb.resetJointState(self.robot_id, jid, float(q[qid]), 0.0)
            
    def apply_position_control(self, q: np.ndarray):
        """Apply position control to robot joints.
        
        Args:
            q: Desired configuration vector
        """
        for jid, qid in self.pos_map.items():
            max_force = pb.getJointInfo(self.robot_id, jid)[10]
            pb.setJointMotorControl2(
                self.robot_id,
                jid,
                pb.POSITION_CONTROL,
                targetPosition=q[qid],
                positionGain=0.4,
                force=max_force * 0.8
            )
            
    def get_configuration(self, nq: int) -> np.ndarray:
        """Read current configuration from PyBullet.
        
        Args:
            nq: Size of configuration vector
            
        Returns:
            Configuration vector [base_pos, base_quat, joints]
        """
        q = np.zeros(nq)
        
        com_pos, com_quat = pb.getBasePositionAndOrientation(self.robot_id)
        
        _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(self.robot_id, -1)
        p_ib, q_ib = pb.invertTransform(p_li, q_li)
        p_base, q_base = pb.multiplyTransforms(com_pos, com_quat, p_ib, q_ib)
        
        q[:7] = np.concatenate([p_base, q_base])
        
        for jid, qid in self.pos_map.items():
            q[qid] = pb.getJointState(self.robot_id, jid)[0]
            
        return q
    
    def get_contact_forces(self) -> Tuple[float, float]:
        """Get vertical contact forces for both feet.
        
        Returns:
            (right_foot_force, left_foot_force) in Newtons
        """
        rf_contacts = pb.getContactPoints(
            bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=self.rf_link_id
        )
        lf_contacts = pb.getContactPoints(
            bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=self.lf_link_id
        )
        
        rf_force = sum(c[9] for c in rf_contacts) if rf_contacts else 0.0
        lf_force = sum(c[9] for c in lf_contacts) if lf_contacts else 0.0
        
        return rf_force, lf_force
    
    def get_base_height(self) -> float:
        """Get robot base height above ground."""
        pos, _ = pb.getBasePositionAndOrientation(self.robot_id)
        return pos[2]
    
    def get_com_position(self) -> np.ndarray:
        """Compute whole-body center of mass position.
        
        Returns:
            CoM position [x, y, z]
        """
        base_pos, base_orn = pb.getBasePositionAndOrientation(self.robot_id)
        dyn_info = pb.getDynamicsInfo(self.robot_id, -1)
        m_base = dyn_info[0]
        
        p_inertial = dyn_info[3]
        q_inertial = dyn_info[4]
        base_com = pb.multiplyTransforms(base_pos, base_orn, p_inertial, q_inertial)[0]
        
        m_total = m_base
        com_sum = m_base * np.array(base_com)
        
        for link_idx in range(pb.getNumJoints(self.robot_id)):
            state = pb.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
            link_com = np.array(state[0])
            m_link = pb.getDynamicsInfo(self.robot_id, link_idx)[0]
            com_sum += m_link * link_com
            m_total += m_link
            
        return com_sum / m_total
    
    def get_com_velocity(self) -> np.ndarray:
        """Compute whole-body center of mass velocity.
        
        Returns:
            CoM velocity [vx, vy, vz]
        """
        base_vel, base_ang_vel = pb.getBaseVelocity(self.robot_id)
        base_vel = np.array(base_vel)
        
        dyn_info = pb.getDynamicsInfo(self.robot_id, -1)
        m_base = dyn_info[0]
        
        m_total = m_base
        vel_sum = m_base * base_vel
        
        for link_idx in range(pb.getNumJoints(self.robot_id)):
            state = pb.getLinkState(
                self.robot_id, link_idx, 
                computeLinkVelocity=True, 
                computeForwardKinematics=True
            )
            link_vel = np.array(state[6])
            m_link = pb.getDynamicsInfo(self.robot_id, link_idx)[0]
            vel_sum += m_link * link_vel
            m_total += m_link
            
        return vel_sum / m_total

    def get_total_mass(self) -> float:
        """Get total mass of base and all links."""
        m_total = pb.getDynamicsInfo(self.robot_id, -1)[0]
        for link_idx in range(pb.getNumJoints(self.robot_id)):
            m_total += pb.getDynamicsInfo(self.robot_id, link_idx)[0]
        return float(m_total)

    def get_joint_torques_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get motor torques and joint velocities for actuated joints."""
        joint_ids = list(self.pos_map.keys())
        torques = np.zeros(len(joint_ids))
        velocities = np.zeros(len(joint_ids))
        for i, jid in enumerate(joint_ids):
            state = pb.getJointState(self.robot_id, jid)
            velocities[i] = state[1]
            torques[i] = state[3]
        return torques, velocities

    def get_zmp_position(self) -> Optional[np.ndarray]:
        """Estimate ZMP from contact forces.
        
        Returns:
            ZMP position [x, y, 0] or None if no contact
        """
        contacts = pb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id)
        
        if not contacts:
            return None
            
        F = np.zeros(3)
        M = np.zeros(3)
        
        for cp in contacts:
            pos = np.array(cp[6])
            normal = np.array(cp[7])
            force_mag = cp[9]
            
            f = force_mag * normal
            F += f
            M += np.cross(pos, f)
            
        if abs(F[2]) < 1e-6:
            return None
            
        px = -M[1] / F[2]
        py = M[0] / F[2]
        
        return np.array([px, py, 0.0])
    
    def update_camera(self, target_pos: np.ndarray):
        """Update camera to follow a position.
        
        Args:
            target_pos: Position to look at [x, y, z]
        """
        pb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=target_pos.tolist()
        )
        
    def apply_external_force(
        self, 
        force: np.ndarray, 
        position: Optional[np.ndarray] = None,
        link_index: int = -1
    ):
        """Apply external force to the robot.
        
        Args:
            force: Force vector [fx, fy, fz] in Newtons (world frame)
            position: Application point (local link frame), defaults to link origin
            link_index: Link to apply force to (-1 for base)
        """
        if position is None:
            position = [0, 0, 0]
        else:
            position = position.tolist()
            
        pb.applyExternalForce(
            objectUniqueId=self.robot_id,
            linkIndex=link_index,
            forceObj=force.tolist(),
            posObj=position,
            flags=pb.WORLD_FRAME
        )
        
    def get_torso_link_id(self) -> int:
        """Get the link ID of the torso for applying disturbances."""
        for i in range(pb.getNumJoints(self.robot_id)):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[12].decode()
            if 'torso' in name.lower():
                return i
        return -1
        
    def start_recording(self, filename: str = "recording.mp4"):
        """Start video recording."""
        self.log_id = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, filename)
        
    def stop_recording(self):
        """Stop video recording."""
        if self.log_id is not None:
            pb.stopStateLogging(self.log_id)
            self.log_id = None
            
    def close(self):
        """Disconnect from PyBullet."""
        pb.disconnect()
