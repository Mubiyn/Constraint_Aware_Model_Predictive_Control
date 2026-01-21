raise RuntimeError("Full-body MPC removed. Use centroidal MPC/ATP path.")

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

from v2.models.talos import TalosModel
from v2.simulation.bullet_env import BulletSimulator
from v2.controllers.full_body_mpc import FullBodyMPCController, FullBodyMPCParams


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _get_yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def snap_feet_to_ground(oMf_lf: pin.SE3, oMf_rf: pin.SE3, z_offset: float = 0.0) -> tuple[pin.SE3, pin.SE3]:
    yl = _get_yaw(oMf_lf.rotation)
    yr = _get_yaw(oMf_rf.rotation)
    pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_offset])
    pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_offset])
    return pin.SE3(_rotz(yl), pl), pin.SE3(_rotz(yr), pr)


def compute_base_from_foot(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    foot_frame_id: int,
    target_pose: pin.SE3,
) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_b = q[:3]
    q_b = q[3:7]
    oMb = pin.SE3(pin.Quaternion(q_b).toRotationMatrix(), p_b)
    bMf = oMb.inverse() * data.oMf[foot_frame_id]
    oMb_new = target_pose * bMf.inverse()
    return oMb_new


def main():
    parser = argparse.ArgumentParser(description="Full-body MPC standing test")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    args = parser.parse_args()

    talos_data_path = Path(__file__).parent / "talos_data"
    
    # --- 1. KINEMATICS SETUP (Floating Model) ---
    # We use this ONLY to calculate the correct base spawn height.
    talos_float = TalosModel(talos_data_path, reduced=False, fixed_base=False)
    q_float = talos_float.get_default_configuration()
    
    talos_float.update_kinematics(q_float)
    oMf_rf = talos_float.get_right_foot_pose()
    oMf_lf = talos_float.get_left_foot_pose()
    
    # Snap base so feet are at z=0 
    # Use 0.02m offset to ensure feet don't clip into ground
    lf_target, rf_target = snap_feet_to_ground(oMf_lf, oMf_rf, z_offset=0.02)
    oMb = compute_base_from_foot(talos_float.model, talos_float.data, q_float, talos_float.right_foot_id, rf_target)
    
    quat = pin.Quaternion(oMb.rotation)
    base_pos = oMb.translation
    # Do NOT override with 0.9, use calculated kinematics
    base_orn = np.array([quat.x, quat.y, quat.z, quat.w])
    
    print(f"Calculated Spawn Height: {base_pos[2]:.3f} m")
    
    # Extract the joint configuration (excluding base) for the controller
    q_joints_ref = q_float[7:].copy()

    # --- 2. CONTROLLER & PHYSICS SETUP (Fixed Base) ---
    # Pinocchio Model: Fixed Base (Root = Universe)
    # This simplifies dynamics ID for standing.
    talos = TalosModel(talos_data_path, reduced=False, fixed_base=True)
    
    urdf_path = talos_data_path / "urdf" / "talos_full_v2.urdf"
    if not urdf_path.exists():
        urdf_path = talos_data_path / "urdf" / "talos_full.urdf"

    # Physics: Fixed Base (Clamped)
    # NOTE: PyBullet "fixed_base=True" means the root link is static.
    # We must set its position once.
    sim = BulletSimulator(
        urdf_path=urdf_path,
        model=talos, # Use Fixed Model for mapping
        dt=0.01,
        gui=not args.no_gui,
        solver_iterations=200,
        fixed_base=True,
    )
    
    # Set PyBullet base to calculated height so feet touch ground
    sim.reset_base_pose(base_pos, base_orn)
    sim.reset_configuration(q_joints_ref)

    sim.pb.setGravity(0, 0, -9.81)
    
    name_to_pb = {}
    for jid in sim.pos_map.keys():
        name = sim.pb.getJointInfo(sim.robot_id, jid)[1].decode()
        name_to_pb[name] = jid

    arm_joint_ids = {
        jid
        for name, jid in name_to_pb.items()
        if name.startswith("arm_left_") or name.startswith("arm_right_")
    }
    arm_v_idx = set()
    for name in name_to_pb.keys():
        if name.startswith("arm_left_") or name.startswith("arm_right_"):
            model_jid = talos.model.getJointId(name)
            if model_jid > 0 and talos.model.joints[model_jid].nv == 1:
                arm_v_idx.add(talos.model.joints[model_jid].idx_v)

    position_control_joint_ids = arm_joint_ids
    leg_hold_joint_ids = {
        jid
        for name, jid in name_to_pb.items()
        if name.startswith("leg_left_") or name.startswith("leg_right_")
    }
    leg_v_idx = set()
    for name in name_to_pb.keys():
        if name.startswith("leg_left_") or name.startswith("leg_right_"):
            model_jid = talos.model.getJointId(name)
            if model_jid > 0 and talos.model.joints[model_jid].nv == 1:
                leg_v_idx.add(talos.model.joints[model_jid].idx_v)

    locked_v_idx = arm_v_idx

    # Start control loop immediately.

    # Controller params
    params = FullBodyMPCParams(
        dt=0.01,
        horizon=25,
        fz_min=0.0,
        fz_max_multiplier=5.0,
        mu=0.8,
        w_vdot=0.2,
        w_tau=1.0,
        vdot_max=50.0,
        tau_max=500.0
    )
    
    controller = FullBodyMPCController(talos.model, params)
    q_ref = q_joints_ref.copy()

    kp = 80.0
    kd = 12.0

    tau_limits = sim.get_joint_torque_limits()
    print("Fixed-base standing test")
    print(f"Torque limits: min={tau_limits.min():.1f}, max={tau_limits.max():.1f}")

    warm_steps = 100

    for step in range(args.steps):
        q = sim.get_configuration(talos.nq)
        
        # Build full velocity vector (nv)
        v = np.zeros(talos.nv)
        _, velocities = sim.get_joint_torques_velocities()
        joint_ids = list(sim.pos_map.keys())
        for idx in range(len(joint_ids)):
            jid = joint_ids[idx]
            v_idx = sim.vel_map[jid]
            v[v_idx] = velocities[idx]

        # Use contact constraints for both feet in fixed-base stance
        contact_active = (True, True)

        q_err = q - q_ref

        kp_vec = np.full(talos.nv, kp)
        kd_vec = np.full(talos.nv, kd)
        if leg_v_idx:
            kp_vec[list(leg_v_idx)] = 600.0
            kd_vec[list(leg_v_idx)] = 60.0
        if arm_v_idx:
            kp_vec[list(arm_v_idx)] = 50.0
            kd_vec[list(arm_v_idx)] = 8.0
        pin.computeAllTerms(talos.model, talos.data, q, v)
        g_joints = talos.data.nle
        tau_pd = -kp * q_err - kd * v + g_joints

        vdot_ref = -kp_vec * q_err - kd_vec * v
        if locked_v_idx:
            vdot_ref[list(locked_v_idx)] = 0.0

        # No ramp on tau_ref to ensure gravity compensation is always active
        tau, info = controller.solve(
            q,
            v,
            talos.left_foot_id,
            talos.right_foot_id,
            contact_active,
            vdot_ref=vdot_ref,
            tau_ref=tau_pd,
        )
        if info["optimal"]:
            tau_cmd = tau
        else:
            tau_cmd = tau_pd

        tau_cmd = np.clip(tau_cmd, -tau_limits, tau_limits)
        tau = tau_cmd
        if position_control_joint_ids:
            sim.apply_position_control_mask(q_ref, position_control_joint_ids, force_scale=0.8, position_gain=0.4)
        sim.apply_torque_control(tau_cmd, disabled_joint_ids=position_control_joint_ids)
        sim.step()

        if step % 50 == 0:
            tau_norm = float(np.linalg.norm(tau))
            tau_cmd_norm = float(np.linalg.norm(tau_cmd))
            q_err_norm = float(np.linalg.norm(q_err))
            print(
                f"Step {step:4d} | "
                f"tau_norm={tau_norm:.2f} | tau_cmd_norm={tau_cmd_norm:.2f} | "
                f"q_err={q_err_norm:.3f} | status={info['status']}"
            )

    sim.close()


if __name__ == "__main__":
    main()
