"""
MPC-based walking controller main simulation script.

This script demonstrates the Model Predictive Control approach for
humanoid locomotion using the LIPM model, as described in the ATP.

Uses DCM (Divergent Component of Motion) based MPC for stable walking:
- DCM provides natural stability through capture point tracking
- MPC enforces ZMP constraints within support polygon
- Real-time QP solving for online optimization
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pinocchio as pin

from v2.models.lipm import LIPMParams, DecoupledLIPM
from v2.models.talos import TalosModel
from v2.controllers.mpc import MPCParams, DecoupledMPCController, DCMTrajectoryGenerator
from v2.utils.foot import (
    Foot, FootParams, FootTrajectoryGenerator,
    compute_steps_sequence, compute_support_polygon, compute_zmp_bounds,
    get_double_support_polygon
)
from v2.utils.state_machine import WalkingState, WalkingParams, WalkingStateMachine
from v2.utils.inverse_kinematic import IKParams, solve_inverse_kinematics
from v2.simulation.bullet_env import BulletSimulator
from v2.rl.dataset import DatasetCollector
from v2.rl.policy import LinearPolicy
from v2.rl.obs import ObservationSpec, build_observation
from v2.rl.bc import fit_linear_policy


@dataclass
class SimulationConfig:
    """Configuration for the walking simulation.
    
    Parameters tuned for stable Talos walking with DCM-MPC.
    """
    dt: float = 1.0 / 100.0
    n_steps: int = 100
    stride_length: float = 0.12
    foot_height: float = 0.05
    t_ss: float = 0.95
    t_ds: float = 1.5
    t_init: float = 2.0
    t_end: float = 1.0
    z_c: float = 0.87
    mpc_horizon: float = 1.6
    solver_iterations: int = 500
    use_mpc: bool = True
    disturbance_force: float = 0.0
    disturbance_time: float = 5.0
    disturbance_duration: float = 0.1


@dataclass
class SimulationMetrics:
    duration: float
    fell: bool
    zmp_compliance: float
    max_zmp_violation: float
    avg_dcm_error: float
    max_dcm_error: float
    avg_solve_ms: float
    max_solve_ms: float


@dataclass
class MPCValidationThresholds:
    min_zmp_compliance: float = 0.95
    max_avg_dcm_error: float = 0.05


@dataclass
class RLConfig:
    dataset_path: Optional[Path] = None
    policy_path: Optional[Path] = None
    policy_mode: str = "residual"  # "direct" or "residual"
    residual_clip: float = 0.03


def compute_zmp_reference(
    t: float,
    state_machine: WalkingStateMachine,
    lf_pos: np.ndarray,
    rf_pos: np.ndarray,
    horizon: int,
    dt: float,
    params: WalkingParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ZMP reference trajectory over the MPC horizon.
    
    Args:
        t: Current time
        state_machine: Walking state machine
        lf_pos: Left foot position
        rf_pos: Right foot position
        horizon: MPC horizon length
        dt: Time step
        params: Walking timing parameters
        
    Returns:
        zmp_ref_x: X-axis ZMP reference, shape (horizon,)
        zmp_ref_y: Y-axis ZMP reference, shape (horizon,)
    """
    zmp_ref_x = np.zeros(horizon)
    zmp_ref_y = np.zeros(horizon)
    
    state = state_machine.state
    step_idx = state_machine.step_idx
    elapsed = state_machine.get_elapsed_time(t)
    steps = state_machine.steps_pose
    steps_foot = state_machine.steps_foot
    
    if steps is None:
        return zmp_ref_x, zmp_ref_y
    
    time_in_phase = elapsed
    current_state = state
    current_step = step_idx
    
    for i in range(horizon):
        if current_state == WalkingState.INIT:
            mid_x = (lf_pos[0] + rf_pos[0]) / 2.0
            mid_y = (lf_pos[1] + rf_pos[1]) / 2.0
            
            if current_step < len(steps_foot):
                next_foot = steps_foot[current_step]
                if next_foot == Foot.LEFT:
                    target_y = rf_pos[1]
                else:
                    target_y = lf_pos[1]
            else:
                target_y = mid_y
                
            progress = min(1.0, time_in_phase / params.t_init)
            zmp_ref_x[i] = mid_x
            zmp_ref_y[i] = mid_y + progress * (target_y - mid_y)
            
            time_in_phase += dt
            if time_in_phase >= params.t_init:
                time_in_phase = 0.0
                if current_step < len(steps_foot):
                    next_foot = steps_foot[current_step]
                    current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
                    
        elif current_state == WalkingState.DS:
            progress = min(1.0, time_in_phase / params.t_ds)
            
            just_completed_step = current_step - 1
            
            if just_completed_step < 0:
                start_pos = (lf_pos + rf_pos) / 2
                end_pos = start_pos
            else:
                landed_foot = steps_foot[just_completed_step]
                landed_pos = steps[just_completed_step]
                
                if landed_foot == Foot.LEFT:
                    if just_completed_step == 0:
                        stance_pos = rf_pos
                    else:
                        stance_pos = steps[just_completed_step - 1]
                else:
                    if just_completed_step == 0:
                        stance_pos = lf_pos
                    else:
                        stance_pos = steps[just_completed_step - 1]
                
                start_pos = stance_pos
                end_pos = landed_pos
                
            zmp_ref_x[i] = start_pos[0] + progress * (end_pos[0] - start_pos[0])
            zmp_ref_y[i] = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                
            time_in_phase += dt
            if time_in_phase >= params.t_ds:
                time_in_phase = 0.0
                current_step += 1
                if current_step < len(steps_foot):
                    next_foot = steps_foot[current_step]
                    current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
                else:
                    current_state = WalkingState.END
                    
        elif current_state in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            if current_step > 0:
                stance_pos = steps[current_step - 1]
            else:
                stance_pos = rf_pos if current_state == WalkingState.SS_RIGHT else lf_pos
                
            zmp_ref_x[i] = stance_pos[0]
            zmp_ref_y[i] = stance_pos[1]
            
            time_in_phase += dt
            if time_in_phase >= params.t_ss:
                time_in_phase = 0.0
                current_step += 1
                if current_step >= len(steps_foot):
                    current_state = WalkingState.END
                else:
                    current_state = WalkingState.DS
                
        elif current_state == WalkingState.END:
            zmp_ref_x[i] = (lf_pos[0] + rf_pos[0]) / 2.0
            zmp_ref_y[i] = (lf_pos[1] + rf_pos[1]) / 2.0
            
    return zmp_ref_x, zmp_ref_y


def compute_zmp_bounds_for_horizon(
    t: float,
    state_machine: WalkingStateMachine,
    lf_pos: np.ndarray,
    rf_pos: np.ndarray,
    horizon: int,
    dt: float,
    params: WalkingParams,
    foot_params: FootParams
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Compute ZMP constraint bounds over the MPC horizon."""
    x_min = np.zeros(horizon)
    x_max = np.zeros(horizon)
    y_min = np.zeros(horizon)
    y_max = np.zeros(horizon)
    
    state = state_machine.state
    step_idx = state_machine.step_idx
    elapsed = state_machine.get_elapsed_time(t)
    steps = state_machine.steps_pose
    steps_foot = state_machine.steps_foot
    
    time_in_phase = elapsed
    current_state = state
    current_step = step_idx
    
    current_lf = lf_pos.copy()
    current_rf = rf_pos.copy()
    
    for i in range(horizon):
        if current_state in (WalkingState.INIT, WalkingState.DS, WalkingState.END):
            polygon = get_double_support_polygon(
                current_lf, current_rf, 0.0, 0.0, foot_params
            )
        elif current_state == WalkingState.SS_LEFT:
            polygon = compute_support_polygon(current_lf, 0.0, foot_params)
        elif current_state == WalkingState.SS_RIGHT:
            polygon = compute_support_polygon(current_rf, 0.0, foot_params)
        else:
            polygon = get_double_support_polygon(
                current_lf, current_rf, 0.0, 0.0, foot_params
            )
            
        bounds = polygon.bounds
        x_min[i] = bounds[0]
        y_min[i] = bounds[1]
        x_max[i] = bounds[2]
        y_max[i] = bounds[3]
        
        time_in_phase += dt
        
        if current_state == WalkingState.INIT and time_in_phase >= params.t_init:
            time_in_phase = 0.0
            if current_step < len(steps_foot):
                next_foot = steps_foot[current_step]
                current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
                
        elif current_state == WalkingState.DS and time_in_phase >= params.t_ds:
            time_in_phase = 0.0
            current_step += 1
            if current_step < len(steps_foot):
                next_foot = steps_foot[current_step]
                current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
            else:
                current_state = WalkingState.END
                
        elif current_state in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            if time_in_phase >= params.t_ss:
                time_in_phase = 0.0
                if current_step < len(steps):
                    target = steps[current_step]
                    if current_state == WalkingState.SS_LEFT:
                        current_rf = target.copy()
                    else:
                        current_lf = target.copy()
                current_step += 1
                if current_step >= len(steps_foot):
                    current_state = WalkingState.END
                else:
                    current_state = WalkingState.DS
                
    return (x_min, x_max), (y_min, y_max)


def compute_dcm_reference(
    t: float,
    state_machine: WalkingStateMachine,
    lf_pos: np.ndarray,
    rf_pos: np.ndarray,
    horizon: int,
    dt: float,
    params: WalkingParams,
    omega: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DCM reference trajectory over the MPC horizon.
    
    Uses the fundamental DCM property: for stable walking, the DCM should
    be between the current ZMP (stance foot) and the next footstep target.
    
    During single support: DCM moves from stance foot toward swing target
    During double support: DCM transitions between footsteps
    
    The DCM offset from ZMP is proportional to the distance to the next footstep
    and decreases as the phase progresses.
    
    Args:
        t: Current time
        state_machine: Walking state machine
        lf_pos: Left foot position (3D)
        rf_pos: Right foot position (3D)
        horizon: MPC horizon length
        dt: Time step
        params: Walking timing parameters
        omega: LIPM natural frequency
        
    Returns:
        dcm_ref_x: X-axis DCM reference, shape (horizon,)
        dcm_ref_y: Y-axis DCM reference, shape (horizon,)
    """
    zmp_ref_x, zmp_ref_y = compute_zmp_reference(
        t, state_machine, lf_pos, rf_pos, horizon, dt, params
    )
    
    dcm_ref_x = zmp_ref_x.copy()
    dcm_ref_y = zmp_ref_y.copy()
    
    state = state_machine.state
    step_idx = state_machine.step_idx
    elapsed = state_machine.get_elapsed_time(t)
    steps = state_machine.steps_pose
    steps_foot = state_machine.steps_foot
    
    if steps is None or len(steps) == 0:
        return dcm_ref_x, dcm_ref_y
    
    dcm_lead_factor = 0.05
    
    time_in_phase = elapsed
    current_state = state
    current_step = step_idx
    
    for i in range(horizon):
        if current_state == WalkingState.INIT:
            time_in_phase += dt
            if time_in_phase >= params.t_init:
                time_in_phase = 0.0
                if current_step < len(steps_foot):
                    next_foot = steps_foot[current_step]
                    current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
                    
        elif current_state in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            phase_progress = min(1.0, time_in_phase / params.t_ss)
            
            if current_step < len(steps):
                swing_target = steps[current_step][:2]
                zmp_pos = np.array([zmp_ref_x[i], zmp_ref_y[i]])
                
                direction = swing_target - zmp_pos
                dist_to_target = np.linalg.norm(direction)
                
                lead_distance = dcm_lead_factor * dist_to_target * (1.0 - phase_progress)
                
                if dist_to_target > 0.01:
                    unit_dir = direction / dist_to_target
                    dcm_ref_x[i] = zmp_ref_x[i] + lead_distance * unit_dir[0]
                    dcm_ref_y[i] = zmp_ref_y[i] + lead_distance * unit_dir[1]
            
            time_in_phase += dt
            if time_in_phase >= params.t_ss:
                time_in_phase = 0.0
                current_step += 1
                if current_step >= len(steps_foot):
                    current_state = WalkingState.END
                else:
                    current_state = WalkingState.DS
                    
        elif current_state == WalkingState.DS:
            phase_progress = min(1.0, time_in_phase / params.t_ds)
            
            if current_step < len(steps):
                next_target = steps[current_step][:2]
                zmp_pos = np.array([zmp_ref_x[i], zmp_ref_y[i]])
                
                direction = next_target - zmp_pos
                dist_to_target = np.linalg.norm(direction)
                
                lead_distance = dcm_lead_factor * dist_to_target * (1.0 - phase_progress * 0.5)
                
                if dist_to_target > 0.01:
                    unit_dir = direction / dist_to_target
                    dcm_ref_x[i] = zmp_ref_x[i] + lead_distance * unit_dir[0]
                    dcm_ref_y[i] = zmp_ref_y[i] + lead_distance * unit_dir[1]
            
            time_in_phase += dt
            if time_in_phase >= params.t_ds:
                time_in_phase = 0.0
                current_step += 1
                if current_step < len(steps_foot):
                    next_foot = steps_foot[current_step]
                    current_state = WalkingState.SS_RIGHT if next_foot == Foot.LEFT else WalkingState.SS_LEFT
                else:
                    current_state = WalkingState.END
                    
        elif current_state == WalkingState.END:
            pass
            
    return dcm_ref_x, dcm_ref_y


def snap_feet_to_ground(
    oMf_lf: pin.SE3, 
    oMf_rf: pin.SE3, 
    z_offset: float = 0.0
) -> Tuple[pin.SE3, pin.SE3]:
    """Project feet poses to ground plane."""
    def rotz(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def get_yaw(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))
    
    yl = get_yaw(oMf_lf.rotation)
    yr = get_yaw(oMf_rf.rotation)
    
    pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_offset])
    pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_offset])
    
    return pin.SE3(rotz(yl), pl), pin.SE3(rotz(yr), pr)


def compute_base_from_foot(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    foot_frame_id: int,
    target_pose: pin.SE3
) -> pin.SE3:
    """Compute base pose that places foot at target pose."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    p_b = q[:3]
    q_b = q[3:7]
    oMb = pin.SE3(pin.Quaternion(q_b).toRotationMatrix(), p_b)
    
    bMf = oMb.inverse() * data.oMf[foot_frame_id]
    oMb_new = target_pose * bMf.inverse()
    
    return oMb_new


def run_simulation(
    config: SimulationConfig,
    gui: bool = True,
    max_steps: Optional[int] = None,
    print_every: int = 50,
    rl_config: Optional[RLConfig] = None,
    return_metrics: bool = False,
):
    """Run the MPC walking simulation.
    
    Args:
        config: Simulation configuration
        gui: Whether to show GUI
        max_steps: Maximum simulation steps (None for full simulation)
        print_every: Print status every N steps
    """
    talos_data_path = Path(__file__).parent / "talos_data"
    urdf_path = talos_data_path / "urdf" / "talos_full_v2.urdf"
    
    if not urdf_path.exists():
        urdf_path = talos_data_path / "urdf" / "talos_full.urdf"
        
    print(f"Loading Talos model from {talos_data_path}")
    talos = TalosModel(talos_data_path, reduced=False)
    q_init = talos.get_default_configuration()
    
    talos.update_kinematics(q_init)
    oMf_lf = talos.get_left_foot_pose()
    oMf_rf = talos.get_right_foot_pose()
    
    oMf_lf, oMf_rf = snap_feet_to_ground(oMf_lf, oMf_rf, z_offset=0.0)
    
    oMb = compute_base_from_foot(
        talos.model, talos.data, q_init,
        talos.right_foot_id, oMf_rf
    )
    
    R = oMb.rotation
    p = oMb.translation
    quat = pin.Quaternion(R)
    q_init[:3] = p
    q_init[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
    
    talos.update_kinematics(q_init)
    
    feet_mid = 0.5 * (oMf_lf.translation + oMf_rf.translation)
    com_init = np.array([feet_mid[0], feet_mid[1], config.z_c])
    
    print(f"Initial CoM target: {com_init}")
    print(f"Left foot: {oMf_lf.translation}")
    print(f"Right foot: {oMf_rf.translation}")
    
    print("Initializing PyBullet simulation...")
    sim = BulletSimulator(
        urdf_path=urdf_path,
        model=talos,
        dt=config.dt,
        gui=gui,
        solver_iterations=config.solver_iterations
    )
    sim.reset_configuration(q_init)
    
    print("Phase 1: Zero-gravity settling...")
    sim.pb.setGravity(0, 0, 0)
    for _ in range(100):
        sim.step()
    sim.reset_configuration(q_init)
    sim.pb.setGravity(0, 0, -9.81)
    
    locked_joints = talos.get_locked_joints_idx()
    
    print("Phase 2: Stabilization with IK...")
    ik_params_stab = IKParams(
        model=talos.model,
        data=talos.data,
        fixed_foot_frame=talos.left_foot_id,
        moving_foot_frame=talos.right_foot_id,
        torso_frame=talos.torso_id,
        w_com=10.0,
        w_foot=100.0,
        w_torso=10.0,
        mu=1e-4,
        dt=config.dt,
        locked_joints=locked_joints
    )
    
    com_target_stab = com_init.copy()
    torso_pose_stab = talos.data.oMf[talos.torso_id].copy()
    q = q_init.copy()
    
    talos.update_kinematics(q)
    q, _ = solve_inverse_kinematics(
        q, com_target_stab,
        oMf_lf, oMf_rf,
        torso_pose_stab, ik_params_stab
    )
    q_init = q.copy()
    sim.reset_configuration(q_init)
    
    stabilization_steps = int(1.0 / config.dt)
    for i in range(stabilization_steps):
        talos.update_kinematics(q)
        q, _ = solve_inverse_kinematics(
            q, com_target_stab,
            oMf_lf, oMf_rf,
            torso_pose_stab, ik_params_stab
        )
        
        sim.apply_position_control(q)
        sim.step()
        
        if i % 50 == 0:
            actual_com = sim.get_com_position()
            base_height = sim.get_base_height()
            print(f"  Stabilization step {i}: CoM=[{actual_com[0]:.3f}, {actual_com[1]:.3f}], Height={base_height:.3f}")
    
    actual_com = sim.get_com_position()
    com_init = actual_com.copy()
    print(f"Stabilization complete. Actual CoM: {com_init}")
    print(f"Base height: {sim.get_base_height():.3f}")
        
    walking_params = WalkingParams(
        t_init=config.t_init,
        t_end=config.t_end,
        t_ss=config.t_ss,
        t_ds=config.t_ds
    )
    
    state_machine = WalkingStateMachine(walking_params)
    
    lf_pos = oMf_lf.translation.copy()
    rf_pos = oMf_rf.translation.copy()
    
    steps_pose, steps_foot = compute_steps_sequence(
        rf_initial=rf_pos,
        lf_initial=lf_pos,
        n_steps=config.n_steps,
        stride_length=config.stride_length
    )
    
    steps_pose = steps_pose[1:]
    steps_foot = steps_foot[1:]
    
    state_machine.set_steps(steps_pose, steps_foot)
    
    print(f"Planned {len(steps_pose)} steps, first swing: {steps_foot[0].name}")
    
    horizon = int(config.mpc_horizon / config.dt)
    mpc_params = MPCParams(
        horizon=horizon,
        dt=config.dt,
        z_c=config.z_c,
        Q_dcm=50.0,
        R_zmp=5e-3,  # Smoother ZMP commands
        Q_terminal=500.0,
        zmp_margin=0.015,  # Larger margin for safety
        max_zmp_rate=1.0  # Slower ZMP rate
    )
    
    mpc = DecoupledMPCController(mpc_params)
    omega = mpc.omega
    
    lipm_params = LIPMParams(z_c=config.z_c, dt=config.dt)
    lipm = DecoupledLIPM(lipm_params)
    
    state_x = np.array([com_init[0], 0.0, 0.0])
    state_y = np.array([com_init[1], 0.0, 0.0])
    
    foot_traj = FootTrajectoryGenerator(foot_height=config.foot_height)
    foot_params = FootParams(length=0.20, width=0.10)
    
    # Moderate weight for torso task
    w_torso_value = 10.0
    
    ik_params = IKParams(
        model=talos.model,
        data=talos.data,
        fixed_foot_frame=talos.right_foot_id,
        moving_foot_frame=talos.left_foot_id,
        torso_frame=talos.torso_id,
        w_com=100.0,
        w_foot=500.0,
        w_torso=w_torso_value,
        mu=1e-4,
        dt=config.dt,
        locked_joints=locked_joints
    )
    
    t = 0.0
    
    total_time = (
        config.t_init 
        + config.n_steps * (config.t_ss + config.t_ds) 
        + config.t_end 
        + 2.0
    )
    n_sim_steps = int(total_time / config.dt)
    
    if max_steps is not None:
        n_sim_steps = min(n_sim_steps, max_steps)
    
    current_lf = oMf_lf.copy()
    current_rf = oMf_rf.copy()
    swing_start_pos = None
    swing_target_pos = None
    ss_com_start = None
    ds_com_start = None
    end_com_start = None
    mpc_used = False
    zmp_des = np.zeros(2)
    
    # Integral term for lateral drift correction
    y_drift_integral = 0.0
    y_drift_prev = 0.0
    
    # Torso roll correction tracking
    roll_integral = 0.0
    
    torso_pose = pin.SE3(np.eye(3), np.array([0, 0, 1.0]))
    
    # Diagnostic logging for asymmetry analysis
    foot_landing_log = []  # Track actual vs planned foot positions
    force_asymmetry_log = []  # Track left/right force differences
    
    print(f"\nStarting simulation for {n_sim_steps} steps...")
    print("-" * 60)
    
    solve_times = []
    zmp_violations = []
    zmp_history = []
    dcm_tracking_errors = []
    fell = False

    observation_spec = ObservationSpec.default()
    dataset_collector = None
    policy = None

    if rl_config is not None:
        if rl_config.dataset_path is not None:
            dataset_collector = DatasetCollector(
                obs_dim=observation_spec.dim,
                action_dim=2,
            )
        if rl_config.policy_path is not None:
            policy = LinearPolicy.load(rl_config.policy_path)

    state_violation_counts = {WalkingState.INIT: [0, 0], WalkingState.SS_LEFT: [0, 0], 
                              WalkingState.SS_RIGHT: [0, 0], WalkingState.DS: [0, 0], 
                              WalkingState.END: [0, 0]}
    
    for step in range(n_sim_steps):
        rf_force, lf_force = sim.get_contact_forces()
        
        # Log force asymmetry
        total_force = rf_force + lf_force
        if total_force > 10.0:  # Only when in contact
            force_balance = (rf_force - lf_force) / total_force if total_force > 0 else 0.0
            force_asymmetry_log.append({
                't': t,
                'rf_force': rf_force,
                'lf_force': lf_force,
                'balance': force_balance,  # Positive = more weight on right
                'state': state_machine.state.name
            })
        
        state_changed = state_machine.update(t, rf_force, lf_force)
        
        if state_changed:
            if state_machine.is_single_support():
                swing_foot = state_machine.get_swing_foot()
                ss_com_start = sim.get_com_position()[:2].copy()
                
                if swing_foot == Foot.LEFT:
                    swing_start_pos = current_lf.translation.copy()
                    ik_params.fixed_foot_frame = talos.right_foot_id
                    ik_params.moving_foot_frame = talos.left_foot_id
                else:
                    swing_start_pos = current_rf.translation.copy()
                    ik_params.fixed_foot_frame = talos.left_foot_id
                    ik_params.moving_foot_frame = talos.right_foot_id
                    
                target = state_machine.get_current_step_target()
                if target is not None:
                    swing_target_pos = target.copy()
                    swing_target_pos[2] = 0.0
                    print(f"SS Phase: step_idx={state_machine.step_idx}, swing={swing_foot.name}")
                    print(f"  swing_start: [{swing_start_pos[0]:.3f}, {swing_start_pos[1]:.3f}]")
                    print(f"  swing_target: [{swing_target_pos[0]:.3f}, {swing_target_pos[1]:.3f}]")
                    
                    # Log planned landing position
                    foot_landing_log.append({
                        't': t,
                        'step_idx': state_machine.step_idx,
                        'foot': swing_foot.name,
                        'planned_pos': swing_target_pos[:2].copy(),
                        'actual_pos': None  # Will be filled when foot lands
                    })
        
        if state_machine.state == WalkingState.INIT:
            elapsed = state_machine.get_elapsed_time(t)
            progress = min(1.0, elapsed / walking_params.t_init)
            
            mid_x = (oMf_lf.translation[0] + oMf_rf.translation[0]) / 2.0
            mid_y = (oMf_lf.translation[1] + oMf_rf.translation[1]) / 2.0
            
            if len(state_machine.steps_foot) > 0:
                next_foot = state_machine.steps_foot[0]
                if next_foot == Foot.LEFT:
                    target_x = oMf_rf.translation[0]
                    target_y = oMf_rf.translation[1]
                else:
                    target_x = oMf_lf.translation[0]
                    target_y = oMf_lf.translation[1]
            else:
                target_x = mid_x
                target_y = mid_y
            
            progress_smooth = 0.5 * (1 - np.cos(np.pi * progress))
            
            com_target_x = mid_x + progress_smooth * (target_x - mid_x)
            com_target_y = mid_y + progress_smooth * (target_y - mid_y)
            
            solve_times.append(0.0)
            com_target = np.array([com_target_x, com_target_y, config.z_c])
            mpc_used = False
        elif state_machine.state == WalkingState.END:
            progress = state_machine.get_phase_progress(t)
            progress_smooth = 0.5 * (1 - np.cos(np.pi * progress))
            
            if state_changed:
                end_com_start = sim.get_com_position()[:2].copy()
            
            mid_x = (current_lf.translation[0] + current_rf.translation[0]) / 2.0
            mid_y = (current_lf.translation[1] + current_rf.translation[1]) / 2.0
            
            com_target_x = end_com_start[0] + progress_smooth * (mid_x - end_com_start[0])
            com_target_y = end_com_start[1] + progress_smooth * (mid_y - end_com_start[1])
            
            com_target = np.array([com_target_x, com_target_y, config.z_c])
            solve_times.append(0.0)
            mpc_used = False
        else:
            zmp_ref_x, zmp_ref_y = compute_zmp_reference(
                t, state_machine, current_lf.translation, current_rf.translation,
                horizon, config.dt, walking_params
            )
            
            dcm_ref_x, dcm_ref_y = compute_dcm_reference(
                t, state_machine, current_lf.translation, current_rf.translation,
                horizon, config.dt, walking_params, omega
            )
            
            bounds_x, bounds_y = compute_zmp_bounds_for_horizon(
                t, state_machine, current_lf.translation, current_rf.translation,
                horizon, config.dt, walking_params, foot_params
            )
            
            actual_com = sim.get_com_position()
            actual_com_vel = sim.get_com_velocity()
            
            dcm_current = mpc.compute_dcm(actual_com, actual_com_vel)
            
            if config.use_mpc:
                start_time = time.time()
                zmp_des, info = mpc.compute_control(
                    actual_com,
                    actual_com_vel,
                    dcm_ref_x, dcm_ref_y,
                    bounds_x, bounds_y
                )
                solve_time = time.time() - start_time
                solve_times.append(solve_time * 1000)
                
                zmp_des[0] = np.clip(zmp_des[0], bounds_x[0][0], bounds_x[1][0])
                zmp_des[1] = np.clip(zmp_des[1], bounds_y[0][0], bounds_y[1][0])
            else:
                k_dcm = 3.0
                dcm_error_x = dcm_current[0] - dcm_ref_x[0]
                dcm_error_y = dcm_current[1] - dcm_ref_y[0]
                
                zmp_x_fb = dcm_current[0] + k_dcm * dcm_error_x / omega
                zmp_y_fb = dcm_current[1] + k_dcm * dcm_error_y / omega
                
                zmp_des = np.array([
                    np.clip(zmp_x_fb, bounds_x[0][0], bounds_x[1][0]),
                    np.clip(zmp_y_fb, bounds_y[0][0], bounds_y[1][0])
                ])
                solve_times.append(0.0)
            
            zmp_des_mpc = zmp_des.copy()

            phase_progress = state_machine.get_phase_progress(t)
            obs = build_observation(
                com=actual_com,
                com_vel=actual_com_vel,
                dcm=dcm_current,
                dcm_ref=np.array([dcm_ref_x[0], dcm_ref_y[0]]),
                zmp_bounds_x=np.array([bounds_x[0][0], bounds_x[1][0]]),
                zmp_bounds_y=np.array([bounds_y[0][0], bounds_y[1][0]]),
                phase_progress=phase_progress,
                state=state_machine.state,
                spec=observation_spec,
            )

            if dataset_collector is not None:
                dataset_collector.add(obs, zmp_des_mpc)

            control_label = "MPC"

            if policy is not None:
                policy_action = np.asarray(policy.act(obs), dtype=float).reshape(-1)
                if policy_action.shape[0] != 2:
                    raise ValueError("Policy action must be 2D [zmp_x, zmp_y]")
                if rl_config.policy_mode == "direct":
                    zmp_des = policy_action
                    control_label = "POLICY"
                elif rl_config.policy_mode == "residual":
                    residual = np.clip(
                        policy_action,
                        -rl_config.residual_clip,
                        rl_config.residual_clip,
                    )
                    zmp_des = zmp_des + residual
                    control_label = "RESIDUAL"
                else:
                    raise ValueError(f"Unknown policy_mode: {rl_config.policy_mode}")

            zmp_des[0] = np.clip(zmp_des[0], bounds_x[0][0], bounds_x[1][0])
            zmp_des[1] = np.clip(zmp_des[1], bounds_y[0][0], bounds_y[1][0])

            if state_changed:
                print(f"  DCM Ctrl: dcm=[{dcm_current[0]:.3f}, {dcm_current[1]:.3f}]")
                print(f"  DCM Ctrl: dcm_ref=[{dcm_ref_x[0]:.3f}, {dcm_ref_y[0]:.3f}]")
                print(f"  DCM Ctrl: zmp_des=[{zmp_des[0]:.3f}, {zmp_des[1]:.3f}]")
            
            mpc_used = True
            
            # Compute CoM target with DCM feedback
            # Base target is ZMP reference (stance foot)
            # Add correction based on DCM error to help convergence
            dcm_error = np.array([dcm_current[0] - dcm_ref_x[0], dcm_current[1] - dcm_ref_y[0]])
            
            # If DCM is ahead of reference, move CoM backward (toward ZMP)
            # If DCM is behind reference, move CoM forward (toward DCM ref)
            # Use different gains for x and y
            k_correction_x = 0.6
            k_correction_y = 0.6
            correction_x = -k_correction_x * dcm_error[0]
            correction_y = -k_correction_y * dcm_error[1]
            
            # Adaptive lateral drift correction based on actual measured position
            # Original working approach with refined time constants
            
            # Measure drift from midpoint between feet
            mid_y = (current_lf.translation[1] + current_rf.translation[1]) / 2.0
            y_drift = actual_com[1] - mid_y  # Positive = left of center
            
            # Feedforward compensation for predicted force asymmetry
            # Three-phase model based on observed force balance trajectory:
            # Phase 1 (0-50s): -0.85 → +0.35 (rapid transition)
            # Phase 2 (50-150s): +0.35 → +0.55 (slow drift, ~0.002/s)
            # Phase 3 (150s+): +0.55 → +1.00 (explosion, ~0.018/s)
            if t <= 50.0:
                predicted_bias = -0.85 + 1.2 * (t / 50.0)
            elif t <= 150.0:
                predicted_bias = 0.35 + 0.002 * (t - 50.0)
            else:
                predicted_bias = 0.55 + 0.018 * (t - 150.0)
                predicted_bias = min(predicted_bias, 1.2)
            feedforward_offset = 0.03 * predicted_bias
            
            # Original working integral settings
            y_drift_integral = 0.998 * y_drift_integral + y_drift * config.dt
            y_drift_integral = np.clip(y_drift_integral, -0.3, 0.3)
            
            # Original working PI gains
            k_drift = 0.15
            k_integral = 0.3
            drift_correction = -k_drift * y_drift - k_integral * y_drift_integral + feedforward_offset
            correction_y += drift_correction
            
            # Blend between ZMP ref and DCM-corrected position
            com_target_x = zmp_ref_x[0] + correction_x
            com_target_y = zmp_ref_y[0] + correction_y
            
            com_target = np.array([com_target_x, com_target_y, config.z_c])
            
            actual_zmp = sim.get_zmp_position()
            if actual_zmp is None:
                print(f"\nWARNING: Lost contact at t={t:.2f}s - robot may have fallen!")
                fell = True
                break
            zmp_history.append(actual_zmp.copy())
            
            # Diagnostic: Monitor torso orientation
            talos.update_kinematics(q)
            torso_actual = talos.data.oMf[talos.torso_id]
            rpy = pin.rpy.matrixToRpy(torso_actual.rotation)
            
            if step % 500 == 0:  # Every 5 seconds (approx)
                print(f"  [DEBUG t={t:.2f}s] Torso RPY: {np.degrees(rpy)} deg")
                print(f"  [DEBUG t={t:.2f}s] CoM Error: {actual_com - com_target}")
                print(f"  [DEBUG t={t:.2f}s] Drift: y_drift={y_drift:.4f}, integral={y_drift_integral:.4f}, correction={drift_correction:.4f}")
                
                # Print force balance summary
                if len(force_asymmetry_log) > 0:
                    recent_balance = np.mean([log['balance'] for log in force_asymmetry_log[-100:]])
                    print(f"  [DEBUG t={t:.2f}s] Force balance (R-L)/Total: {recent_balance:.3f} (positive=right-heavy)")
                
            x_in_bounds = bounds_x[0][0] <= actual_zmp[0] <= bounds_x[1][0]
            y_in_bounds = bounds_y[0][0] <= actual_zmp[1] <= bounds_y[1][0]
            
            current_walking_state = state_machine.state
            state_violation_counts[current_walking_state][1] += 1  # Total samples
            
            if not (x_in_bounds and y_in_bounds):
                violation = max(
                    max(bounds_x[0][0] - actual_zmp[0], actual_zmp[0] - bounds_x[1][0], 0),
                    max(bounds_y[0][0] - actual_zmp[1], actual_zmp[1] - bounds_y[1][0], 0)
                )
                zmp_violations.append(violation)
                state_violation_counts[current_walking_state][0] += 1  # Violations
            else:
                zmp_violations.append(0.0)
                
            dcm_err = np.sqrt((dcm_current[0] - dcm_ref_x[0])**2 + (dcm_current[1] - dcm_ref_y[0])**2)
            dcm_tracking_errors.append(dcm_err)
        
        if state_machine.is_single_support() and swing_start_pos is not None and swing_target_pos is not None:
            phase = state_machine.get_phase_progress(t)
            swing_pos = foot_traj.generate(
                swing_start_pos, swing_target_pos, np.array([phase])
            )[0]
            
            swing_foot = state_machine.get_swing_foot()
            if swing_foot == Foot.LEFT:
                current_lf = pin.SE3(np.eye(3), swing_pos)
                fixed_foot_pose = current_rf
                moving_foot_pose = current_lf
            else:
                current_rf = pin.SE3(np.eye(3), swing_pos)
                fixed_foot_pose = current_lf
                moving_foot_pose = current_rf
        else:
            if state_machine.state == WalkingState.SS_LEFT:
                fixed_foot_pose = current_lf
                moving_foot_pose = current_rf
            elif state_machine.state == WalkingState.DS:
                just_completed = state_machine.step_idx - 1
                if just_completed >= 0 and state_machine.steps_foot[just_completed] == Foot.LEFT:
                    fixed_foot_pose = current_rf
                    moving_foot_pose = current_lf
                    ik_params.fixed_foot_frame = talos.right_foot_id
                    ik_params.moving_foot_frame = talos.left_foot_id
                else:
                    fixed_foot_pose = current_lf
                    moving_foot_pose = current_rf
                    ik_params.fixed_foot_frame = talos.left_foot_id
                    ik_params.moving_foot_frame = talos.right_foot_id
            elif state_machine.state == WalkingState.END:
                last_step_idx = len(state_machine.steps_foot) - 1
                if last_step_idx >= 0:
                    last_swing_foot = state_machine.steps_foot[last_step_idx]
                    if last_swing_foot == Foot.LEFT:
                        current_lf = pin.SE3(np.eye(3), swing_target_pos if swing_target_pos is not None else current_lf.translation)
                    else:
                        current_rf = pin.SE3(np.eye(3), swing_target_pos if swing_target_pos is not None else current_rf.translation)
                fixed_foot_pose = current_lf
                moving_foot_pose = current_rf
                ik_params.fixed_foot_frame = talos.left_foot_id
                ik_params.moving_foot_frame = talos.right_foot_id
            else:
                fixed_foot_pose = current_rf
                moving_foot_pose = current_lf
                
        torso_pose.translation = com_target
        torso_pose.rotation = np.eye(3)  # Keep torso upright
        
        talos.update_kinematics(q)
        
        q, _ = solve_inverse_kinematics(
            q, com_target,
            fixed_foot_pose, moving_foot_pose,
            torso_pose, ik_params
        )
        
        sim.apply_position_control(q)
        
        if config.disturbance_force > 0:
            dist_start = config.disturbance_time
            dist_end = config.disturbance_time + config.disturbance_duration
            if dist_start <= t < dist_end:
                force = np.array([0.0, config.disturbance_force, 0.0])
                sim.apply_external_force(force, link_index=-1)
                if abs(t - dist_start) < config.dt:
                    state_str = state_machine.state.name
                    print(f"\n*** APPLYING {config.disturbance_force}N LATERAL PUSH at t={t:.2f}s (state={state_str}) ***\n")
        
        sim.step()
        
        if gui:
            sim.update_camera(com_target)
        
        if mpc_used:
            disp_info = f"ZMP: [{zmp_des[0]:.3f}, {zmp_des[1]:.3f}]"
        else:
            disp_info = f"Target CoM: [{com_target[0]:.3f}, {com_target[1]:.3f}]"
            
        if step % print_every == 0:
            actual_com = sim.get_com_position()
            zmp = sim.get_zmp_position()
            base_height = sim.get_base_height()
            
            if mpc_used:
                mode_tag = f"[{control_label}]"
            else:
                mode_tag = "[Direct]"
            print(
                f"Step {step:5d} | t={t:.3f}s | {state_machine.state.name:10s} {mode_tag:8s} | "
                f"CoM: [{actual_com[0]:.3f}, {actual_com[1]:.3f}] | "
                f"{disp_info} | "
                f"H: {base_height:.3f}"
            )
            
        t += config.dt
        
        if not state_machine.is_walking() and state_machine.state == WalkingState.END:
            elapsed_end = state_machine.get_elapsed_time(t)
            if elapsed_end > config.t_end:
                print("\nWalking sequence completed!")
                break
                
        base_height = sim.get_base_height()
        if base_height < 0.5:
            print(f"\nRobot fell! Base height: {base_height:.3f}")
            fell = True
            break
            
    print("-" * 60)
    print(f"Simulation completed at t={t:.3f}s")
    print(f"Average MPC solve time: {np.mean(solve_times):.2f}ms")
    print(f"Max MPC solve time: {np.max(solve_times):.2f}ms")
    
    # Asymmetry diagnostics
    if len(force_asymmetry_log) > 0:
        print("\n" + "=" * 60)
        print("ASYMMETRY DIAGNOSTICS")
        print("=" * 60)
        
        # Force balance analysis
        balances = [log['balance'] for log in force_asymmetry_log]
        avg_balance = np.mean(balances)
        print(f"\nForce Balance (R-L)/Total:")
        print(f"  Average: {avg_balance:.3f} (positive = right-biased)")
        print(f"  Std Dev: {np.std(balances):.3f}")
        print(f"  Min: {min(balances):.3f}, Max: {max(balances):.3f}")
        
        # Per-state force balance
        print(f"\nForce Balance by Walking State:")
        for state_name in ['SS_LEFT', 'SS_RIGHT', 'DS']:
            state_balances = [log['balance'] for log in force_asymmetry_log if log['state'] == state_name]
            if state_balances:
                print(f"  {state_name:10s}: {np.mean(state_balances):+.3f} ± {np.std(state_balances):.3f}")
        
        # Foot landing accuracy
        if len(foot_landing_log) > 0:
            print(f"\nFoot Landing Accuracy:")
            print(f"  Total planned steps: {len(foot_landing_log)}")
            
    print("=" * 60)
    
    if len(zmp_violations) > 0:
        n_violations = sum(1 for v in zmp_violations if v > 0.001)
        compliance_rate = 100.0 * (1 - n_violations / len(zmp_violations))
        max_violation = max(zmp_violations) if zmp_violations else 0.0
        print(f"ZMP Constraint Compliance: {compliance_rate:.1f}%")
        print(f"ZMP Violations: {n_violations}/{len(zmp_violations)} samples")
        print(f"Max ZMP Violation: {max_violation:.4f}m")
        
        print("\nViolations by state:")
        for state, (violations, total) in state_violation_counts.items():
            if total > 0:
                pct = 100.0 * violations / total
                print(f"  {state.name}: {violations}/{total} ({pct:.1f}% violation rate)")
        
    if len(dcm_tracking_errors) > 0:
        avg_dcm_err = np.mean(dcm_tracking_errors)
        max_dcm_err = max(dcm_tracking_errors)
        print(f"Average DCM Tracking Error: {avg_dcm_err:.4f}m")
        print(f"Max DCM Tracking Error: {max_dcm_err:.4f}m")
    
    final_com = sim.get_com_position()
    print(f"Final CoM position: [{final_com[0]:.3f}, {final_com[1]:.3f}, {final_com[2]:.3f}]")
    
    if dataset_collector is not None and rl_config is not None and rl_config.dataset_path is not None:
        dataset_collector.save(rl_config.dataset_path)
        print(f"Saved behavior cloning dataset to {rl_config.dataset_path}")

    sim.close()

    if return_metrics:
        if len(zmp_violations) > 0:
            n_violations = sum(1 for v in zmp_violations if v > 0.001)
            compliance_rate = 1.0 - n_violations / len(zmp_violations)
            max_violation = max(zmp_violations)
        else:
            compliance_rate = 1.0
            max_violation = 0.0

        if len(dcm_tracking_errors) > 0:
            avg_dcm_err = float(np.mean(dcm_tracking_errors))
            max_dcm_err = float(max(dcm_tracking_errors))
        else:
            avg_dcm_err = 0.0
            max_dcm_err = 0.0

        avg_solve = float(np.mean(solve_times)) if solve_times else 0.0
        max_solve = float(np.max(solve_times)) if solve_times else 0.0

        return SimulationMetrics(
            duration=t,
            fell=fell,
            zmp_compliance=compliance_rate,
            max_zmp_violation=max_violation,
            avg_dcm_error=avg_dcm_err,
            max_dcm_error=max_dcm_err,
            avg_solve_ms=avg_solve,
            max_solve_ms=max_solve,
        )

    return None


def validate_mpc(
    config: SimulationConfig,
    thresholds: MPCValidationThresholds,
    gui: bool = False,
) -> bool:
    metrics = run_simulation(
        config,
        gui=gui,
        max_steps=None,
        print_every=200,
        rl_config=None,
        return_metrics=True,
    )

    if metrics is None:
        print("No metrics returned.")
        return False

    zmp_ok = metrics.zmp_compliance >= thresholds.min_zmp_compliance
    dcm_ok = metrics.avg_dcm_error <= thresholds.max_avg_dcm_error
    fall_ok = not metrics.fell

    print("\nMPC Validation Summary")
    print("-" * 40)
    print(f"ZMP compliance: {metrics.zmp_compliance:.3f} (min {thresholds.min_zmp_compliance:.3f})")
    print(f"Avg DCM error: {metrics.avg_dcm_error:.4f} (max {thresholds.max_avg_dcm_error:.4f})")
    print(f"Fell: {metrics.fell}")

    passed = zmp_ok and dcm_ok and fall_ok
    print(f"Validation result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MPC Walking Controller Simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of walking steps")
    parser.add_argument("--stride", type=float, default=0.12, help="Stride length in meters")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum simulation steps")
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N steps")
    parser.add_argument("--disturbance", type=float, default=0.0, 
                        help="Apply lateral disturbance force (N) during single support")
    parser.add_argument("--disturbance-time", type=float, default=5.0,
                        help="Time (s) to apply disturbance")
    parser.add_argument("--validate-mpc", action="store_true", help="Run MPC validation and exit")
    parser.add_argument("--collect-bc", type=str, default=None, help="Path to save MPC rollout dataset (.npz)")
    parser.add_argument("--train-bc", type=str, default=None, help="Path to dataset (.npz) for training a linear policy")
    parser.add_argument("--bc-output", type=str, default="v2/rl/linear_policy.npz", help="Output path for trained policy")
    parser.add_argument("--policy", type=str, default=None, help="Path to a trained linear policy (.npz)")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="residual",
        choices=["direct", "residual"],
        help="How to apply policy: direct ZMP or residual on MPC",
    )
    parser.add_argument("--residual-clip", type=float, default=0.03, help="Residual ZMP clip (m)")
    args = parser.parse_args()
    
    config = SimulationConfig(
        n_steps=args.n_steps,
        stride_length=args.stride,
        disturbance_force=args.disturbance,
        disturbance_time=args.disturbance_time
    )
    
    if args.train_bc is not None:
        dataset_path = Path(args.train_bc)
        output_path = Path(args.bc_output)
        fit_linear_policy(dataset_path, output_path)
        print(f"Trained linear policy saved to {output_path}")
        return

    if args.validate_mpc:
        validate_mpc(config, MPCValidationThresholds(), gui=not args.no_gui)
        return

    rl_config = None
    if args.collect_bc is not None or args.policy is not None:
        rl_config = RLConfig(
            dataset_path=Path(args.collect_bc) if args.collect_bc is not None else None,
            policy_path=Path(args.policy) if args.policy is not None else None,
            policy_mode=args.policy_mode,
            residual_clip=args.residual_clip,
        )

    run_simulation(
        config,
        gui=not args.no_gui,
        max_steps=args.max_steps,
        print_every=args.print_every,
        rl_config=rl_config,
    )


if __name__ == "__main__":
    main()
