from dataclasses import dataclass
from typing import List

import numpy as np

from v2.utils.state_machine import WalkingState


@dataclass(frozen=True)
class ObservationSpec:
    """Observation specification for policy inputs."""

    state_order: List[WalkingState]

    @staticmethod
    def default() -> "ObservationSpec":
        return ObservationSpec(
            state_order=[
                WalkingState.INIT,
                WalkingState.DS,
                WalkingState.SS_LEFT,
                WalkingState.SS_RIGHT,
                WalkingState.END,
            ]
        )

    @property
    def dim(self) -> int:
        # com_xy + com_vel_xy + dcm_xy + dcm_ref_xy + zmp_bounds_x + zmp_bounds_y
        # + phase_progress + state_one_hot + dcm_error + avg_speed + recent_violations
        return 2 + 2 + 2 + 2 + 2 + 2 + 1 + len(self.state_order) + 1 + 1 + 1


def build_observation(
    com: np.ndarray,
    com_vel: np.ndarray,
    dcm: np.ndarray,
    dcm_ref: np.ndarray,
    zmp_bounds_x: np.ndarray,
    zmp_bounds_y: np.ndarray,
    phase_progress: float,
    state: WalkingState,
    spec: ObservationSpec,
    dcm_error: float = 0.0,
    avg_speed: float = 0.0,
    recent_violations: float = 0.0,
) -> np.ndarray:
    """Build a flat observation vector.

    Observation layout:
        com_xy (2)
        com_vel_xy (2)
        dcm_xy (2)
        dcm_ref_xy (2)
        zmp_bounds_x_minmax (2)
        zmp_bounds_y_minmax (2)
        phase_progress (1)
        state_one_hot (len(state_order))
        dcm_error (1) - magnitude of DCM tracking error
        avg_speed (1) - average forward velocity over recent history
        recent_violations (1) - fraction of recent ZMP violations
    """
    com_xy = np.asarray(com[:2], dtype=float)
    com_vel_xy = np.asarray(com_vel[:2], dtype=float)
    dcm_xy = np.asarray(dcm[:2], dtype=float)
    dcm_ref_xy = np.asarray(dcm_ref[:2], dtype=float)

    bounds_x = np.array([zmp_bounds_x[0], zmp_bounds_x[1]], dtype=float)
    bounds_y = np.array([zmp_bounds_y[0], zmp_bounds_y[1]], dtype=float)

    one_hot = np.zeros(len(spec.state_order), dtype=float)
    for i, st in enumerate(spec.state_order):
        if st == state:
            one_hot[i] = 1.0
            break

    return np.concatenate(
        [
            com_xy,
            com_vel_xy,
            dcm_xy,
            dcm_ref_xy,
            bounds_x,
            bounds_y,
            np.array([float(phase_progress)]),
            one_hot,
            np.array([float(dcm_error)]),
            np.array([float(avg_speed)]),
            np.array([float(recent_violations)]),
        ]
    )
