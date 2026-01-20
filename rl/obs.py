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
        return 2 + 2 + 2 + 2 + 4 + 1 + len(self.state_order)


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
        ]
    )
