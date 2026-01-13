"""
Walking state machine for managing gait phases.

Controls transitions between:
- INIT: Initial weight transfer phase
- DS: Double support (both feet in contact)
- SS_LEFT: Single support on left foot
- SS_RIGHT: Single support on right foot
- END: Terminal phase
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from v2.utils.foot import Foot


class WalkingState(Enum):
    INIT = 0
    DS = 1
    SS_LEFT = 2
    SS_RIGHT = 3
    END = 4


@dataclass
class WalkingParams:
    """Timing parameters for walking phases.
    
    Attributes:
        t_init: Duration of initialization phase (s)
        t_end: Duration of terminal phase (s)
        t_ss: Duration of single support phase (s)
        t_ds: Duration of double support phase (s)
        force_threshold: Contact force threshold (N)
    """
    t_init: float = 1.5
    t_end: float = 1.0
    t_ss: float = 0.6
    t_ds: float = 0.2
    force_threshold: float = 50.0


class WalkingStateMachine:
    """Finite state machine for walking gait control.
    
    Manages transitions between walking phases based on time
    and contact force feedback.
    """
    
    def __init__(self, params: WalkingParams):
        self.params = params
        self.state = WalkingState.INIT
        self.t_start = 0.0
        
        self.steps_pose: Optional[np.ndarray] = None
        self.steps_foot: Optional[List[Foot]] = None
        self.step_idx = 0
        
    def set_steps(self, steps_pose: np.ndarray, steps_foot: List[Foot]):
        """Set the planned step sequence.
        
        Args:
            steps_pose: Array of step positions, shape (N, 3)
            steps_foot: List of which foot takes each step
        """
        self.steps_pose = steps_pose
        self.steps_foot = steps_foot
        self.step_idx = 0
        
    def get_elapsed_time(self, t: float) -> float:
        """Get time elapsed in current state."""
        return t - self.t_start
    
    def get_phase_progress(self, t: float) -> float:
        """Get normalized phase progress [0, 1].
        
        Args:
            t: Current time
            
        Returns:
            Progress through current phase, 0 at start, 1 at end
        """
        elapsed = self.get_elapsed_time(t)
        
        if self.state == WalkingState.INIT:
            return min(1.0, elapsed / self.params.t_init)
        elif self.state == WalkingState.DS:
            return min(1.0, elapsed / self.params.t_ds)
        elif self.state in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            return min(1.0, elapsed / self.params.t_ss)
        elif self.state == WalkingState.END:
            return min(1.0, elapsed / self.params.t_end)
        return 0.0
    
    def update(
        self,
        t: float,
        rf_force: float,
        lf_force: float
    ) -> bool:
        """Update state machine based on time and contact forces.
        
        Args:
            t: Current time
            rf_force: Right foot contact force (N)
            lf_force: Left foot contact force (N)
            
        Returns:
            True if state changed
        """
        if self.steps_pose is None:
            return False
            
        elapsed = self.get_elapsed_time(t)
        state_changed = False
        
        if self.state == WalkingState.INIT:
            if elapsed >= self.params.t_init:
                state_changed = self._transition_to_single_support(t)
                
        elif self.state == WalkingState.DS:
            if elapsed >= self.params.t_ds:
                state_changed = self._transition_to_single_support(t)
                
        elif self.state == WalkingState.SS_RIGHT:
            if elapsed >= self.params.t_ss:
                if lf_force > self.params.force_threshold or elapsed > self.params.t_ss * 1.2:
                    state_changed = self._transition_from_single_support(t)
                    
        elif self.state == WalkingState.SS_LEFT:
            if elapsed >= self.params.t_ss:
                if rf_force > self.params.force_threshold or elapsed > self.params.t_ss * 1.2:
                    state_changed = self._transition_from_single_support(t)
                    
        elif self.state == WalkingState.END:
            if elapsed >= self.params.t_end:
                self.steps_pose = None
                
        return state_changed
    
    def _transition_to_single_support(self, t: float) -> bool:
        """Transition from DS/INIT to single support."""
        if self.step_idx >= len(self.steps_foot):
            return False
            
        next_foot = self.steps_foot[self.step_idx]
        
        if next_foot == Foot.LEFT:
            self.state = WalkingState.SS_RIGHT
        else:
            self.state = WalkingState.SS_LEFT
            
        self.t_start = t
        return True
    
    def _transition_from_single_support(self, t: float) -> bool:
        """Transition from single support to DS or END."""
        self.step_idx += 1
        self.t_start = t
        
        if self.step_idx >= len(self.steps_foot):
            self.state = WalkingState.END
        else:
            self.state = WalkingState.DS
            
        return True
    
    def get_stance_foot(self) -> Optional[Foot]:
        """Get the current stance foot."""
        if self.state == WalkingState.SS_LEFT:
            return Foot.LEFT
        elif self.state == WalkingState.SS_RIGHT:
            return Foot.RIGHT
        return None
    
    def get_swing_foot(self) -> Optional[Foot]:
        """Get the current swing foot."""
        if self.state == WalkingState.SS_LEFT:
            return Foot.RIGHT
        elif self.state == WalkingState.SS_RIGHT:
            return Foot.LEFT
        return None
    
    def is_single_support(self) -> bool:
        """Check if in single support phase."""
        return self.state in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT)
    
    def is_double_support(self) -> bool:
        """Check if in double support phase."""
        return self.state in (WalkingState.DS, WalkingState.INIT, WalkingState.END)
    
    def is_walking(self) -> bool:
        """Check if walking sequence is active."""
        return self.steps_pose is not None and self.state != WalkingState.END
    
    def get_current_step_target(self) -> Optional[np.ndarray]:
        """Get position target for current step."""
        if self.steps_pose is None or self.step_idx >= len(self.steps_pose):
            return None
        return self.steps_pose[self.step_idx].copy()
    
    def get_previous_step_position(self) -> Optional[np.ndarray]:
        """Get position of the previous step (stance foot start)."""
        if self.steps_pose is None or self.step_idx == 0:
            return None
        return self.steps_pose[self.step_idx - 1].copy()
