from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class DatasetCollector:
    """Collect (observation, action) pairs for behavior cloning."""

    obs_dim: int
    action_dim: int
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)

    def add(self, obs: np.ndarray, action: np.ndarray):
        obs = np.asarray(obs, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Observation dim mismatch: {obs.shape[0]} != {self.obs_dim}")
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dim mismatch: {action.shape[0]} != {self.action_dim}")
        self.observations.append(obs)
        self.actions.append(action)

    def save(self, path: Path):
        if not self.observations:
            raise ValueError("No samples collected.")
        obs = np.stack(self.observations, axis=0)
        actions = np.stack(self.actions, axis=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, observations=obs, actions=actions)
