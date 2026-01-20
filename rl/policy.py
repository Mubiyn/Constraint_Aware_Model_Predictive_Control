from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Policy:
    """Base policy interface."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LinearPolicy(Policy):
    """Linear policy: action = W * obs + b."""

    weights: np.ndarray
    bias: np.ndarray

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float).reshape(-1)
        return self.weights @ obs + self.bias

    @classmethod
    def load(cls, path: Path) -> "LinearPolicy":
        data = np.load(path)
        weights = data["weights"]
        bias = data["bias"]
        return cls(weights=weights, bias=bias)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=self.bias)
