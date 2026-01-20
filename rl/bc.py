from pathlib import Path
from typing import Tuple

import numpy as np

from v2.rl.policy import LinearPolicy


def fit_linear_policy(
    dataset_path: Path,
    output_path: Path,
    l2_reg: float = 1e-6,
) -> LinearPolicy:
    """Fit a linear policy using ridge regression.

    Args:
        dataset_path: Path to .npz dataset with observations/actions.
        output_path: Path to save the trained policy.
        l2_reg: L2 regularization strength.

    Returns:
        Trained LinearPolicy.
    """
    data = np.load(dataset_path)
    observations = data["observations"]
    actions = data["actions"]

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Dataset must be 2D arrays.")

    n_samples, obs_dim = observations.shape
    _, action_dim = actions.shape

    x = observations
    y = actions

    xtx = x.T @ x
    xty = x.T @ y

    reg = l2_reg * np.eye(obs_dim)
    weights = np.linalg.solve(xtx + reg, xty).T
    bias = y.mean(axis=0) - weights @ x.mean(axis=0)

    policy = LinearPolicy(weights=weights, bias=bias)
    policy.save(output_path)
    return policy


def load_dataset_shapes(dataset_path: Path) -> Tuple[int, int]:
    data = np.load(dataset_path)
    observations = data["observations"]
    actions = data["actions"]
    return observations.shape[1], actions.shape[1]
