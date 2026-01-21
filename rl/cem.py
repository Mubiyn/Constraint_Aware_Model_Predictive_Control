from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import uuid

import numpy as np

from typing import TYPE_CHECKING

from v2.rl.obs import ObservationSpec
from v2.rl.policy import LinearPolicy

if TYPE_CHECKING:
    from v2.main_mpc import SimulationConfig


@dataclass
class CEMConfig:
    iterations: int = 20
    population: int = 24
    elite_frac: float = 0.25
    init_std: float = 0.3
    min_std: float = 0.02
    eval_steps: Optional[int] = None
    residual_clip: float = 0.03
    action_dim: int = 2
    policy_mode: str = "residual"
    seed: int = 42


def _flatten_policy(weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.concatenate([weights.reshape(-1), bias.reshape(-1)])


def _unflatten_policy(vec: np.ndarray, action_dim: int, obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    w_size = action_dim * obs_dim
    weights = vec[:w_size].reshape(action_dim, obs_dim)
    bias = vec[w_size:w_size + action_dim]
    return weights, bias


def _score_metrics(metrics, policy_mode: str = "residual") -> float:
    """Compute score from simulation metrics.
    
    For gait policies, emphasize speed while maintaining stability.
    For other policies, emphasize compliance and tracking.
    """
    score = 0.0
    
    if metrics.fell:
        score -= 200.0
        return float(score)  # Early exit for falls
    
    if policy_mode == "gait":
        # Gait policy: prioritize speed with stability constraints
        score += 50.0 * metrics.avg_speed_mps  # Primary objective: speed
        score += 100.0 * metrics.zmp_compliance  # Maintain safety
        score -= 100.0 * metrics.max_zmp_violation  # Heavy penalty for violations
        score -= 20.0 * metrics.avg_dcm_error  # Stability
        score -= 10.0 * metrics.forward_vel_mae_mps  # Smooth velocity
        score -= 0.5 * metrics.cot  # Energy efficiency (minor)
    else:
        # Default policy: prioritize compliance and tracking
        score += 120.0 * metrics.zmp_compliance
        score -= 50.0 * metrics.max_zmp_violation
        score -= 20.0 * metrics.forward_vel_mae_mps
        score += 5.0 * metrics.distance_m
        score -= 5.0 * metrics.avg_dcm_error
    
    return float(score)


def _evaluate_policy(
    weights: np.ndarray,
    bias: np.ndarray,
    base_config: SimulationConfig,
    residual_clip: float,
    eval_steps: Optional[int],
    policy_mode: str,
    temp_dir: Path,
) -> float:
    from v2.main_mpc import RLConfig, run_simulation

    policy_path = temp_dir / f"policy_{uuid.uuid4().hex}.npz"
    LinearPolicy(weights=weights, bias=bias).save(policy_path)
    rl_config = RLConfig(
        dataset_path=None,
        policy_path=policy_path,
        policy_mode=policy_mode,
        residual_clip=residual_clip,
    )
    metrics = run_simulation(
        base_config,
        gui=False,
        max_steps=eval_steps,
        print_every=200,
        rl_config=rl_config,
        return_metrics=True,
    )
    policy_path.unlink(missing_ok=True)
    if metrics is None:
        return -1e6
    return _score_metrics(metrics, policy_mode=policy_mode)


def train_residual_cem(
    config: "SimulationConfig",
    output_path: Path,
    cem: Optional[CEMConfig] = None,
) -> LinearPolicy:
    if cem is None:
        cem = CEMConfig()

    rng = np.random.default_rng(cem.seed)
    spec = ObservationSpec.default()
    obs_dim = spec.dim
    action_dim = cem.action_dim

    mean = np.zeros(action_dim * obs_dim + action_dim)
    std = np.full_like(mean, cem.init_std)

    temp_dir = output_path.parent / "_cem_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    best_vec = mean.copy()
    best_score = -1e9

    for _ in range(cem.iterations):
        samples = rng.normal(loc=mean, scale=std, size=(cem.population, mean.shape[0]))
        scores = []
        for vec in samples:
            weights, bias = _unflatten_policy(vec, action_dim, obs_dim)
            score = _evaluate_policy(
                weights,
                bias,
                config,
                cem.residual_clip,
                cem.eval_steps,
                cem.policy_mode,
                temp_dir,
            )
            scores.append(score)
        scores = np.asarray(scores)
        elite_count = max(1, int(cem.population * cem.elite_frac))
        elite_idx = np.argsort(scores)[-elite_count:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), cem.min_std)
        if scores[elite_idx[-1]] > best_score:
            best_score = float(scores[elite_idx[-1]])
            best_vec = elite[-1]

    weights, bias = _unflatten_policy(best_vec, action_dim, obs_dim)
    policy = LinearPolicy(weights=weights, bias=bias)
    policy.save(output_path)
    return policy
