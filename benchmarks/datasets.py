"""Small deterministic synthetic cases shared by tests and benchmarks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BorutaCase:
    X: np.ndarray
    y: np.ndarray
    relevant_features: tuple[int, ...]


def make_boruta_case(
    scenario: str,
    *,
    seed: int,
    n_samples: int,
    n_features: int,
) -> BorutaCase:
    """Create a seeded classification case with known relevant columns."""
    if scenario not in {"mixed", "null", "correlated", "xor"}:
        raise ValueError(f"unknown scenario: {scenario}")
    if n_samples < 8 or n_features < 1:
        raise ValueError("n_samples must be at least 8 and n_features must be positive")
    if scenario != "null" and n_features < 2:
        raise ValueError(f"scenario {scenario!r} requires at least two features")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    if scenario == "null":
        y = rng.integers(0, 2, size=n_samples)
        relevant: tuple[int, ...] = ()
    elif scenario == "xor":
        X[:, :2] = rng.integers(0, 2, size=(n_samples, 2))
        y = (X[:, 0] != X[:, 1]).astype(np.int64)
        relevant = (0, 1)
    else:
        if scenario == "correlated":
            X[:, 1] = X[:, 0] + rng.normal(0, 0.03, size=n_samples)
        y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, size=n_samples) > 0).astype(np.int64)
        relevant = (0, 1)
    return BorutaCase(X=X, y=y, relevant_features=relevant)
