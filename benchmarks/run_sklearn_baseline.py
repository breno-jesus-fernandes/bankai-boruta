"""Run a reproducible scikit-learn forest baseline and emit one JSON record."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import sklearn
from sklearn.ensemble import RandomForestClassifier

from benchmarks.datasets import make_boruta_case


def run_baseline(
    *,
    scenario: str,
    seed: int,
    n_samples: int,
    n_features: int,
    n_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    case = make_boruta_case(
        scenario,
        seed=seed,
        n_samples=n_samples,
        n_features=n_features,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(case.X, case.y)
    elapsed = time.perf_counter() - started

    return {
        "method": "sklearn_random_forest_fit",
        "scenario": scenario,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "relevant_features": list(case.relevant_features),
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "fit_seconds": elapsed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("mixed", "null", "correlated", "xor"), default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.n_estimators < 1 or args.n_jobs < 1:
        parser.error("n-estimators and n-jobs must be positive")
    record = run_baseline(
        scenario=args.scenario,
        seed=args.seed,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    encoded = json.dumps(record, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
