"""Paired quality and runtime comparison for M5 pruning and worker modes."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from bankai_boruta import BankaiBoruta
from benchmarks.datasets import make_boruta_case


def run_comparison(
    *,
    scenarios: tuple[str, ...] = ("mixed", "null", "correlated", "xor"),
    seeds: tuple[int, ...] = (11, 29, 47),
    n_samples: int = 500,
    n_features: int = 16,
    n_estimators: int = 64,
    max_iter: int = 24,
) -> dict:
    records = []
    for scenario in scenarios:
        for seed in seeds:
            case = make_boruta_case(
                scenario,
                seed=seed,
                n_samples=n_samples,
                n_features=n_features,
            )
            for pruning in ("canonical", "aggressive"):
                for n_jobs in (1, 4):
                    selector = BankaiBoruta(
                        n_estimators=n_estimators,
                        max_iter=max_iter,
                        alpha=0.01,
                        pruning=pruning,
                        n_jobs=n_jobs,
                        random_state=seed,
                        store_history=True,
                    )
                    started = time.perf_counter()
                    selector.fit(case.X, case.y)
                    elapsed = time.perf_counter() - started
                    selected = set(np.flatnonzero(selector.support_).tolist())
                    relevant = set(case.relevant_features)
                    true_positive = len(selected & relevant)
                    precision = true_positive / len(selected) if selected else float(not relevant)
                    recall = true_positive / len(relevant) if relevant else float(not selected)
                    records.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "pruning": pruning,
                            "n_jobs": n_jobs,
                            "fit_seconds": elapsed,
                            "selected": sorted(selected),
                            "tentative": np.flatnonzero(selector.support_weak_).tolist(),
                            "precision": precision,
                            "recall": recall,
                            "n_iter": selector.n_iter_,
                            "stop_reason": selector.stop_reason_,
                            "active_features_per_iteration": selector.diagnostics_[
                                "active_features_per_iteration"
                            ],
                        }
                    )

    groups = {}
    for row in records:
        key = (row["scenario"], row["seed"], row["pruning"])
        groups.setdefault(key, {})[row["n_jobs"]] = row
    parallel_checks = [
        {
            "scenario": scenario,
            "seed": seed,
            "pruning": pruning,
            "decisions_match": serial["selected"] == parallel["selected"]
            and serial["tentative"] == parallel["tentative"],
            "serial_seconds": serial["fit_seconds"],
            "parallel_seconds": parallel["fit_seconds"],
        }
        for (scenario, seed, pruning), modes in groups.items()
        if (serial := modes[1]) is not None and (parallel := modes[4]) is not None
    ]
    return {
        "metadata": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "n_samples": n_samples,
            "n_features": n_features,
            "n_estimators": n_estimators,
            "max_iter": max_iter,
            "scenarios": scenarios,
            "seeds": seeds,
        },
        "parallel_checks": parallel_checks,
        "runs": records,
    }


def main() -> None:
    output = Path("benchmarks/results/m5_pruning_parallel_2026-09-27.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_comparison()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
