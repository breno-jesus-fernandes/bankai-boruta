"""Measure process RSS across valid and invalid Arrow fits after allocator warmup."""
from __future__ import annotations

import gc
import json

import numpy as np
import polars as pl
import psutil

from bankai_boruta import BankaiBoruta


def measure(*, warmups: int = 10, repetitions: int = 60) -> dict:
    rng = np.random.default_rng(19)
    process = psutil.Process()
    model = BankaiBoruta(n_estimators=4, max_iter=1, max_depth=5, random_state=4)
    labels = np.tile([0, 1], 2000)

    def cycle() -> None:
        X = pl.DataFrame(rng.normal(size=(4000, 16)), orient="row")
        model.fit(X, labels)
        invalid = X.with_columns(pl.lit(None, dtype=pl.Float64).alias(X.columns[-1]))
        try:
            model.fit(invalid, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("null input must fail")

    for _ in range(warmups):
        cycle()
    gc.collect()
    baseline = process.memory_info().rss
    samples = []
    for _ in range(repetitions):
        cycle()
        gc.collect()
        samples.append(process.memory_info().rss)
    peak = max(samples)
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "samples_rss_bytes": samples,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak,
        "growth_bytes": peak - baseline,
        "allowed_growth_bytes": 16 * 1024 * 1024,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), sort_keys=True))
