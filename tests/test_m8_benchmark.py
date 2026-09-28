"""M8 benchmark protocol tests, independent of optional estimators."""

import pytest

from benchmarks.run_m8_estimators import (
    BENCHMARK_FEATURES,
    BENCHMARK_ROWS,
    METHODS,
    _markdown,
    _summarize,
    sklearn_forest_parameters,
)


def test_m8_comparison_is_pinned_to_ten_thousand_by_one_hundred():
    assert (BENCHMARK_ROWS, BENCHMARK_FEATURES) == (10_000, 100)
    assert METHODS == (
        "bankai_native",
        "bankai_external_xrf",
        "bankai_external_sklearn",
        "borutapy_sklearn",
    )


def test_external_sklearn_and_borutapy_share_the_same_estimator_configuration():
    assert sklearn_forest_parameters(trees=8, seed=41, features=100) == {
        "n_estimators": 8,
        "max_depth": 16,
        "max_features": "sqrt",
        "n_jobs": 1,
        "class_weight": "balanced",
        "random_state": 41,
    }


def test_summary_reports_bytes_calls_time_and_excludes_warmups():
    common = {
        "method": "bankai_external_sklearn",
        "rows": 10_000,
        "features": 100,
        "fit_seconds": 2.0,
        "rss_delta_bytes": 1024,
        "copied_bytes": 8_000_000,
        "external_estimator_calls": 2,
        "external_estimator_seconds": 1.0,
        "selected_count": 0,
        "n_iter": 2,
    }
    rows = _summarize([
        {**common, "phase": "warmup"},
        {**common, "phase": "performance"},
    ])
    assert len(rows) == 1
    assert rows[0]["runs"] == 1
    assert rows[0]["median_copied_bytes"] == 8_000_000
    assert rows[0]["median_external_estimator_calls"] == 2
    assert rows[0]["median_external_estimator_seconds"] == 1.0


def test_m8_markdown_names_the_matched_pair_and_disclaims_quality():
    report = _markdown({
        "metadata": {
            "date": "2026-09-27", "platform": "test", "python": "3.13",
            "scikit_learn": "test", "numpy": "test", "trees": 8,
            "iterations": 2, "repeats": 5,
        },
        "summary": [],
    })
    assert "somente 10.000×100" in report
    assert "bankai_external_sklearn` versus `borutapy_sklearn" in report
    assert "não permitem concluir qualidade" in report


@pytest.mark.parametrize("rows", [10_001, 100_000])
def test_benchmark_protocol_rejects_more_than_ten_thousand_rows(rows):
    from benchmarks.run_m8_estimators import validate_dimensions

    with pytest.raises(ValueError, match="10.000"):
        validate_dimensions(rows, BENCHMARK_FEATURES)
