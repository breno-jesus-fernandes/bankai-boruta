"""Small, dependency-free checks for the M7 benchmark protocol."""

import subprocess
import sys
from pathlib import Path

from benchmarks.run_m7_comparisons import _summarize
from benchmarks.run_m7_comparisons import _case
from benchmarks.run_m7_comparisons import QUALITY_SEEDS
from benchmarks.run_m7_comparisons import METHODS
from benchmarks.run_m7_comparisons import SELECTORS
from benchmarks.run_m7_comparisons import BENCHMARK_ROWS, BENCHMARK_FEATURES, PERFORMANCE_DIMENSIONS
from benchmarks.datasets import make_boruta_case


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_m7_comparisons.py"


def test_m7_summary_excludes_warmup_and_separates_phases():
    common = {
        "method": "bankai_boruta",
        "scenario": "mixed",
        "rows": 1_000,
        "features": 20,
        "fit_seconds": 1.0,
        "end_to_end_seconds": 1.2,
        "rss_delta_bytes": 1024,
        "precision": 1.0,
        "recall": 0.5,
        "false_positives": 0,
        "selected_count": 1,
        "n_iter": 3,
    }
    records = [
        {**common, "phase": "warmup", "fit_seconds": 0.01},
        {**common, "phase": "performance"},
        {**common, "phase": "quality", "fit_seconds": 2.0},
    ]

    summary = _summarize(records)

    assert [item["phase"] for item in summary] == ["performance", "quality"]
    assert [item["runs"] for item in summary] == [1, 1]
    assert summary[0]["median_fit_seconds"] == 1.0
    assert summary[0]["mean_precision"] is None
    assert summary[1]["mean_precision"] == 1.0


def test_m7_worker_reuses_the_shared_deterministic_benchmark_fixtures():
    actual = _case("correlated", 47, 256, 20)
    expected = make_boruta_case("correlated", seed=47, n_samples=256, n_features=20)

    assert (actual[0] == expected.X).all()
    assert (actual[1] == expected.y).all()
    assert actual[2] == expected.relevant_features


def test_m7_quality_seed_plan_is_stable_and_has_thirty_unique_seeds():
    assert len(QUALITY_SEEDS) == 30
    assert len(set(QUALITY_SEEDS)) == 30
    assert QUALITY_SEEDS[:3] == (11, 29, 47)


def test_every_benchmark_phase_uses_only_the_ten_thousand_row_dataset():
    assert (BENCHMARK_ROWS, BENCHMARK_FEATURES) == (10_000, 100)
    assert PERFORMANCE_DIMENSIONS == ((10_000, 100),)


def test_m7_markdown_states_quality_and_performance_share_ten_thousand_rows():
    from benchmarks.run_m7_comparisons import _markdown

    markdown = _markdown({
        "metadata": {
            "date": "2026-09-27", "platform": "test", "python": "3.12.3",
            "scikit_learn": "1.9.1", "numpy": "2.5.3", "max_rows": 10_000,
            "repeats": 5, "trees": 8, "iterations": 2, "quality_trees": 16,
            "quality_iterations": 12, "quality_seeds": 30,
            "borutapy_version": "0.4.3", "boruta_fru_version": "0.1.0",
            "borutashap_version": "1.0.17", "pandas_version": "3.0.6",
            "polars_version": "1.44.2", "scipy_version": "1.18.1",
            "shap_version": "0.52.0", "bankai_boruta_revision": "abc",
            "xrf_revision": "def",
        },
        "summary": [],
    })

    assert "inclusive qualidade" in markdown
    assert "não há matriz multi-dataset" in markdown
    assert "1k×20" not in markdown
    assert "não alocação negativa" in markdown


def test_m7_has_a_paired_sklearn_and_bankai_forest_borutapy_comparison():
    assert "borutapy" in METHODS
    assert "borutapy_bankai_xrf" in METHODS
    assert "borutapy_bankai_xrf" in SELECTORS


def test_m7_worker_rejects_inputs_above_ten_thousand_rows():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--worker", "--method", "sklearn_rf", "--rows", "10001"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "capped at 10,000 rows" in completed.stderr


def test_m7_markdown_has_result_links_and_scope_disclaimers():
    from benchmarks.run_m7_comparisons import _markdown

    result = {
        "metadata": {
            "date": "2026-09-27",
            "platform": "test-macOS-arm64",
            "python": "3.12.3",
            "scikit_learn": "1.9.1",
            "numpy": "2.5.3",
            "max_rows": 10_000,
            "repeats": 5,
            "trees": 16,
            "iterations": 8,
            "quality_trees": 16,
            "quality_iterations": 12,
            "quality_seeds": 3,
            "borutapy_version": "0.4.3",
            "boruta_fru_version": "0.1.0",
            "borutashap_version": "1.0.17",
            "pandas_version": "3.0.6",
            "polars_version": "1.44.2",
            "scipy_version": "1.18.1",
            "shap_version": "0.52.0",
            "bankai_boruta_revision": "abc",
            "xrf_revision": "def",
        },
        "summary": [],
    }

    markdown = _markdown(result)

    assert "10 mil linhas" in markdown
    assert "m7_comparisons.json" in markdown
    assert "não é Boruta estatístico" in markdown
