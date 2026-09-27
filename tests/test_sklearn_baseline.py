from benchmarks.run_sklearn_baseline import run_baseline


def test_sklearn_baseline_emits_machine_readable_reproducibility_metadata():
    result = run_baseline(
        scenario="mixed",
        seed=23,
        n_samples=64,
        n_features=5,
        n_estimators=3,
        n_jobs=1,
    )

    assert result["method"] == "sklearn_random_forest_fit"
    assert result["seed"] == 23
    assert result["n_samples"] == 64
    assert result["n_features"] == 5
    assert result["relevant_features"] == [0, 1]
    assert result["n_estimators"] == 3
    assert result["fit_seconds"] >= 0
    assert result["scikit_learn"]
    assert result["python"]
    assert result["platform"]
