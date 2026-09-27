import numpy as np
import pytest

from benchmarks.datasets import make_boruta_case


def test_benchmark_fixtures_are_reproducible_and_have_expected_signal_metadata():
    first = make_boruta_case("mixed", seed=17, n_samples=128, n_features=8)
    second = make_boruta_case("mixed", seed=17, n_samples=128, n_features=8)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.X.shape == (128, 8)
    assert first.relevant_features == (0, 1)


def test_fixture_factory_rejects_unknown_scenarios():
    with pytest.raises(ValueError, match="scenario"):
        make_boruta_case("typo", seed=0, n_samples=20, n_features=3)


@pytest.mark.parametrize("scenario", ["mixed", "null", "correlated", "xor"])
def test_every_scenario_has_finite_data_and_binary_targets(scenario):
    case = make_boruta_case(scenario, seed=5, n_samples=128, n_features=6)

    assert np.isfinite(case.X).all()
    assert set(np.unique(case.y)) <= {0, 1}
    assert set(np.unique(case.y)) == {0, 1}
    assert all(0 <= index < case.X.shape[1] for index in case.relevant_features)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 7, "n_features": 2},
        {"n_samples": 20, "n_features": 0},
    ],
)
def test_fixture_factory_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError, match="n_samples|n_features"):
        make_boruta_case("mixed", seed=0, **kwargs)
