import pytest
from sklearn.base import clone

from bankai_boruta import BankaiBoruta, _core


def test_native_extension_is_loaded_before_the_engine_milestone():
    assert _core.ENGINE_READY is False


def test_default_parameters_are_cloneable_and_match_the_public_contract():
    selector = BankaiBoruta()

    assert selector.get_params() == {
        "n_estimators": 500,
        "max_iter": 100,
        "alpha": 0.01,
        "pruning": "canonical",
        "importance": "permutation",
        "importance_scale": "zscore",
        "input_mode": "zero_copy",
        "max_features": "sqrt",
        "max_depth": 512,
        "n_jobs": 1,
        "random_state": None,
        "store_history": False,
    }
    assert clone(selector).get_params() == selector.get_params()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"max_iter": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"pruning": "unknown"},
        {"importance": "unknown"},
        {"importance_scale": "unknown"},
        {"input_mode": "unknown"},
        {"n_jobs": 0},
        {"max_depth": 0},
    ],
)
def test_invalid_parameters_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        BankaiBoruta(**kwargs)


def test_fit_fails_clearly_until_the_native_engine_is_available():
    selector = BankaiBoruta()

    with pytest.raises(RuntimeError, match="native Boruta engine is not available"):
        selector.fit([[0.0], [1.0]], [0, 1])
