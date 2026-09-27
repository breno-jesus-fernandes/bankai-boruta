import pytest
from sklearn.base import clone

from bankai_boruta import BankaiBoruta, _core


def test_native_extension_exposes_the_engine():
    assert _core.ENGINE_READY is True


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
        {"random_state": -1},
        {"max_features": float("nan")},
    ],
)
def test_invalid_parameters_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        BankaiBoruta(**kwargs)


def test_fit_selects_strong_signal_and_transform_preserves_original_columns():
    import numpy as np

    rng = np.random.default_rng(41)
    y = np.tile([0, 1], 80)
    signal = y.astype(np.float64) + rng.normal(0, 0.03, y.size)
    noise = rng.normal(size=y.size)
    X = np.column_stack([signal, noise])
    selector = BankaiBoruta(
        n_estimators=80,
        max_iter=12,
        alpha=0.05,
        random_state=91,
        store_history=True,
    )

    assert selector.fit(X, y) is selector
    assert selector.support_.shape == (2,)
    assert selector.support_[0]
    assert selector.transform(X).shape[1] == selector.get_support().sum()
    assert selector.n_features_in_ == 2
    assert 1 <= selector.n_iter_ <= selector.max_iter
    assert selector.stop_reason_ in {"no_tentative_features", "max_iter"}
    assert len(selector.history_) == selector.n_iter_


def test_single_feature_uses_five_virtual_shadows_and_is_selectable():
    import numpy as np

    y = np.tile([0, 1], 48)
    X = y.astype(np.float64).reshape(-1, 1)
    selector = BankaiBoruta(n_estimators=48, max_iter=12, random_state=6).fit(X, y)

    assert selector.support_.tolist() == [True]
    assert selector.n_iter_ < selector.max_iter


def test_transform_before_fit_raises_sklearn_not_fitted_error():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        BankaiBoruta().transform([[0.0, 1.0]])


def test_refit_clears_optional_history_when_disabled():
    import numpy as np

    X = np.column_stack([np.arange(48), np.arange(48) % 7]).astype(float)
    y = np.arange(48) % 2
    selector = BankaiBoruta(
        n_estimators=12, max_iter=2, store_history=True, random_state=1
    ).fit(X, y)
    assert hasattr(selector, "history_")
    selector.set_params(store_history=False).fit(X, y)
    assert not hasattr(selector, "history_")


def test_fit_is_deterministic_and_leaves_tentatives_unresolved_at_limit():
    import numpy as np

    rng = np.random.default_rng(12)
    X = rng.normal(size=(72, 3))
    y = rng.integers(0, 2, size=72)
    options = dict(
        n_estimators=12,
        max_iter=3,
        alpha=1e-300,
        importance_scale="raw",
        random_state=7,
    )
    first = BankaiBoruta(**options).fit(X, y)
    second = BankaiBoruta(**options).fit(X, y)

    assert first.n_iter_ == 3
    assert first.stop_reason_ == "max_iter"
    assert first.support_.tolist() == second.support_.tolist()
    assert first.support_weak_.tolist() == second.support_weak_.tolist()


def test_fit_zero_copy_rejects_noncontiguous_and_copy_mode_accepts_it():
    import numpy as np

    X = np.arange(160, dtype=np.float64).reshape(40, 4)[:, ::2]
    y = np.tile([0, 1], 20)
    with pytest.raises(ValueError, match="zero_copy"):
        BankaiBoruta(n_estimators=8, max_iter=1, random_state=2).fit(X, y)

    fitted = BankaiBoruta(
        n_estimators=8,
        max_iter=1,
        input_mode="copy",
        random_state=2,
    ).fit(X, y)
    assert fitted.n_features_in_ == 2
    assert fitted.diagnostics_["copied_bytes"] == 40 * 2 * 8

    from_list = BankaiBoruta(
        n_estimators=8,
        max_iter=1,
        input_mode="copy",
        random_state=2,
    ).fit(X.tolist(), y)
    assert from_list.diagnostics_["copied_bytes"] >= 40 * 2 * 8


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fit_borrows_fortran_order_and_records_no_matrix_copy(dtype):
    import numpy as np

    rng = np.random.default_rng(9)
    y = np.tile([0, 1], 40)
    X = np.asfortranarray(
        np.column_stack([y, rng.normal(size=y.size)]).astype(dtype)
    )
    fitted = BankaiBoruta(
        n_estimators=24,
        max_iter=8,
        input_mode="zero_copy",
        random_state=3,
    ).fit(X, y)
    assert fitted.n_features_in_ == 2
    assert fitted.diagnostics_["copied_bytes"] == 0


def test_fit_supports_string_labels_and_multiclass_targets():
    import numpy as np

    rng = np.random.default_rng(19)
    y = np.tile(np.array(["a", "b", "c"]), 40)
    X = np.column_stack([y == "b", y == "c", rng.normal(size=y.size)]).astype(float)
    fitted = BankaiBoruta(n_estimators=32, max_iter=12, random_state=8).fit(X, y)
    assert fitted.n_features_in_ == 3
    assert fitted.support_.shape == (3,)
    assert fitted.support_[:2].tolist() == [True, True]
    assert fitted.diagnostics_["copied_bytes"] == 0


def test_fit_rejects_invalid_class_count_and_non_numpy_zero_copy_inputs():
    import numpy as np

    X = np.arange(32, dtype=float).reshape(16, 2)
    with pytest.raises(ValueError, match="at least two classes"):
        BankaiBoruta(n_estimators=8, max_iter=1).fit(X, np.zeros(16))
    with pytest.raises(ValueError, match="Unknown label type"):
        BankaiBoruta(n_estimators=8, max_iter=1).fit(X, np.arange(16) / 10)
    with pytest.raises(ValueError, match="Input contains NaN"):
        BankaiBoruta(n_estimators=8, max_iter=1).fit(
            X, np.where(np.arange(16) % 2, 1.0, np.nan)
        )
    with pytest.raises(ValueError, match="NumPy ndarray"):
        BankaiBoruta(n_estimators=8, max_iter=1).fit(X.tolist(), np.tile([0, 1], 8))


def test_aggressive_pruning_fits_and_retains_confirmed_features():
    import numpy as np

    rng = np.random.default_rng(51)
    y = np.tile([0, 1], 80)
    X = np.column_stack(
        [y + rng.normal(0, 0.02, y.size), rng.normal(size=(y.size, 3))]
    )
    fitted = BankaiBoruta(
        n_estimators=48,
        max_iter=12,
        pruning="aggressive",
        random_state=23,
        store_history=True,
    ).fit(X, y)

    assert fitted.support_[0]
    assert fitted.n_iter_ <= fitted.max_iter
    assert min(fitted.diagnostics_["active_features_per_iteration"]) < X.shape[1]


def test_parallel_fit_is_repeatable_and_matches_serial_decisions():
    import numpy as np

    rng = np.random.default_rng(61)
    y = np.tile([0, 1], 80)
    X = np.column_stack([y + rng.normal(0, 0.05, y.size), rng.normal(size=(y.size, 3))])
    options = dict(n_estimators=40, max_iter=12, random_state=29, store_history=True)
    serial = BankaiBoruta(**options).fit(X, y)
    parallel = BankaiBoruta(**options, n_jobs=4).fit(X, y)
    repeated = BankaiBoruta(**options, n_jobs=4).fit(X, y)

    assert parallel.decision_.tolist() == serial.decision_.tolist()
    assert parallel.decision_.tolist() == repeated.decision_.tolist()
    assert parallel.history_ == repeated.history_


def test_fit_crosses_into_native_code_once_for_the_entire_iteration_loop(monkeypatch):
    import numpy as np

    from bankai_boruta import _core

    original = _core.fit_boruta
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(_core, "fit_boruta", counted)
    X = np.column_stack([np.arange(48), np.arange(48) % 7]).astype(float)
    y = np.arange(48) % 2
    BankaiBoruta(n_estimators=12, max_iter=3, random_state=1).fit(X, y)

    assert calls == 1
