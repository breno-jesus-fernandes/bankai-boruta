import pytest
from sklearn.base import clone

from bankai_boruta import BankaiBoruta, _core


def test_native_extension_exposes_the_engine():
    assert _core.ENGINE_READY is True


def test_default_parameters_are_cloneable_and_match_the_public_contract():
    selector = BankaiBoruta()

    assert selector.get_params() == {
        "n_estimators": 40,
        "max_iter": 100,
        "alpha": 0.05,
        "pruning": "canonical",
        "importance": None,
        "importance_scale": "zscore",
        "input_mode": "zero_copy",
        "max_features": "sqrt",
        "max_depth": 6,
        "n_jobs": -1,
        "random_state": 2077,
        "max_bins": 63,
        "store_history": False,
        "estimator": None,
        "importance_getter": "auto",
        "forest_backend": "native",
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
        {"forest_backend": "unknown"},
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


def test_default_fit_keeps_native_diagnostics_and_does_not_retain_training_arrays():
    import numpy as np

    rng = np.random.default_rng(902)
    X = rng.normal(size=(48, 3))
    y = np.tile([0, 1], 24)
    selector = BankaiBoruta(n_estimators=12, max_iter=2, random_state=17).fit(X, y)

    assert selector.diagnostics_["copied_bytes"] == 0
    assert "undefined_importance_count" in selector.diagnostics_
    assert "non_finite_importance_count" in selector.diagnostics_
    assert not any(
        key in vars(selector)
        for key in ("X", "y", "forest_", "training_data_", "estimator_")
    )


def test_optional_sklearn_estimator_is_cloned_and_fitted_once_per_native_iteration():
    import numpy as np
    from sklearn.base import BaseEstimator

    class SpyImportanceEstimator(BaseEstimator):
        seen = []

        def __init__(self, marker=3):
            self.marker = marker

        def fit(self, X, y):
            type(self).seen.append((self, X.shape, y.shape, tuple(X[0, :3])))
            self.feature_importances_ = np.ones(X.shape[1], dtype=float)
            return self

    SpyImportanceEstimator.seen = []
    template = SpyImportanceEstimator()
    X = np.arange(72, dtype=float).reshape(24, 3)
    y = np.tile([0, 1], 12)
    selector = BankaiBoruta(
        estimator=template, n_estimators=8, max_iter=2, random_state=12
    ).fit(X, y)

    assert len(SpyImportanceEstimator.seen) == selector.n_iter_ == 2
    assert all(shape == (24, 8) for _, shape, _, _ in SpyImportanceEstimator.seen)
    assert all(target_shape == (24,) for _, _, target_shape, _ in SpyImportanceEstimator.seen)
    assert all(first_row == (0.0, 1.0, 2.0) for _, _, _, first_row in SpyImportanceEstimator.seen)
    assert len({id(fitted) for fitted, _, _, _ in SpyImportanceEstimator.seen}) == 2
    assert not hasattr(template, "feature_importances_")
    assert selector.diagnostics_["external_estimator_calls"] == 2
    assert selector.diagnostics_["external_estimator_seconds"] >= 0.0
    assert selector.diagnostics_["fit_time_seconds"] >= selector.diagnostics_["external_estimator_seconds"]
    assert selector.diagnostics_["copied_bytes"] == 2 * 24 * 8 * 8


@pytest.mark.parametrize("importance", ["gain", "split", "shap"])
def test_bankai_backend_accepts_random_forest_importance_modes(importance):
    selector = BankaiBoruta(forest_backend="bankai", importance=importance)

    assert selector.importance == importance


@pytest.mark.parametrize("importance", ["gain", "split", "shap"])
def test_native_backend_rejects_importance_modes_it_does_not_implement(importance):
    with pytest.raises(ValueError, match="importance must be one of: permutation"):
        BankaiBoruta(importance=importance)


def test_bankai_backend_builds_the_release_estimator_and_rank_preserving_getter(monkeypatch):
    import sys
    import types
    import numpy as np
    from sklearn.base import BaseEstimator

    class FakeBankaiRandomForestClassifier(BaseEstimator):
        def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            n_jobs=None,
            random_state=None,
            importance_type="gain",
            max_bins=None,
        ):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.max_features = max_features
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.importance_type = importance_type
            self.max_bins = max_bins

    monkeypatch.setitem(
        sys.modules,
        "bankai_random_forest",
        types.SimpleNamespace(BankaiRandomForestClassifier=FakeBankaiRandomForestClassifier),
    )
    selector = BankaiBoruta(
        forest_backend="bankai",
        n_estimators=17,
        max_depth=9,
        max_features=0.75,
        n_jobs=3,
        random_state=41,
        importance="permutation",
    )

    estimator, getter = selector._resolve_estimator()

    assert isinstance(estimator, FakeBankaiRandomForestClassifier)
    assert estimator.get_params() == {
        "importance_type": "permutation",
        "max_bins": 63,
        "max_depth": 9,
        "max_features": 0.75,
        "n_estimators": 17,
        "n_jobs": 3,
        "random_state": 41,
    }
    assert callable(getter)
    np.testing.assert_array_equal(
        getter(types.SimpleNamespace(feature_importances_=np.array([-2.0, 0.5, 1.0]))),
        np.array([0.0, 2.5, 3.0]),
    )


def test_bankai_backend_fits_cloned_estimator_with_nonnegative_permutation_scores(monkeypatch):
    import sys
    import types
    import numpy as np
    from sklearn.base import BaseEstimator

    class FakeBankaiRandomForestClassifier(BaseEstimator):
        seen = []

        def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            n_jobs=None,
            random_state=None,
            importance_type="gain",
            max_bins=None,
        ):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.max_features = max_features
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.importance_type = importance_type
            self.max_bins = max_bins

        def fit(self, X, y):
            type(self).seen.append((X.shape, y.shape, self.importance_type))
            self.feature_importances_ = np.linspace(-1.0, 1.0, X.shape[1])
            return self

    FakeBankaiRandomForestClassifier.seen = []
    monkeypatch.setitem(
        sys.modules,
        "bankai_random_forest",
        types.SimpleNamespace(BankaiRandomForestClassifier=FakeBankaiRandomForestClassifier),
    )
    X = np.arange(72, dtype=float).reshape(24, 3)
    y = np.tile([0, 1], 12)

    selector = BankaiBoruta(
        forest_backend="bankai", importance="permutation", n_estimators=8, max_iter=1, random_state=12
    ).fit(X, y)

    assert selector.n_iter_ == 1
    assert selector.diagnostics_["external_estimator_calls"] == 1
    assert FakeBankaiRandomForestClassifier.seen == [((24, 8), (24,), "permutation")]


def test_bankai_backend_maps_shap_mode_and_uses_auto_getter(monkeypatch):
    import sys
    import types
    from sklearn.base import BaseEstimator

    class FakeBankaiRandomForestClassifier(BaseEstimator):
        def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            n_jobs=None,
            random_state=None,
            importance_type="gain",
            max_bins=None,
        ):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.max_features = max_features
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.importance_type = importance_type
            self.max_bins = max_bins

    monkeypatch.setitem(
        sys.modules,
        "bankai_random_forest",
        types.SimpleNamespace(BankaiRandomForestClassifier=FakeBankaiRandomForestClassifier),
    )
    selector = BankaiBoruta(forest_backend="bankai", importance="shap")

    estimator, getter = selector._resolve_estimator()

    assert estimator.importance_type == "shap"
    assert getter == "auto"


def test_external_estimator_fit_errors_are_propagated_without_losing_the_exception():
    import numpy as np
    from sklearn.base import BaseEstimator

    class BrokenEstimator(BaseEstimator):
        def fit(self, X, y):
            raise RuntimeError("sentinel estimator failure")

    with pytest.raises(RuntimeError, match="sentinel estimator failure"):
        BankaiBoruta(
            estimator=BrokenEstimator(), n_estimators=8, max_iter=1, random_state=2
        ).fit(np.ones((24, 2)), np.tile([0, 1], 12))


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (lambda width: [1.0] * (width - 1), "expected"),
        (lambda width: [1.0] * (width + 1), "expected"),
        (lambda width: [float("nan")] * width, "finite"),
        (lambda width: [-1.0] * width, "non-negative"),
        (lambda width: [[1.0] * width], "shape"),
    ],
)
def test_external_estimator_rejects_malformed_importance_vectors(scores, message):
    import numpy as np
    from sklearn.base import BaseEstimator

    class MalformedEstimator(BaseEstimator):
        def __init__(self, scorer):
            self.scorer = scorer

        def fit(self, X, y):
            self.feature_importances_ = self.scorer(X.shape[1])
            return self

    with pytest.raises(ValueError, match=message):
        BankaiBoruta(
            estimator=MalformedEstimator(scores),
            n_estimators=8,
            max_iter=1,
            random_state=3,
        ).fit(np.ones((24, 2)), np.tile([0, 1], 12))


def test_custom_importance_getter_exception_is_preserved():
    import numpy as np
    from sklearn.base import BaseEstimator

    class PlainEstimator(BaseEstimator):
        def fit(self, X, y):
            return self

    def broken_getter(_):
        raise LookupError("sentinel importance getter failure")

    with pytest.raises(LookupError, match="sentinel importance getter failure"):
        BankaiBoruta(
            estimator=PlainEstimator(),
            importance_getter=broken_getter,
            n_estimators=8,
            max_iter=1,
            random_state=2,
        ).fit(np.ones((24, 2)), np.tile([0, 1], 12))


def test_external_fit_failure_clears_fitted_state_and_allows_refit():
    import numpy as np
    from sklearn.base import BaseEstimator

    class ToggleEstimator(BaseEstimator):
        fail = True

        def fit(self, X, y):
            if type(self).fail:
                raise RuntimeError("first fit fails")
            self.feature_importances_ = np.ones(X.shape[1])
            return self

    selector = BankaiBoruta(
        estimator=ToggleEstimator(), n_estimators=8, max_iter=1, random_state=2
    )
    X = np.ones((24, 2))
    y = np.tile([0, 1], 12)
    with pytest.raises(RuntimeError, match="first fit fails"):
        selector.fit(X, y)
    assert not hasattr(selector, "support_")

    ToggleEstimator.fail = False
    assert selector.fit(X, y) is selector
    assert hasattr(selector, "support_")


@pytest.mark.parametrize("layout", ["float32_fortran", "float64_c", "strided_copy"])
def test_external_estimator_accepts_supported_numpy_layouts_and_counts_matrix_bytes(layout):
    import numpy as np
    from sklearn.base import BaseEstimator

    class StableEstimator(BaseEstimator):
        def fit(self, X, y):
            self.feature_importances_ = np.ones(X.shape[1])
            return self

    rows = 32
    base = np.arange(rows * 6, dtype=np.float64).reshape(rows, 6)
    y = np.tile([0, 1], rows // 2)
    if layout == "float32_fortran":
        X = np.asfortranarray(base[:, :3].astype(np.float32))
        input_mode = "zero_copy"
    elif layout == "float64_c":
        X = np.ascontiguousarray(base[:, :3])
        input_mode = "zero_copy"
    else:
        X = base[:, ::2]
        input_mode = "copy"

    selector = BankaiBoruta(
        estimator=StableEstimator(),
        n_estimators=8,
        max_iter=1,
        input_mode=input_mode,
        random_state=5,
    ).fit(X, y)

    external_matrix_bytes = rows * (X.shape[1] + max(X.shape[1], 5)) * 8
    ingress_bytes = rows * X.shape[1] * 8 if layout == "strided_copy" else 0
    assert selector.diagnostics_["copied_bytes"] == external_matrix_bytes + ingress_bytes


@pytest.mark.parametrize("getter_kind", ["auto_coef", "dotted", "callable"])
def test_external_importance_getters_accept_linear_and_wrapped_estimators(getter_kind):
    import numpy as np
    from types import SimpleNamespace
    from sklearn.base import BaseEstimator

    class GetterEstimator(BaseEstimator):
        def fit(self, X, y):
            self.coef_ = np.vstack(
                [np.arange(1, X.shape[1] + 1), -np.arange(2, X.shape[1] + 2)]
            )
            self.nested_ = SimpleNamespace(scores_=np.ones(X.shape[1]))
            return self

    estimator = GetterEstimator()
    getter = {
        "auto_coef": "auto",
        "dotted": "nested_.scores_",
        "callable": lambda fitted: np.abs(fitted.coef_).sum(axis=0),
    }[getter_kind]
    selector = BankaiBoruta(
        estimator=estimator,
        importance_getter=getter,
        n_estimators=8,
        max_iter=1,
        random_state=4,
    ).fit(np.ones((24, 2)), np.tile([0, 1], 12))

    assert selector.n_iter_ == 1
    assert selector.diagnostics_["external_estimator_calls"] == 1


def test_auto_coef_aggregates_multiclass_magnitudes_per_feature():
    import numpy as np
    from sklearn.base import BaseEstimator

    class MulticlassLinearEstimator(BaseEstimator):
        def fit(self, X, y):
            coef = np.zeros((3, X.shape[1]))
            coef[0, 0] = 2.0
            coef[1, 0] = -1.0
            self.coef_ = coef
            return self

    y = np.tile([0, 1], 48)
    X = np.column_stack([y, np.arange(y.size) % 7]).astype(float)
    selector = BankaiBoruta(
        estimator=MulticlassLinearEstimator(),
        n_estimators=8,
        max_iter=12,
        alpha=0.05,
        random_state=14,
    ).fit(X, y)

    assert selector.support_.tolist() == [True, False]


def test_external_estimator_works_inside_sklearn_pipeline_cross_validation():
    import numpy as np
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    class StableEstimator(BaseEstimator):
        def fit(self, X, y):
            self.feature_importances_ = np.ones(X.shape[1], dtype=float)
            real_features = X.shape[1] - max(5, X.shape[1] // 2)
            self.feature_importances_[:real_features] = 2.0
            return self

    rng = np.random.default_rng(101)
    X = rng.normal(size=(36, 3))
    y = np.tile([0, 1], 18)
    pipeline = make_pipeline(
        BankaiBoruta(
            estimator=StableEstimator(),
            n_estimators=8,
            max_iter=12,
            alpha=0.01,
            random_state=9,
        ),
        LogisticRegression(),
    )

    scores = cross_val_score(pipeline, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_external_estimator_template_is_cloneable_pickleable_and_unfitted_after_fit():
    import pickle

    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    template = DecisionTreeClassifier(max_depth=3, random_state=15)
    selector = BankaiBoruta(
        estimator=template,
        n_estimators=8,
        max_iter=2,
        random_state=16,
    ).fit(np.arange(96, dtype=float).reshape(48, 2), np.tile([0, 1], 24))
    restored = pickle.loads(pickle.dumps(selector))

    recloned = clone(restored)
    assert recloned.estimator is not restored.estimator
    assert recloned.estimator.get_params() == restored.estimator.get_params()
    assert not hasattr(template, "tree_")
    assert not hasattr(restored.estimator, "tree_")
    assert not any(key in vars(restored) for key in ("X", "y", "estimator_", "forest_"))
    assert restored.n_features_in_ == 2
    assert restored.decision_.tolist() == selector.decision_.tolist()


def test_fitted_selector_is_pickleable_without_retaining_training_data():
    import pickle

    import numpy as np

    rng = np.random.default_rng(17)
    y = np.tile([0, 1], 32)
    X = np.column_stack([y + rng.normal(0, 0.02, y.size), rng.normal(size=y.size)])
    fitted = BankaiBoruta(
        n_estimators=64, max_iter=8, alpha=0.05, random_state=4
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))

    assert restored.decision_.tolist() == fitted.decision_.tolist()
    assert restored.transform(X).shape == (64, fitted.support_.sum())
    assert not any(key in vars(restored) for key in ("X", "y", "forest_", "training_data_"))


def test_refit_updates_feature_names_and_selector_works_in_pipeline():
    import numpy as np
    from sklearn.pipeline import Pipeline

    class NamedMatrix:
        def __init__(self, values, columns):
            self.values = values
            self.columns = columns

        def __array__(self, dtype=None, copy=None):
            return np.array(self.values, dtype=dtype, copy=copy)

    rng = np.random.default_rng(88)
    y = np.tile([0, 1], 40)
    X_values = np.column_stack([y + rng.normal(0, 0.03, y.size), rng.normal(size=y.size)])
    X = NamedMatrix(X_values, ["signal", "noise"])
    selector = BankaiBoruta(n_estimators=28, max_iter=8, random_state=7, input_mode="copy")
    selector.fit(X, y)
    assert selector.get_feature_names_out().tolist() == [
        name for name, selected in zip(X.columns, selector.get_support()) if selected
    ]
    selector.fit(X_values, y)
    assert not hasattr(selector, "feature_names_in_")

    plain_pipeline = Pipeline([("select", BankaiBoruta(n_estimators=28, max_iter=8, random_state=7))])
    transformed = plain_pipeline.fit_transform(X_values, y)
    assert transformed.shape[0] == y.size


def test_polars_dataframe_copy_mode_reports_materialization():
    pl = pytest.importorskip("polars")
    import numpy as np

    y = np.tile([0, 1], 32)
    X = pl.DataFrame({"signal": y.astype(float), "noise": np.arange(y.size, dtype=float)})

    fitted = BankaiBoruta(
        n_estimators=8, max_iter=1, input_mode="copy", random_state=2
    ).fit(X, y)
    assert fitted.n_features_in_ == 2
    assert fitted.feature_names_in_.tolist() == ["signal", "noise"]
    assert fitted.diagnostics_["copied_bytes"] >= X.shape[0] * X.shape[1] * 8
