"""Scikit-learn estimator interface for the native Boruta engine."""

from __future__ import annotations

from numbers import Integral, Real
from math import isfinite
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class BankaiBoruta(SelectorMixin, BaseEstimator):
    """Select relevant numeric features with the native Bankai Boruta engine."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_iter: int = 100,
        alpha: float = 0.01,
        pruning: str = "canonical",
        importance: str = "permutation",
        importance_scale: str = "zscore",
        input_mode: str = "zero_copy",
        max_features: str | int | float | None = "sqrt",
        max_depth: int | None = 512,
        n_jobs: int = 1,
        random_state: int | None = None,
        store_history: bool = False,
        estimator: Any = None,
        importance_getter: Any = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.pruning = pruning
        self.importance = importance
        self.importance_scale = importance_scale
        self.input_mode = input_mode
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_history = store_history
        self.estimator = estimator
        self.importance_getter = importance_getter
        self._validate_parameters()

    def fit(self, X: Any, y: Any) -> BankaiBoruta:
        import numpy as np
        import time
        from sklearn.utils.multiclass import check_classification_targets
        from sklearn.utils.validation import check_array

        from . import _core
        fit_started = time.perf_counter()

        # A failed refit must not expose stale fitted state from the preceding fit.
        for name in tuple(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)
        self._validate_parameters()
        is_numpy = isinstance(X, np.ndarray)
        is_arrow = hasattr(X, "__arrow_c_stream__")
        if self.input_mode == "zero_copy" and not (is_numpy or is_arrow):
            raise ValueError("zero_copy requires a NumPy ndarray or Arrow C Stream input; use input_mode='copy'")
        y_array = np.asarray(y)
        if y_array.ndim != 1 or (is_numpy and X.ndim != 2):
            raise ValueError("X must be 2D and y must be 1D")
        if is_numpy and X.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        y_array = check_array(y_array, ensure_2d=False, dtype=None)
        check_classification_targets(y_array)
        classes, encoded_y = np.unique(y_array, return_inverse=True)
        if classes.size < 2:
            raise ValueError("classification requires at least two classes")
        encoded_y = np.ascontiguousarray(encoded_y, dtype=np.int64)
        if isinstance(self.max_features, str) or self.max_features is None:
            max_features = self.max_features
        else:
            max_features = float(self.max_features) if isinstance(self.max_features, float) else int(self.max_features)
        # Polars' copy path exports once; borrow that buffer inside Rust when its
        # dtype is already supported. No second np.array(copy=True) is needed.
        import sys
        native_X, native_mode, materialized_bytes = X, self.input_mode, 0
        polars = sys.modules.get("polars")
        if self.input_mode == "copy" and polars is not None and isinstance(X, polars.DataFrame):
            native_X = X.to_numpy(order="c", writable=True)
            materialized_bytes = int(native_X.nbytes)
            if native_X.dtype in (np.dtype("float32"), np.dtype("float64")):
                native_mode = "zero_copy"
        result = _core.fit_boruta(
            native_X,
            encoded_y,
            int(classes.size),
            self.n_estimators,
            self.max_iter,
            float(self.alpha),
            max_features,
            self.pruning,
            self.max_depth,
            self.n_jobs,
            self.random_state,
            self.importance_scale,
            self.store_history,
            native_mode,
            self.estimator,
            self.importance_getter,
        )

        decisions = np.asarray(result["decisions"], dtype=np.uint8)
        self.support_ = decisions == 1
        self.support_weak_ = decisions == 2
        self.decision_ = decisions
        self.n_features_in_ = result["shape"][1]
        if "feature_names" in result:
            self.feature_names_in_ = np.asarray(result["feature_names"], dtype=object)
        elif hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_iter_ = result["n_iter"]
        self.stop_reason_ = result["stop_reason"]
        self.diagnostics_ = result["diagnostics"]
        self.diagnostics_["copied_bytes"] += materialized_bytes
        if self.estimator is not None:
            self.diagnostics_["fit_time_seconds"] = time.perf_counter() - fit_started
        if self.store_history:
            self.history_ = result["history"]
        elif hasattr(self, "history_"):
            del self.history_
        return self

    def _validate_parameters(self) -> None:
        for name, value in (("n_estimators", self.n_estimators), ("max_iter", self.max_iter), ("max_depth", self.max_depth)):
            if value is None and name == "max_depth":
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be a number strictly between 0 and 1")
        if self.pruning not in {"canonical", "aggressive"}:
            raise ValueError("pruning must be 'canonical' or 'aggressive'")
        if self.importance != "permutation":
            raise ValueError("importance must be 'permutation' in this release")
        if self.importance_scale not in {"zscore", "raw"}:
            raise ValueError("importance_scale must be 'zscore' or 'raw'")
        if self.input_mode not in {"zero_copy", "copy"}:
            raise ValueError("input_mode must be 'zero_copy' or 'copy'")
        if isinstance(self.max_features, str):
            if self.max_features not in {"sqrt", "log2"}:
                raise ValueError("max_features must be 'sqrt', 'log2', a positive number, or None")
        elif self.max_features is not None:
            if isinstance(self.max_features, bool) or not isinstance(self.max_features, Real) or self.max_features <= 0:
                raise ValueError("max_features must be 'sqrt', 'log2', a positive number, or None")
            if not isfinite(float(self.max_features)):
                raise ValueError("max_features must be finite")
            if isinstance(self.max_features, float) and self.max_features > 1:
                raise ValueError("a floating max_features value must be at most 1")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, Integral) or self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None")
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not isinstance(self.store_history, bool):
            raise ValueError("store_history must be a bool")
        if not (
            callable(self.importance_getter)
            or isinstance(self.importance_getter, str)
            and bool(self.importance_getter)
            and all(self.importance_getter.split("."))
        ):
            raise ValueError(
                "importance_getter must be 'auto', a dotted attribute path, or callable"
            )

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
