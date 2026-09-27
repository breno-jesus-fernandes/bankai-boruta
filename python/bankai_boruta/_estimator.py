"""Scikit-learn estimator interface for the native Boruta engine."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class BankaiBoruta(SelectorMixin, BaseEstimator):
    """Select all relevant features with the native Bankai Boruta engine.

    The public estimator contract is established before the native selection
    loop is available. Calling ``fit`` in this development milestone fails
    explicitly rather than returning an unfitted estimator.
    """

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
        self._validate_parameters()

    def fit(self, X: Any, y: Any) -> BankaiBoruta:
        raise RuntimeError("native Boruta engine is not available before milestone M4")

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
            if isinstance(self.max_features, float) and self.max_features > 1:
                raise ValueError("a floating max_features value must be at most 1")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, Integral) or self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None")
        if not isinstance(self.store_history, bool):
            raise ValueError("store_history must be a bool")

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
