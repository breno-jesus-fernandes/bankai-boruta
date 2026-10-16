# Bankai Boruta 0.1.0a3

Classification feature selection with a scikit-learn interface. Classic Boruta
is the default; `greedy=True` selects features that beat the shadow percentile
at least once in a fixed campaign.

```sh
python -m pip install bankai-boruta
```

BankaiRF **0.1.0a15** is installed automatically.

```python
from bankai_boruta import BankaiBoruta
selector = BankaiBoruta(random_state=42)
selected = selector.fit_transform(X, y)
print(selector.support_, selector.support_weak_, selector.ranking_)
```

The default classifier is BankaiRF with `extra_trees=True`,
`extra_trees_backend="binned"`, `max_bins=63`, `importance_type="gain"`,
`max_depth=6`, and `n_jobs=-1`. Other settings follow a15 defaults.

Pass a classifier to customize it; nested parameters use scikit-learn's normal
`estimator__max_depth` syntax. The supplied estimator is cloned. External
classifiers receive original labels, including strings. NumPy and pandas inputs
are supported; pandas is converted to NumPy. Polars and Arrow are outside this release.

```python
from bankai_random_forest import BankaiRandomForestClassifier
selector = BankaiBoruta(
    BankaiRandomForestClassifier(class_weight="balanced", max_depth=5),
    n_estimators=None, greedy=True, random_state=42,
)
selector.fit(X, y, sample_weight=weights)
```

`n_estimators="auto"` uses BorutaPy's feature/depth heuristic; an integer overrides
the tree count and `None` preserves it (also suitable for classifiers without a
tree count). `importance_getter="auto"` uses `feature_importances_` or absolute
coefficient magnitude (L2 across classes). A dotted attribute path or callable
can return another finite importance vector; larger scores must mean more important.

Classic mode uses `perc=100`, `alpha=.05`, `two_step=True`, `max_iter=100`,
`early_stopping=False`, and `n_iter_no_change=20`. Like BorutaPy, its loop runs
while the iteration index is less than `max_iter`. Hits require strictly greater
importance than the shadow percentile. Two-step correction combines BH FDR with
Bonferroni across rounds; `two_step=False` uses Bonferroni across original
features. The median rough fix marks surviving tentative features as weak.
Confirmed features rank 1, weak features rank 2, and other features rank by median
importance ranks. `transform` includes confirmed features only.

Greedy keeps every feature in every round. Its round count is
`ceil(log2(p) - log2(alpha))` with `bonferroni=True`, and
`ceil(-log2(alpha))` otherwise. The ceiling is computed with the exact float
ratio to avoid overflow and rounding errors at power-of-two boundaries. Any hit confirms a feature, all others rank 2, and `support_weak_` is false.
`max_iter`, `two_step`, and early stopping do not control greedy.
Constructor parameters never change during fit; `max_iter_` records the effective
budget and `n_iter_` the rounds actually performed.

`store_history=True` retains `importance_history_` (initial zero row, then rounds)
and `shadow_threshold_history_`. `diagnostics_` reports execution path, fallback
reason, preparation/training time, input conversion time, augmented matrix copy
time/bytes and storage rebuilds. Copy metrics cover the augmented float matrix;
they do not count a15 internal bin/cache allocations. On the native path,
`estimator_` holds cloned configuration; on the adapter path it holds the last
round's fitted classifier, trained on augmented columns.
`verbose` enables per-round output. Feature names, cloning, pipelines,
`get_support`, and `get_feature_names_out` follow scikit-learn.

## Native integration

The Rust dependency is fixed to a15 commit
`caa46c22ef93434cd87319490513e877d4378564`. The validated native subset is the
default binned Extra Trees gain configuration, optionally changing tree count,
seed, depth, bins or Extra Trees seed. It uses a15 `DenseInput` and XRF directly,
with virtual shadow access and per-tree gain normalization. Trees run in parallel;
rounds run sequentially. Training storage is rebuilt each round as required by a15.
No cache persists between fits. Other settings, sample weights, monotonic
constraints, and custom getters use the Python adapter. Shadow monotonic
constraints are zero. See [validation](docs/validation.md) for measured limits.

## Migration from a2

The selector now takes `estimator` first. Removed selector arguments include
`forest_backend`, `pruning`, `importance`, `importance_scale`, `input_mode`,
`max_features`, `max_depth`, `n_jobs`, and `max_bins`. Configure forest parameters
on a supplied classifier instead. Use `greedy=True` for hit-once selection;
the former aggressive pruning mode is not equivalent. Optional `bankai-rf`,
`polars`, and `arrow-test` extras were removed. Old performance claims and
benchmarks do not apply to this rewritten implementation.

[Development and checks](docs/development.md). Licensed GPL-3.0-or-later.
