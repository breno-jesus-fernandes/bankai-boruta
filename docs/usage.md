# Usage and API notes

## Selector parameters

`BankaiBoruta()` defaults to 40 trees, depth 6, all available workers, seed
2077, and 63 bins for the Bankai backend. When `importance` is omitted, the
native backend uses permutation importance and the Bankai backend uses gain
importance.

| Parameter | Default | Description |
| --- | --- | --- |
| `n_estimators` | `40` | Number of trees per Boruta iteration. |
| `max_iter` | `100` | Maximum number of Boruta iterations. |
| `alpha` | `0.05` | Significance level for feature decisions. |
| `pruning` | `"canonical"` | Boruta pruning mode. Canonical mode keeps confirmed features as context; `"aggressive"` removes them and is experimental. |
| `importance` | `None` | Selects the backend default: permutation for native, gain for Bankai. Bankai also supports split and SHAP. |
| `importance_scale` | `"zscore"` | Scale applied to built-in native importance scores. |
| `max_features` | `"sqrt"` | Candidate features considered at each split: `"sqrt"`, `"log2"`, a count, a fraction, or `None`. |
| `max_depth` | `6` | Maximum forest tree depth. `None` uses the engine's depth cap. |
| `n_jobs` | `-1` | Worker count. `-1` uses all available CPUs. |
| `random_state` | `2077` | Seed for repeatable fits. |
| `max_bins` | `63` | Histogram bins for the Bankai backend; `None` disables histogram splitting. |
| `input_mode` | `"zero_copy"` | Input handling mode. `"copy"` enables conversion for otherwise unsupported numeric layouts. |
| `store_history` | `False` | Whether to retain per-iteration decision history. |

The selector exposes scikit-learn style `support_`, `support_weak_`,
`decision_`, `transform`, and `get_support` attributes and methods.

## NumPy, Polars, and Arrow input

The native backend accepts C- or F-contiguous NumPy `float32` and `float64`
arrays without copying the feature matrix. Use `input_mode="copy"` for numeric
inputs that need conversion.

With `bankai-boruta[polars]`, supported Polars Float32 and Float64 DataFrames
can be imported through the Arrow C Stream interface. Slices and independently
chunked columns are supported. Nulls, NaN/Inf values, unsupported types, and
unaligned buffers raise an error; use `input_mode="copy"` for integer data.
The Bankai forest backend materializes active features and shadow columns for
each estimator fit. `transform` also allocates the selected output.

Other CPU Arrow C Stream table exporters are accepted by the importer, though
their own export methods may allocate. PyArrow is not a runtime dependency.

## Custom estimators

Pass a cloneable scikit-learn compatible estimator that exposes
`feature_importances_` or `coef_` to use another importance provider:

```python
from bankai_boruta import BankaiBoruta
from bankai_random_forest import BankaiRandomForestClassifier

selector = BankaiBoruta(
    estimator=BankaiRandomForestClassifier(
        n_estimators=40,
        max_depth=6,
        max_bins=63,
        importance_type="gain",
        n_jobs=-1,
        random_state=2077,
    ),
    random_state=2077,
)
```

Rust continues to handle shadows, pruning, and statistical decisions. The
estimator is fitted once per Boruta iteration on a materialized matrix of the
active features and shadows. `importance_getter="auto"` reads
`feature_importances_`, falling back to `coef_`; a dotted path or callable can
provide custom scores. A custom getter must return one finite, non-negative
score per augmented input column.

