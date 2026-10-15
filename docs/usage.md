# Input and estimator notes

See the [selector reference](../README.md#selector-reference) for all parameters,
defaults, methods, and fitted results.

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

## Pipeline and cross-validation

Selection belongs inside the pipeline so each fold learns its support mask
using only training data:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from bankai_boruta import BankaiBoruta

X, y = make_classification(n_samples=800, n_features=20, random_state=2077)
pipeline = make_pipeline(
    BankaiBoruta(random_state=2077),
    LogisticRegression(max_iter=1000),
)
scores = cross_val_score(pipeline, X, y, cv=3)
print(scores)
```

Custom `coef_` scores are reduced by summing absolute values over class axes.
The built-in Bankai permutation getter shifts all scores by their shared minimum
to preserve ordering while satisfying the non-negative score contract. Custom
getters must implement any necessary transformation themselves.

For the Arrow zero-copy resolution and ownership constraints, see the
[M6 notes](limitations/M6-resolution-plan.md) and
[FFI safety notes](../crates/arrow-ffi/SAFETY.md).
