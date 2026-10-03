# Bankai Boruta

Native Rust implementation of the Boruta all-relevant feature selection
algorithm with a scikit-learn compatible Python API.

The canonical classification selector keeps its Boruta iteration loop and
statistical decisions in Rust. It can use the built-in Rust forest or
`BankaiRandomForestClassifier` as the importance provider. With the native
backend, `BankaiBoruta.fit(X, y)` accepts C- or F-contiguous NumPy `float32`/
`float64` arrays without copying `X`; incompatible numeric inputs can use
`input_mode="copy"`. The Bankai backend materializes the active features and
shadows for each estimator fit. Rejected features are pruned between iterations,
confirmed features remain as context in canonical mode, and unresolved
features stay tentative when `max_iter` is reached. `pruning="aggressive"`
removes confirmed features from subsequent training and is experimental; use it
only when the benchmarked trade-off is acceptable for your data. Deterministic
multi-worker training is available with `n_jobs`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for details.

## Development

Python wheels and editable installs use Maturin:

```bash
uv sync --extra test
uv run maturin develop --skip-install
uv run pytest
```

Rust core tests run with Cargo:

```bash
cargo test --workspace
```

## Polars and Arrow input

Install `bankai-boruta[polars]` to select directly from Polars DataFrames.
The strict path supports Float32/Float64 columns, mixed float widths, slices,
and independently chunked columns. It imports each Series through Arrow C
Stream, retaining the original value buffers for the duration of `fit` with the
native backend. The Bankai backend materializes its augmented estimator input.
Nulls, NaN/Inf,
unsupported types and unaligned buffers raise an error. Integers can use the
explicit `input_mode="copy"` conversion path.

```python
from bankai_boruta import BankaiBoruta
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

selector = BankaiBoruta(random_state=42, n_jobs=4)
pipeline = make_pipeline(selector, DecisionTreeClassifier(random_state=42))
scores = cross_val_score(pipeline, X_polars, y, cv=3)
```

Selection belongs inside the cross-validation pipeline. For a fitted selector,
`set_output(transform="polars")` preserves named Polars output. `transform` may
allocate its selected output. Pickle/joblib store decisions and metadata, not
training buffers or forests; a failed refit clears the preceding fitted state.

Other CPU Arrow C Stream table exporters are accepted without copying buffers
inside the importer. Their own export methods may allocate; the end-to-end
zero-copy guarantee is tested specifically for supported Polars float layouts.
`copied_bytes` measures known matrix materializations, not all allocations or
allocations internal to third-party exporters. Labels, shadow indices, chunk
metadata and forest workspaces still allocate. Keep inputs unchanged during fit.
PyArrow is not a runtime dependency.

## Feature selection with Bankai Random Forest

Install the optional Bankai Random Forest integration with:

```bash
python -m pip install 'bankai-boruta[bankai-rf]'
```

This example splits the data before selection, so the test set does not affect
which features Boruta accepts:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bankai_boruta import BankaiBoruta
from bankai_random_forest import BankaiRandomForestClassifier

X, y = make_classification(
    n_samples=800, n_features=20, n_informative=5, n_redundant=0,
    random_state=42,
)
feature_names = np.asarray([f"feature_{i:02d}" for i in range(X.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Reuse Bankai Random Forest's native Rust training and importance implementation.
selector = BankaiBoruta(
    forest_backend="bankai", n_estimators=300, max_iter=100, random_state=42
)
selector.fit(X_train, y_train)

accepted = feature_names[selector.support_]
tentative = feature_names[selector.support_weak_]
print("Accepted:", accepted.tolist())
print("Still tentative:", tentative.tolist())

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
model = BankaiRandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_selected, y_train)
print("Test accuracy:", model.score(X_test_selected, y_test))
```

The native-backend example is also available in [`examples/`](examples/).

## Forest backends and custom estimators

`forest_backend="native"` is the default and uses Boruta's built-in Rust forest.
Set `forest_backend="bankai"` to use `BankaiRandomForestClassifier` from the
optional `bankai-rf` extra. The backend maps the Boruta tree count, depth,
feature sampling, worker count and seed to the Random Forest estimator. Its
permutation importances are shifted by one shared minimum when needed; this
preserves real-versus-shadow comparisons while satisfying the non-negative
score contract. Set `importance="gain"`, `"split"`, or `"shap"` to use the
corresponding importance mode from the Random Forest package; the built-in
native backend continues to use permutation importance.

The Bankai provider fits a cloned Python estimator once per Boruta iteration.
Rust still owns shadow generation, pruning and statistical decisions, but each
iteration materializes the active features and shadows into a matrix. This
path trades memory and callback overhead for the latest Random Forest engine.
`importance_scale` applies to the built-in native provider; external providers
use the estimator's score scale.

To pass a configured Random Forest (including options such as `max_bins` or
`importance_type="shap"`), or use another scikit-learn-compatible classifier,
pass a cloneable
estimator that exposes `feature_importances_`. Bankai keeps the Boruta
controller, shadow generation, pruning and statistical decisions in Rust, but
fits the Python estimator once per iteration on a materialized matrix of active
features and shadows. This callback path trades speed and memory for model
flexibility. `diagnostics_["copied_bytes"]` counts those matrix bytes;
`external_estimator_calls` and `external_estimator_seconds` report callback
count and time. The estimator template is not fitted or otherwise mutated.
`importance_getter="auto"` reads `feature_importances_`, falling back to
`coef_`; a dotted attribute path or callable can provide custom scores. Getters
must return one finite, non-negative score per augmented input column.

For example, custom Random Forest settings expose the new histogram and SHAP
options directly:

```python
from bankai_boruta import BankaiBoruta
from bankai_random_forest import BankaiRandomForestClassifier

selector = BankaiBoruta(
    estimator=BankaiRandomForestClassifier(
        n_estimators=500, max_bins=63, importance_type="shap", random_state=42
    ),
    importance_getter="auto",
    random_state=42,
)
```

The Arrow development suite must run with its optional dependencies installed:

```bash
uv sync --extra test --extra arrow-test
uv run --no-sync maturin develop --release --skip-install
uv run --no-sync pytest
```

See [the M6 plan](docs/limitations/M6-resolution-plan.md) and
[FFI ownership notes](crates/arrow-ffi/SAFETY.md).
