# Bankai Boruta

Native Rust implementation of the Boruta all-relevant feature selection
algorithm with a scikit-learn compatible Python API.

The canonical classification selector now runs its complete Boruta iteration
loop inside Rust. `BankaiBoruta.fit(X, y)` accepts C- or F-contiguous NumPy
`float32`/`float64` arrays without copying `X`; incompatible numeric inputs can
use `input_mode="copy"`. Rejected features are pruned between iterations,
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
Stream, retaining the original value buffers for the duration of `fit`.
No rechunk, NumPy conversion or augmented matrix is needed. Nulls, NaN/Inf,
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

## Optional scikit-learn estimator

The default `BankaiBoruta()` continues to use the native Rust forest and does
not call Python inside the iteration loop. To use a scikit-learn estimator,
pass a cloneable classifier that exposes `feature_importances_`:

```python
from bankai_boruta import BankaiBoruta
from sklearn.ensemble import ExtraTreesClassifier

selector = BankaiBoruta(
    estimator=ExtraTreesClassifier(n_estimators=64, random_state=42),
    importance_getter="auto",
    random_state=42,
)
selector.fit(X, y)
```

Bankai still owns the Boruta controller, shadow generation, pruning and
statistical decisions in Rust. The opt-in path clones and fits the Python
estimator once per iteration on a C-contiguous matrix containing active real
features followed by shadows. This requires a Python callback and materializes
that augmented matrix on every iteration, so it trades speed and memory for
model flexibility. `diagnostics_["copied_bytes"]` counts those matrix bytes;
`external_estimator_calls` and `external_estimator_seconds` report callback
count and time, while `fit_time_seconds` reports total selector fit time. The
estimator template is not fitted or otherwise mutated. At present,
`importance_getter="auto"` reads `feature_importances_`, falling back to
`coef_` and summing absolute coefficients across classes. A dotted attribute
path or callable can supply custom scores. Getters must return one finite,
non-negative score per augmented input column. The native mode remains the
recommended default for performance and zero-copy input.

The Arrow development suite must run with its optional dependencies installed:

```bash
uv sync --extra test --extra arrow-test
uv run --no-sync maturin develop --release --skip-install
uv run --no-sync pytest
```

See [the M6 plan](docs/limitations/M6-resolution-plan.md) and
[FFI ownership notes](crates/arrow-ffi/SAFETY.md).
