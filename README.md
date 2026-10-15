# 🔥 Bankai Boruta

Bankai Boruta is a feature selection library powered by a native Rust core. It
implements the Boruta all-relevant feature selection algorithm and provides a
scikit-learn compatible Python API. Boruta compares each real feature with
shuffled shadow features across repeated forest fits, then classifies features
as accepted, rejected, or tentative.

The project is in alpha, so APIs and serialized selectors may change.

The native backend handles shadow generation, forest training, pruning, and
statistical decisions in Rust. An optional Bankai Random Forest backend can
provide feature importances while the Rust core runs the Boruta loop.

## Install

Python 3.11 or newer is required. Install the base package from PyPI:

```bash
python -m pip install bankai-boruta
```

Install optional integrations with extras:

```bash
python -m pip install 'bankai-boruta[bankai-rf]'
python -m pip install 'bankai-boruta[polars]'
```

## Quick start

This example uses the built-in Rust forest and prints the selected and tentative
feature names:

```python
import numpy as np
from sklearn.datasets import make_classification

from bankai_boruta import BankaiBoruta

X, y = make_classification(
    n_samples=800,
    n_features=20,
    n_informative=5,
    n_redundant=0,
    random_state=2077,
)
feature_names = np.asarray([f"feature_{i:02d}" for i in range(X.shape[1])])

selector = BankaiBoruta(random_state=2077)
selector.fit(X, y)

print("Accepted:", feature_names[selector.support_].tolist())
print("Tentative:", feature_names[selector.support_weak_].tolist())
```

For unbiased model evaluation, put selection inside the cross-validation
pipeline so each fold selects features using only its training data.

## Use Bankai Random Forest

Install the `bankai-rf` extra, then select the Bankai forest as the importance
provider. This configuration uses 40 trees, depth 6, gain importance, histogram
binning with 63 bins, all available workers, and seed 2077:

```python
from bankai_boruta import BankaiBoruta

selector = BankaiBoruta(
    forest_backend="bankai",
    n_estimators=40,
    max_depth=6,
    importance="gain",
    max_bins=63,
    n_jobs=-1,
    random_state=2077,
)
selector.fit(X, y)
```

The Bankai backend maps `n_estimators`, `max_depth`, `max_features`, `n_jobs`,
`random_state`, `importance`, and `max_bins` to
`BankaiRandomForestClassifier`. The native backend is the default and uses
permutation importance. Bankai supports `importance="gain"`, `"split"`,
`"permutation"`, or `"shap"`. `max_bins=None` disables histogram split search;
an integer from 2 to 255 enables it.

## API and supported inputs

- `support_` marks accepted features; `support_weak_` marks tentative features.
  Use `transform` or `get_support` to retrieve the selection.
- The native backend accepts contiguous NumPy `float32` and `float64` arrays
  without copying the feature matrix. Use `input_mode="copy"` for conversion.
- Optional Polars and Arrow input supports finite Float32/Float64 columns.
  Nulls, NaN, and infinite values are rejected.
- Custom scikit-learn estimators can provide `feature_importances_` or `coef_`.
  External estimators materialize active features and shadows on each iteration.

See [usage and API notes](docs/usage.md) for selector parameters, input
requirements, and custom estimator examples.

## Development

Building from source requires Python 3.11 or newer, Rust, and `uv`:

```bash
uv sync --locked --no-install-project --extra test --extra arrow-test
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest -q
uv run --no-sync cargo test --workspace --locked
```

See [CI and releases](docs/development.md) for the automation and PyPI setup.
The [roadmap](docs/ROADMAP.md) tracks project status; the
[FFI ownership notes](crates/arrow-ffi/SAFETY.md) describe buffer ownership.

## License

Bankai Boruta is licensed under GPL-3.0-or-later. See [COPYING](COPYING).
