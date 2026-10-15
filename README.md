# 🔥 Bankai Boruta

[![PyPI Latest Release](https://img.shields.io/pypi/v/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Package Status](https://img.shields.io/pypi/status/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Python versions](https://img.shields.io/pypi/pyversions/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![uv managed](https://img.shields.io/badge/uv-managed-blue)](https://docs.astral.sh/uv/)
[![Downloads](https://static.pepy.tech/badge/bankai-boruta)](https://pepy.tech/project/bankai-boruta)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](COPYING)
[![CI](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml)
[![Dependency security](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/breno-jesus-fernandes/bankai-boruta/branch/main/graph/badge.svg)](https://codecov.io/gh/breno-jesus-fernandes/bankai-boruta)
[![GitHub Release](https://img.shields.io/github/v/release/breno-jesus-fernandes/bankai-boruta?include_prereleases)](https://github.com/breno-jesus-fernandes/bankai-boruta/releases)
[![PyPI deployment](https://img.shields.io/github/deployments/breno-jesus-fernandes/bankai-boruta/pypi?label=PyPI%20deployment)](https://github.com/breno-jesus-fernandes/bankai-boruta/deployments)
[![GitHub stars](https://img.shields.io/github/stars/breno-jesus-fernandes/bankai-boruta.svg?style=social)](https://github.com/breno-jesus-fernandes/bankai-boruta)

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

## Selector reference

`BankaiBoruta` follows scikit-learn's selector pattern. The complete defaults are:

```python
selector = BankaiBoruta(
    n_estimators=40,
    max_iter=100,
    alpha=0.05,
    pruning="canonical",
    importance=None,
    importance_scale="zscore",
    input_mode="zero_copy",
    max_features="sqrt",
    max_depth=6,
    n_jobs=-1,
    random_state=2077,
    max_bins=63,
    store_history=False,
    estimator=None,
    importance_getter="auto",
    forest_backend="native",
)
```

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
| `forest_backend` | `"native"` | Built-in Rust forest or `"bankai"` for the optional Python Bankai estimator. |
| `estimator` | `None` | Cloneable custom estimator; overrides the backend when provided. |
| `importance_getter` | `"auto"` | Reads `feature_importances_`, falling back to `coef_`; also accepts a dotted attribute path or callable. Used by external estimators. |

The native backend uses permutation importance by default. The Bankai backend
uses gain by default and also supports split, permutation, and SHAP importance.
`importance_scale` accepts `"zscore"` or `"raw"` and applies only to the native
forest. `max_bins` applies only to the Bankai backend; use `None` to disable
histograms or an integer from 2 to 255 to enable them. A custom `estimator`
controls its own forest parameters and importance calculation.

### Methods and fitted results

Call `fit(X, y)` for binary or multiclass classification with finite numeric
features. Use `transform(X)` or `fit_transform(X, y)` to obtain accepted columns,
`get_support(indices=True)` for their indices, and `get_feature_names_out()` for
their names. Tentative features are reported separately and are not selected
by `transform`.

| Result | Meaning |
| --- | --- |
| `support_` | Boolean mask of accepted features. |
| `support_weak_` | Boolean mask of tentative features. |
| `decision_` | Per-feature status: 0 rejected, 1 accepted, 2 tentative. |
| `n_features_in_`, `feature_names_in_` | Input width and, when supplied, string feature names. |
| `n_iter_`, `stop_reason_` | Completed iterations and reason selection stopped. |
| `diagnostics_` | Timing, memory, input, and external-estimator diagnostics. |
| `history_` | Per-iteration results, present only with `store_history=True`. |

Canonical pruning retains accepted features as context and removes rejected
features from later fits. Aggressive pruning also removes accepted features
and is experimental. Repeated statistical decisions do not guarantee global
control of false discoveries across all iterations.

## Backends and supported inputs

The default native backend runs forest training and the Boruta loop in Rust.
It borrows C- or F-contiguous, aligned NumPy `float32`/`float64` features.
Use `input_mode="copy"` for numeric layouts requiring conversion.

Install the `bankai-rf` extra to use the optional Bankai Random Forest backend:

```python
selector = BankaiBoruta(forest_backend="bankai", importance="gain")
selector.fit(X, y)
```

Both the Bankai backend and custom estimators materialize active features and
shadows per iteration; the Boruta controller remains in Rust. Optional Polars
and Arrow inputs support finite Float32/Float64 columns. Nulls, NaN, infinite
values, sparse features, regression, and multioutput targets are unsupported.
`transform` allocates its output.

See [input and estimator notes](docs/usage.md) for Arrow ownership, custom
importance getters, and examples with scikit-learn pipelines.

## Save and restore selectors

Fitted selectors can be saved with joblib:

```python
import joblib

joblib.dump(selector, "selector.joblib")
restored = joblib.load("selector.joblib")
X_selected = restored.transform(X)
```

The selector retains parameters and selection results rather than fitted
per-iteration forests or training data. A custom estimator template is also
serialized; provide an unfitted template to avoid retaining its training state.
Only load trusted joblib files and keep package versions consistent.

## Development

Install Git, Python 3.11 or newer, stable Rust, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). Windows requires
Rust MSVC build tools; macOS requires the Xcode command-line tools.

Clone the repository and build the locked development environment:

```bash
git clone https://github.com/breno-jesus-fernandes/bankai-boruta.git
cd bankai-boruta
uv python install 3.11
uv sync --locked --no-install-project --extra test --extra arrow-test
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest -q
uv run --no-sync cargo test --workspace --locked
```

See the [contribution guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
See [CI and releases](docs/development.md) for the automation and PyPI setup.
The [roadmap](docs/ROADMAP.md) tracks project status; the
[FFI ownership notes](crates/arrow-ffi/SAFETY.md) describe buffer ownership.

## License

Bankai Boruta is licensed under GPL-3.0-or-later. See [COPYING](COPYING).
