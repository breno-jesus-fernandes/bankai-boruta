# 🔥 Bankai Boruta

[![PyPI Latest Release](https://img.shields.io/pypi/v/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Package Status](https://img.shields.io/pypi/status/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Python versions](https://img.shields.io/pypi/pyversions/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![uv managed](https://img.shields.io/badge/uv-managed-blue)](https://docs.astral.sh/uv/)
[![Downloads](https://static.pepy.tech/badge/bankai-boruta)](https://pepy.tech/project/bankai-boruta)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](COPYING)
[![CI](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/breno-jesus-fernandes/bankai-boruta?include_prereleases)](https://github.com/breno-jesus-fernandes/bankai-boruta/releases)
[![GitHub stars](https://img.shields.io/github/stars/breno-jesus-fernandes/bankai-boruta.svg?style=social)](https://github.com/breno-jesus-fernandes/bankai-boruta)

Bankai Boruta is a feature selection library powered by a native Rust core. It
implements the Boruta all-relevant feature selection algorithm and provides a
scikit-learn compatible Python API. Boruta compares each real feature with
shuffled shadow features across repeated forest fits, then classifies features
as accepted, rejected, or tentative.

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

## Development

Build an editable install and run the Python suite with `uv`:

```bash
uv sync --extra test
uv run maturin develop --skip-install
uv run pytest
```

Run Rust core tests with Cargo:

```bash
cargo test --workspace
```

The Arrow development suite requires its optional dependencies:

```bash
uv sync --extra test --extra arrow-test
uv run --no-sync maturin develop --release --skip-install
uv run --no-sync pytest
```

See [the M6 plan](docs/limitations/M6-resolution-plan.md),
[FFI ownership notes](crates/arrow-ffi/SAFETY.md), and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

Bankai Boruta is licensed under GPL-3.0-or-later. See [COPYING](COPYING).
