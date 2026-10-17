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

BankaiBoruta is a fast, scalable feature selection library built for large datasets.

It combines the Boruta all-relevant selection algorithm with the native Rust
backend of [Bankai Random Forest](https://github.com/breno-jesus-fernandes/bankai-random-forest).
Classic Boruta is the default; `greedy=True` enables the fixed-round Greedy Boruta
variant. BankaiRF supplies binned Extra Trees, gain importance and parallel tree
training through a scikit-learn-compatible selector for classification.

⚠️ Bankai Boruta is in Alpha. Its API and fitted objects may change.

## Install

Python 3.11 or newer is required. The release installation command is:

```bash
python -m pip install bankai-boruta
```

## Quick start

This example generates a binary classification dataset with 1,000 rows and 20
features. It fits selection on the training partition and applies the same
feature mask to the held-out partition. A pandas DataFrame supplies feature names.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bankai_boruta import BankaiBoruta

values, y = make_classification(
    n_samples=1_000,
    n_features=20,
    n_informative=5,
    n_redundant=3,
    random_state=2077,
)
X = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(values.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2077,
)

selector = BankaiBoruta(
    n_estimators="auto",
    perc=100,
    alpha=0.05,
    max_iter=100,
    random_state=2077,
)
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
print(f"Selected features: {selector.n_features_}")
print(selector.get_feature_names_out())
print(f"Train shape: {X_train_selected.shape}; test shape: {X_test_selected.shape}")

selection = pd.DataFrame({
    "feature": selector.feature_names_in_,
    "confirmed": selector.support_,
    "tentative": selector.support_weak_,
    "rank": selector.ranking_,
})
print(selection.sort_values("rank"))
```

`transform` returns confirmed features only. `fit_transform(X, y)` combines the
fit and transform steps. NumPy arrays and pandas DataFrames are supported;
pandas inputs are converted to NumPy for training.

## Selector reference

`BankaiBoruta` follows scikit-learn's estimator conventions and uses parameter
names familiar from BorutaPy. Its constructor defaults are:

```python
from bankai_boruta import BankaiBoruta

selector = BankaiBoruta(
    estimator=None,
    n_estimators="auto",
    perc=100,
    alpha=0.05,
    two_step=True,
    max_iter=100,
    random_state=None,
    verbose=0,
    early_stopping=False,
    n_iter_no_change=20,
    greedy=False,
    bonferroni=True,
    importance_getter="auto",
    store_history=False,
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `estimator` | `None` | A clonable classifier. `None` creates BankaiRF with the default configuration shown below. The supplied estimator is preserved. |
| `n_estimators` | `"auto"` | BorutaPy's feature/depth heuristic for tree count. A positive integer overrides the count; `None` preserves the estimator's setting and supports classifiers without a tree-count parameter. |
| `perc` | `100` | Shadow-importance percentile in `(0, 100]`. A real feature scores a hit only when its importance is strictly above this threshold. |
| `alpha` | `0.05` | Significance level in `(0, 1)` for classic tests and the greedy round budget. |
| `two_step` | `True` | Classic: combine Benjamini–Hochberg FDR across active features with Bonferroni across rounds. `False` uses Bonferroni across the original features. Ignored in greedy mode. |
| `max_iter` | `100` | Classic iteration bound. Like BorutaPy 0.4.3, the loop uses iteration indices below this bound, allowing up to `max_iter - 1` rounds. Ignored in greedy mode. |
| `random_state` | `None` | Integer seed, NumPy `RandomState`, or `None`. Controls shadows and per-round forest seeds without replacing constructor parameters. |
| `verbose` | `0` | Non-negative verbosity level. Positive values print round progress and confirmed-feature counts. |
| `early_stopping` | `False` | Classic: stop when the decisions remain unchanged beyond the patience limit. Ignored in greedy mode. |
| `n_iter_no_change` | `20` | Positive patience limit for classic early stopping. |
| `greedy` | `False` | Enable fixed-round, hit-once selection instead of classic binomial decisions. |
| `bonferroni` | `True` | Greedy: include the original feature count in the round-budget correction. Classic correction is controlled by `two_step`. |
| `importance_getter` | `"auto"` | Use `feature_importances_`, or coefficient magnitudes from `coef_` (L2 across classes). Also accepts a dotted attribute path or a callable returning one finite score per augmented feature. Larger scores mean greater importance. |
| `store_history` | `False` | Retain per-round real importances and shadow thresholds after fitting. |

The default classifier is configured as follows; remaining parameters follow
BankaiRF a15 defaults:

```python
from bankai_random_forest import BankaiRandomForestClassifier

forest = BankaiRandomForestClassifier(
    extra_trees=True,
    extra_trees_backend="binned",
    max_bins=63,
    importance_type="gain",
    max_depth=6,
    n_jobs=-1,
)
```

Pass a classifier for full forest customization, then use nested parameters
such as `estimator__max_depth` with `set_params` or parameter search. Weights and
original training labels, including text labels, are passed to the Python
adapter without re-encoding the labels supplied to the classifier.

```python
forest = BankaiRandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    n_jobs=-1,
)
selector = BankaiBoruta(forest, n_estimators=None, random_state=42)
selector.set_params(estimator__max_depth=7)
selector.fit(X_train, y_train)
```

### Native forest controls for benchmarks

The validated binned Extra Trees gain configuration honors the supplied
`BankaiRandomForestClassifier.n_jobs` and `max_features` on the native path.
For example, this configuration keeps virtual shadows with four workers and
fractional feature sampling:

```python
forest = BankaiRandomForestClassifier(
    n_estimators=150, max_depth=8, max_bins=63,
    extra_trees=True, extra_trees_backend="binned", importance_type="gain",
    n_jobs=4, max_features=0.15,
)
selector = BankaiBoruta(forest, n_estimators=150, random_state=42)
selector.fit(X_train, y_train)
assert selector.diagnostics_["path"] == "rust"
```

Both controls use the pinned a15 forest's validation and resolution rules:
`n_jobs=None` means one worker, positive values specify the worker count, and
negative values resolve relative to the CPU count. `max_features` accepts
`"sqrt"`, `"log2"`, a positive integer, a fraction in `(0, 1]`, or `None`.
Feature counts resolve against each round's current real-plus-shadow width,
including after Classic rejects columns, and fractional counts are floored
with a minimum of one. Integer counts must fit that round's width. The resolved
worker count controls native preparation and forest training; tree parallelism
is capped by the number of trees.

Other forest settings must still meet native eligibility. To benchmark the
materialized Python adapter with the same forest, set
`importance_getter="feature_importances_"` and assert
`selector.diagnostics_["path"] == "python"`. Native virtual shadows avoid the
augmented dense matrix; input conversion to float64 can still allocate.

`fit(X, y, sample_weight=None)` accepts non-negative sample weights with positive
total weight. Classic selection counts hits and uses binomial tests; its final
median comparison marks surviving tentative features as weak. The fitted
selector exposes:

| Attribute or method | Meaning |
| --- | --- |
| `support_`, `get_support(indices=False)` | Confirmed-feature mask, or column indices with `indices=True`. |
| `support_weak_` | Tentative features retained by the classic median rough fix; always false in greedy mode. |
| `ranking_` | Classic: confirmed rank 1, weak rank 2, other features ordered by median importance ranks. Other ranks start at 2 when there are no weak features, or 3 otherwise. Greedy: rank 1 or 2. |
| `n_features_`, `n_features_in_` | Confirmed-feature count and original feature count. |
| `feature_names_in_`, `get_feature_names_out()` | Original string column names when available, and selected output names. |
| `n_iter_`, `max_iter_` | Completed rounds and effective iteration bound. Constructor parameters remain unchanged during fit. |
| `importance_history_`, `shadow_threshold_history_` | Available with `store_history=True`. Importance history starts with a zero row followed by one row per round. |
| `diagnostics_` | Execution path, fallback reason, timing, augmented float-copy bytes and native storage-rebuild count. |

The selector supports scikit-learn pipelines and cloning. A fresh clone is used
for each round when a supplied classifier enables `warm_start`, preventing trees
from retaining previous shadows. `estimator_` contains cloned configuration on
the Rust path and the last round's fitted classifier on the adapter path.

### BankaiRF backend and performance foundations

The backend combines several sources of computational efficiency:

| Foundation | Role in Bankai Boruta |
| --- | --- |
| [Greedy Boruta](https://github.com/Nicolas-Vana/GreedyBorutaPy) | Optional fixed-round selection confirms a feature after its first hit, replacing repeated binomial confirmation tests with a bounded campaign. |
| [Extra Trees](https://doi.org/10.1007/s10994-006-6226-1) | Random cut points reduce exhaustive threshold search in the forest. BankaiRF's default here combines Extra Trees with bootstrap sampling and per-tree gain normalization. |
| [LightGBM histogram methods](https://lightgbm.readthedocs.io/en/stable/Features.html#optimization-in-speed-and-memory-usage) | Discretization replaces continuous values with compact bins and supports per-bin statistics. BankaiRF a15's binned Extra Trees stores bins by column and uses a single statistics pass for binary splits; this is its own forest backend. |
| [BankaiRF a15 Rust engine](https://github.com/breno-jesus-fernandes/bankai-random-forest/tree/caa46c22ef93434cd87319490513e877d4378564) | Parallel trees and virtual shadow access avoid constructing an expanded floating-point input matrix on the validated native path. Rounds remain sequential. |

The scientific papers and author article are linked under [References](#references).
The LightGBM reference describes binning and histogram principles; the a15
implementation determines the actual forest storage and split behavior.
The default binned Extra Trees path uses its own column-bin statistics rather
than the global histogram caches of BankaiRF's ordinary Random Forest path.

Rust execution is selected automatically for the validated default binned Extra
Trees gain configuration, with supported changes to tree count, random seed,
depth, bins (2–255), and Extra Trees seed. Other settings, weights, monotonic
constraints and custom getters use the Python adapter, preserving estimator
parameters and a15 validation. Shadow monotonic constraints are always zero.
`diagnostics_` records the path and its reason.

Both Python and Rust dependencies are aligned to BankaiRF **0.1.0a15**; the Rust
revision is `caa46c22ef93434cd87319490513e877d4378564`. Training storage is rebuilt
each round as required by a15. Immutable inputs and permutation buffers are
reused within a fit; training caches are not shared across independent fits.
Copy metrics cover augmented float-matrix writes, while a15's internal bin and
cache allocations remain part of preparation/training time and measured RSS.

See the [reproducible validation report](docs/validation.md) for Rust/adapter
importance parity, reference-algorithm comparisons, timing, peak memory, recall
and false positives. The measurements describe specific datasets and seeds;
actual speed and selection quality depend on the workload and configuration.

### Migration from a2

`estimator` is now the first argument. Configure forest parameters on that
classifier instead of the removed selector arguments `forest_backend`,
`pruning`, `importance`, `importance_scale`, `input_mode`, `max_features`,
`max_depth`, `n_jobs`, and `max_bins`. `greedy=True` enables hit-once selection;
the former aggressive pruning rule is a different algorithm. The `bankai-rf`,
`polars`, and `arrow-test` extras were removed. This release supports NumPy and
pandas; Polars and Arrow integration is outside its scope.

## Greedy mode

```python
selector = BankaiBoruta(
    greedy=True,
    bonferroni=True,
    alpha=0.05,
    random_state=2077,
)
X_selected = selector.fit_transform(X_train, y_train)
print(f"Rounds: {selector.n_iter_}; effective budget: {selector.max_iter_}")
```

Greedy retains every original feature throughout the campaign. With `p` original
features, it runs `K = ceil(log2(p / alpha))` rounds when `bonferroni=True`, or
`K = ceil(log2(1 / alpha))` otherwise. The exact floating-point ratio is used to
compute the ceiling safely at power-of-two boundaries and very small alpha.

A feature is confirmed after at least one importance above the shadow
percentile. Features with no hits are rejected at completion. `support_weak_`
is always false and `ranking_` uses 1 for confirmed and 2 for rejected features.
`max_iter`, `two_step`, `early_stopping`, and `n_iter_no_change` do not control
this mode. Greedy relaxes confirmation and can select more noise; the validation
report measures that tradeoff without assuming a universal recall advantage.

## Development

Local builds need Git, Python 3.11+, stable Rust, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Install Xcode command-line
tools on macOS or the Rust MSVC build tools on Windows.

```bash
git clone https://github.com/breno-jesus-fernandes/bankai-boruta.git
cd bankai-boruta
git switch feature/greedy-bankai-a15
uv python install 3.11
uv sync --locked --no-install-project --extra test
uv run --no-sync maturin develop --release --locked
```

Run the Python and Rust checks, then reproduce the benchmark campaign as needed:

```bash
uv run --no-sync pytest -q
uv run --no-sync cargo test --workspace --locked
uv run --no-sync python benchmarks/validate_a3.py
uv run --no-sync python benchmarks/summarize_a3.py
```

Both lockfiles are committed. CI tests Python 3.11–3.14 and builds ABI3 wheels
for Linux x86_64/ARM64, macOS x86_64/ARM64 and Windows x86_64, with installed-wheel
smoke tests. The scheduled BankaiRF release check reports upstream versions;
upgrades must align both dependency pins and pass compatibility/performance checks.

Read the [development guide](docs/development.md),
[contribution guide](CONTRIBUTING.md), and [Code of Conduct](CODE_OF_CONDUCT.md).

## References

1. Nicolas Vana (2025), [The Greedy Boruta Algorithm: Faster Feature Selection Without Sacrificing Recall](https://towardsdatascience.com/the-greedy-boruta-algorithm-faster-feature-selection-without-sacrificing-recall/), author article in Towards Data Science. The [published GreedyBoruta 1.0.0 implementation](https://pypi.org/project/greedyboruta/1.0.0/) supplies the fixed-round, hit-once reference used here.
2. Pierre Geurts, Damien Ernst and Louis Wehenkel (2006), [Extremely randomized trees](https://doi.org/10.1007/s10994-006-6226-1), *Machine Learning* 63, 3–42. [Author preprint and publication record](https://orbi.uliege.be/handle/2268/9357).
3. Guolin Ke et al. (2017), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), *NeurIPS 30*, particularly Section 2.1 on histogram-based learning. [Paper PDF](https://papers.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) and [official histogram documentation](https://lightgbm.readthedocs.io/en/stable/Features.html#optimization-in-speed-and-memory-usage).
4. Miron B. Kursa and Witold R. Rudnicki (2010), [Feature Selection with the Boruta Package](https://www.jstatsoft.org/article/view/v036i11), *Journal of Statistical Software* 36(11), 1–13. [BorutaPy implementation](https://github.com/scikit-learn-contrib/boruta_py).

## License

Bankai Boruta is distributed under [GPL-3.0-or-later](COPYING). Its BankaiRF
Rust backend includes the maintained GPL-3.0-or-later
[XRF fork](https://github.com/breno-jesus-fernandes/bankai-random-forest/blob/v0.1.0a15/crates/xrf/UPSTREAM.md).
