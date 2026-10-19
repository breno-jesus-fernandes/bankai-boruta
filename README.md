# 🔥 Bankai Boruta

[![PyPI Latest Release](https://img.shields.io/pypi/v/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Package Status](https://img.shields.io/pypi/status/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![Python versions](https://img.shields.io/pypi/pyversions/bankai-boruta.svg)](https://pypi.org/project/bankai-boruta/)
[![uv managed](https://img.shields.io/badge/uv-managed-blue)](https://docs.astral.sh/uv/)
[![Downloads](https://static.pepy.tech/badge/bankai-boruta)](https://pepy.tech/project/bankai-boruta)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/COPYING)
[![CI](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/ci.yml)
[![Dependency security](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/breno-jesus-fernandes/bankai-boruta/branch/main/graph/badge.svg)](https://codecov.io/gh/breno-jesus-fernandes/bankai-boruta)
[![GitHub Release](https://img.shields.io/github/v/release/breno-jesus-fernandes/bankai-boruta?include_prereleases)](https://github.com/breno-jesus-fernandes/bankai-boruta/releases)
[![PyPI deployment](https://img.shields.io/github/deployments/breno-jesus-fernandes/bankai-boruta/pypi?label=PyPI%20deployment)](https://github.com/breno-jesus-fernandes/bankai-boruta/deployments)
[![GitHub stars](https://img.shields.io/github/stars/breno-jesus-fernandes/bankai-boruta.svg?style=social)](https://github.com/breno-jesus-fernandes/bankai-boruta)

BankaiBoruta is a fast, memory-efficient implementation of the Boruta
all-relevant feature selection algorithm for large tabular datasets.

Boruta rounds run inside the native Rust engine of
[Bankai Random Forest](https://github.com/breno-jesus-fernandes/bankai-random-forest):
real columns are prepared once per fit, shadow columns are shuffled in parallel
and trees train without building an augmented real-plus-shadow matrix. The
decision rule is the same as BorutaPy's, behind a scikit-learn-compatible
selector. Classic Boruta is the default; `greedy=True` enables the fixed-round
Greedy Boruta variant.

⚠️ Bankai Boruta is in Alpha. Its API and fitted objects may change.

## Install

Python 3.11 or newer is required:

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
fit and transform steps. NumPy arrays and pandas DataFrames are supported.

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
| `estimator` | `None` | A clonable classifier. `None` uses the default BankaiRF forest shown below. The supplied estimator is not modified. |
| `n_estimators` | `"auto"` | BorutaPy's feature/depth heuristic for tree count. A positive integer overrides the count; `None` keeps the estimator's setting and supports classifiers without a tree-count parameter. |
| `perc` | `100` | Shadow-importance percentile in `(0, 100]`. A real feature scores a hit only when its importance is strictly above this threshold. |
| `alpha` | `0.05` | Significance level in `(0, 1)` for classic tests and the greedy round budget. |
| `two_step` | `True` | Classic: combine Benjamini–Hochberg FDR across active features with Bonferroni across rounds. `False` uses Bonferroni across the original features. Ignored in greedy mode. |
| `max_iter` | `100` | Classic iteration bound. As in BorutaPy 0.4.3, the loop runs at most `max_iter - 1` rounds. Ignored in greedy mode. |
| `random_state` | `None` | Integer seed, NumPy `RandomState`, or `None`. Controls shadows and per-round forest seeds. |
| `verbose` | `0` | Non-negative verbosity level. Positive values print round progress and confirmed-feature counts. |
| `early_stopping` | `False` | Classic: stop when the decisions remain unchanged beyond the patience limit. Ignored in greedy mode. |
| `n_iter_no_change` | `20` | Positive patience limit for classic early stopping. |
| `greedy` | `False` | Enable fixed-round, hit-once selection instead of classic binomial decisions. |
| `bonferroni` | `True` | Greedy: include the original feature count in the round-budget correction. Classic correction is controlled by `two_step`. |
| `importance_getter` | `"auto"` | Use `feature_importances_`, or coefficient magnitudes from `coef_` (L2 across classes). Also accepts a dotted attribute path or a callable returning one finite score per augmented feature. Larger scores mean greater importance. |
| `store_history` | `False` | Keep per-round real importances and shadow thresholds after fitting. |

`fit(X, y, sample_weight=None)` accepts non-negative sample weights with a
positive total. The fitted selector exposes:

| Attribute or method | Meaning |
| --- | --- |
| `support_`, `get_support(indices=False)` | Confirmed-feature mask, or column indices with `indices=True`. |
| `support_weak_` | Tentative features kept by the classic median rough fix; always false in greedy mode. |
| `ranking_` | Classic: confirmed rank 1, weak rank 2, other features ordered by median importance ranks. Greedy: rank 1 or 2. |
| `n_features_`, `n_features_in_` | Confirmed-feature count and original feature count. |
| `feature_names_in_`, `get_feature_names_out()` | Original string column names when available, and selected output names. |
| `n_iter_`, `max_iter_` | Completed rounds and effective iteration bound. |
| `importance_history_`, `shadow_threshold_history_` | Available with `store_history=True`. Importance history starts with a zero row followed by one row per round. |
| `diagnostics_` | Execution path and its reason, storage mode, timings and buffer sizes (see below). |

The selector supports scikit-learn pipelines, cloning and parameter search.

## Forests

The default forest is binned Extra Trees with 63 bins:

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

Pass any `BankaiRandomForestClassifier` to choose another backend. Nested
parameters such as `estimator__max_depth` work with `set_params` and parameter
search:

```python
forest = BankaiRandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
selector = BankaiBoruta(forest, n_estimators=None, random_state=42)
selector.set_params(estimator__max_depth=7)
selector.fit(X_train, y_train)
```

All four BankaiRF backends run natively:

| Backend | Forest settings | Native input |
| --- | --- | --- |
| Exact Extra Trees | `extra_trees=True`, `extra_trees_backend="exact"`, `max_bins=None` | float32 |
| Binned Extra Trees | `extra_trees=True`, `extra_trees_backend="binned"`, `max_bins` in 2–255 | float32 or float64 |
| Exact Random Forest | `extra_trees=False`, `max_bins=None` | float32 |
| Binned Random Forest | `extra_trees=False`, `max_bins` in 2–255 | float32 or float64 |

On the native path, one shadow session serves the whole fit. Real columns are
stored once, as float32 values for exact forests or as one-byte bin codes for
binned forests, and every round shuffles shadow columns in parallel inside that
session. Float64 binned input is binned again each round so its edges come from
the original float64 values. Exact forests with float64 input use the Python
adapter, which preserves BankaiRF's float64 split semantics; convert to float32
to use the native path.

The native path forwards `n_estimators`, `max_depth`, `max_features`, `n_jobs`,
`random_state`, `extra_seed`, `criterion`, `min_samples_split`,
`min_samples_leaf`, `min_weight_fraction_leaf`, `min_impurity_decrease`,
`max_leaf_nodes`, `bootstrap`, `max_samples`, `max_bins`, `binning_strategy`,
`bin_sample_size` and `binning_seed`. `max_features` is resolved against each
round's current real-plus-shadow width. Gain importance is normalized per tree
for Extra Trees and over the forest for Random Forests.

Other configurations use the Python adapter, which fits the estimator on a
materialized real-plus-shadow matrix each round. That includes sample or class
weights, other importance types, `warm_start`, OOB scoring, monotonic
constraints, custom importance getters and non-BankaiRF classifiers. Both paths
apply the same decision rule.

```python
selector = BankaiBoruta(forest, n_estimators=100, random_state=42)
selector.fit(X_train.astype("float32"), y_train)
print(selector.diagnostics_["path"], selector.diagnostics_["reason"])
print(selector.diagnostics_["storage_mode"])  # "shadow_session" on the native path
```

`diagnostics_` reports the path (`"rust"` or `"python"`) and the reason for it,
the storage mode, preparation, training and shuffle times, and explicit buffer
sizes such as `real_bank_bytes`, `permutation_bytes`, `input_conversion_bytes`
and `copied_bytes` (augmented-matrix writes on the adapter path). These are
buffer sizes, not RSS.

## Performance

Medians over five seeds on an AWS c7a.4xlarge (16 AMD cores, 16 threads),
against BorutaPy 0.4.3 with scikit-learn `ExtraTreesClassifier` and the same
forest settings. Every method ran up to 40 rounds with the same decision rule.
RSS is the fit's peak increase over the loaded data.

| Profile (train n × p) | Bankai exact | Bankai binned | BorutaPy + ExtraTrees |
| --- | ---: | ---: | ---: |
| Industrial 200k × 500 | 22.5 s / 1,283 MiB | 15.6 s / 667 MiB | 37.5 s / 1,817 MiB |
| Credit 300k × 300 | 20.8 s / 1,474 MiB | 16.1 s / 819 MiB | 45.7 s / 1,806 MiB |
| Cybersecurity 400k × 150 | 54.8 s / 1,395 MiB | 42.7 s / 908 MiB | 107.5 s / 1,472 MiB |
| Bioinformatics 50k × 2000 | 135.6 s / 982 MiB | 64.0 s / 450 MiB | 145.1 s / 1,608 MiB |

Both Bankai backends were faster and used less memory than BorutaPy in all 20
profile × seed pairs. No method confirmed an irrelevant feature. In Industrial,
Credit and Bioinformatics, Bankai selected the same sets as BorutaPy. In
Cybersecurity, every method reached the round limit and borderline features
varied: median recall was 0.67 for Bankai exact, 0.65 for BorutaPy and 0.62
for Bankai binned. These are synthetic datasets; results depend on the workload.
Protocol, raw results and the LightGBM comparison are in the
[benchmark repository](https://github.com/breno-jesus-fernandes/bankai-boruta-benchmark/tree/main/verification/native-shadow-amd-20261019).

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

Greedy keeps every original feature throughout the campaign. With `p` original
features, it runs `K = ceil(log2(p / alpha))` rounds when `bonferroni=True`, or
`K = ceil(log2(1 / alpha))` otherwise.

A feature is confirmed after at least one importance above the shadow
percentile. Features with no hits are rejected at completion. `support_weak_`
is always false and `ranking_` uses 1 for confirmed and 2 for rejected features.
`max_iter`, `two_step`, `early_stopping`, and `n_iter_no_change` do not control
this mode. Greedy relaxes confirmation and can select more noise.

## Development

Local builds need Git, Python 3.11+, stable Rust, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Install the Xcode
command-line tools on macOS or the Rust MSVC build tools on Windows.

```bash
git clone https://github.com/breno-jesus-fernandes/bankai-boruta.git
cd bankai-boruta
uv python install 3.11
uv sync --locked --no-install-project --extra test
uv run --no-sync maturin develop --release --locked
```

Run the Python and Rust tests:

```bash
uv run --no-sync pytest -q
uv run --no-sync cargo test --workspace --locked
```

CI tests Python 3.11–3.14 and builds ABI3 wheels for Linux x86_64/ARM64, macOS
x86_64/ARM64 and Windows x86_64. Merging a new version to `main` publishes it
to PyPI. See the [development guide](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/docs/development.md) for the BankaiRF pin
and release order, the [changelog](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/CHANGELOG.md) for release history, the
[contribution guide](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/CODE_OF_CONDUCT.md).

## References

1. Nicolas Vana (2025), [The Greedy Boruta Algorithm: Faster Feature Selection Without Sacrificing Recall](https://towardsdatascience.com/the-greedy-boruta-algorithm-faster-feature-selection-without-sacrificing-recall/), author article in Towards Data Science. The [published GreedyBoruta 1.0.0 implementation](https://pypi.org/project/greedyboruta/1.0.0/) supplies the fixed-round, hit-once reference used here.
2. Pierre Geurts, Damien Ernst and Louis Wehenkel (2006), [Extremely randomized trees](https://doi.org/10.1007/s10994-006-6226-1), *Machine Learning* 63, 3–42. [Author preprint and publication record](https://orbi.uliege.be/handle/2268/9357).
3. Guolin Ke et al. (2017), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), *NeurIPS 30*, particularly Section 2.1 on histogram-based learning. [Paper PDF](https://papers.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) and [official histogram documentation](https://lightgbm.readthedocs.io/en/stable/Features.html#optimization-in-speed-and-memory-usage).
4. Miron B. Kursa and Witold R. Rudnicki (2010), [Feature Selection with the Boruta Package](https://www.jstatsoft.org/article/view/v036i11), *Journal of Statistical Software* 36(11), 1–13. [BorutaPy implementation](https://github.com/scikit-learn-contrib/boruta_py).

## License

Bankai Boruta is distributed under [GPL-3.0-or-later](https://github.com/breno-jesus-fernandes/bankai-boruta/blob/main/COPYING). Its BankaiRF
Rust backend includes the maintained GPL-3.0-or-later
[XRF fork](https://github.com/breno-jesus-fernandes/bankai-random-forest/blob/master/crates/xrf/UPSTREAM.md).
