# a3 validation report — 2026-10-15

This campaign compares Bankai Boruta's Rust path and Python adapter with the same
BankaiRF **0.1.0a15** classifier, and compares selection rules with **BorutaPy
0.4.3** and **GreedyBoruta 1.0.0**. BankaiRF itself was not modified.

## Reproduction

```sh
uv sync --locked --no-install-project --extra test
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest -q
cargo test --workspace --locked
uv run --no-sync python benchmarks/validate_a3.py --output benchmarks/results/a3-final.json
uv run --no-sync python benchmarks/summarize_a3.py
```

On the local host, uv 0.5.20 could not support maturin's `uv pip install --group`
invocation. We instead built a release wheel with `maturin build --release
--locked`, installed it with `uv pip install --no-deps`, ran tests against that
installed wheel, and ran `python -I scripts/smoke_installed.py`. CI uses its
existing newer uv pin. Both wheel and source distribution built successfully.

The local suite contains 60 Python cases and one Rust integration test. It
covers controlled constant and changing importance sequences, both statistical
corrections, percentile changes, weak features and ranking, early stopping,
strict tie handling, invalid importances, original string labels, multiclasse,
weights, monotonic constraints with zero-constrained shadows, nested parameters,
coefficient importances, pretrained estimator preservation, warm-start isolation, a15 parameter-type/error semantics,
refits, zero selected
features, few features, and exact/subnormal greedy round boundaries. Native
parity also varies seeds, bins, depth, tree count and discrete/signed-zero input.

## Method

For each of 200 × 10 and 1000 × 30 inputs, use seeds 1, 7 and 42. The target
contains three signal variables; a fourth variable is correlated with the first.
All remaining variables are independent noise. Recall counts all four relevant
variables; noise FPR counts only independent noise columns. Every method uses
BankaiRF with 20 binned Extra Trees, depth 6, 63 bins and all available cores.
Classic campaigns have max_iter=20; greedy campaigns derive K from alpha=.05.
Each measurement runs in a fresh subprocess, excluding imports from fit timing.
The parent samples process RSS every 2 ms, including while Rust holds the child
GIL. RSS includes Python and imports, and sampling can miss short peaks.

Rust and adapter receive identical permutations and derived forest seeds.
The native accessor reproduces a15's canonical row order and its second seed
derivation. Comparisons require absolute importance error <=1e-12 and equal
selection decisions outside numerical ties. The summary script checks every
recorded paired history and selection mask. Reference-library tests use
controlled importances to separate algorithm rules from their differing random
number consumption. Reference timing campaigns use their normal RNG policies;
seed equality alone does not give identical forests/shadows across libraries.

Stage timing has different boundaries: native preparation includes canonical
sorting and DenseInput rebuilding; adapter preparation measures permutations
and float augmentation, while adapter training includes BankaiRF validation,
sorting, storage construction, fitting and importance extraction. Therefore
compare total fit time for speed, rather than equating individual stage labels.
Copy bytes are cumulative augmented float bytes written, excluding input
conversion, permutation arrays and a15's internal storage/cache allocations.
Native copy time zero means no augmented float matrix, not zero internal copies.

## Recorded environment

- Host: `macOS-26.6.2-arm64-arm-64bit`; logical CPUs: 8.
- Python: `3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]`.
- Versions: `bankai-boruta==0.1.0a3`, `bankai-random-forest==0.1.0a15`, `Boruta==0.4.3`, `greedyboruta==1.0.0`, `numpy==2.4.6`, `scikit-learn==1.9.1`.

## Final campaign

Maximum absolute Rust/adapter importance difference: 1.39e-16; every paired decision agrees.

| Rows × columns | Mode | Engine | Median ms | Median peak RSS MiB | Mean recall | Mean noise FPR |
|---|---|---|---:|---:|---:|---:|
| 200 × 10 | classic | rust | 17.48 | 157.92 | 0.917 | 0.000 |
| 200 × 10 | classic | adapter | 21.58 | 158.33 | 0.917 | 0.000 |
| 200 × 10 | classic | borutapy | 19.78 | 158.55 | 0.917 | 0.000 |
| 200 × 10 | greedy | rust | 7.69 | 157.95 | 1.000 | 0.611 |
| 200 × 10 | greedy | adapter | 9.76 | 158.12 | 1.000 | 0.611 |
| 200 × 10 | greedy | greedyboruta | 10.12 | 157.84 | 0.917 | 0.278 |
| 1000 × 30 | classic | rust | 36.52 | 158.50 | 1.000 | 0.000 |
| 1000 × 30 | classic | adapter | 41.14 | 159.78 | 1.000 | 0.000 |
| 1000 × 30 | classic | borutapy | 40.26 | 161.27 | 1.000 | 0.000 |
| 1000 × 30 | greedy | rust | 23.47 | 158.31 | 1.000 | 0.256 |
| 1000 × 30 | greedy | adapter | 27.00 | 159.31 | 1.000 | 0.256 |
| 1000 × 30 | greedy | greedyboruta | 32.75 | 158.64 | 1.000 | 0.231 |

| Rows × columns | Mode | Engine | Preparation ms | Training ms | Augmentation copy ms | Augmented float bytes |
|---|---|---|---:|---:|---:|---:|
| 200 × 10 | classic | rust | 3.67 | 5.35 | 0.00 | 0 |
| 200 × 10 | classic | adapter | 0.95 | 14.62 | 0.48 | 492800 |
| 200 × 10 | greedy | rust | 1.71 | 2.47 | 0.00 | 0 |
| 200 × 10 | greedy | adapter | 0.49 | 6.67 | 0.25 | 256000 |
| 1000 × 30 | classic | rust | 14.78 | 13.18 | 0.00 | 0 |
| 1000 × 30 | classic | adapter | 6.01 | 28.95 | 2.76 | 5024000 |
| 1000 × 30 | greedy | rust | 11.13 | 8.23 | 0.00 | 0 |
| 1000 × 30 | greedy | adapter | 5.52 | 18.71 | 2.50 | 4800000 |

## Hypotheses and limitations

- **a15 gain/decision parity:** supported on the tested subset. Every paired
  history is within tolerance and every decision agrees. Native selection is
  enabled only for that conservative subset. Weights, constraints, alternative
  growth settings and importance getters retain Python adapter behavior.
- **Bitwise reproduction under tree parallelism:** refuted. Repeated fits can
  differ around 1e-16 because accumulation order varies. Tests use a 1e-12
  tolerance for importance values and check decision reproducibility separately.
  Numerical ties can change a strict greater-than comparison.
- **Universal speedup or historical multi-fold acceleration:** unsupported.
  Final medians favor Rust modestly here, but the exploratory campaign recorded
  an outlier where Rust classic took 74 ms and adapter 47 ms. Three seeds on two
  small sizes do not establish production scaling or a universal speed ratio.
- **Greedy guarantees higher recall with unchanged false positives:** unsupported.
  Greedy recall equals or exceeds classic in this small fixture, while selecting
  substantially more independent noise. This is evidence of the tradeoff, not a
  general recall guarantee. Published GreedyBoruta selects different noise
  columns under its own RNG policy, even though controlled rule tests agree.
- **Virtual shadows guarantee a large RSS reduction:** unsupported at these
  sizes. They eliminate the expanded float matrix, but RSS is dominated by
  imports and other allocations; the measured differences are small. Larger
  memory campaigns remain necessary before making a memory scaling claim.
- **a15 allows reusing bins across rounds:** not available through its current
  public training API. DenseInput is rebuilt every native round; the diagnostic
  rebuild count equals n_iter_. At 1000 × 30, storage preparation is a material
  part of fit time. Immutable X/encoded labels and permutation buffers are reused
  within fit; bins and caches are never shared across independent fits.

The exploratory [a3.json](../benchmarks/results/a3.json) preserves the first
campaign, including the timing outlier. Its RSS was sampled inside the worker
and may miss native peaks, and its early preparation metric omitted boundary
sorting. Use [a3-final.json](../benchmarks/results/a3-final.json) for the tables
above. Both are observations, not portable performance guarantees.

The Rust source is pinned to a15 commit
`caa46c22ef93434cd87319490513e877d4378564`. Python requires exactly a15.
The scheduled release checker reports newer prereleases and stable releases
without changing either pin. Future upgrades require renewed parity/performance
validation.

## Wheel/platform verification

The macOS ARM64 CPython 3.11 ABI3 wheel was built, installed and smoke-tested
locally. A clean environment automatically installed the pinned BankaiRF
dependency and passed the same wheel smoke on Python 3.14.7. A source distribution
was also built and successfully rebuilt into a wheel from its extracted source. BankaiRF a15 publishes wheels
for all five platform families in the existing CI matrix. CI now runs the
installed-wheel smoke on each platform in addition to Python 3.11–3.14 and Rust
tests. The [final CI run](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/runs/38834117718)
passed on code commit `d2c2684025be07440e59cd9ae0ef50449b8889ac`: all eight
Python/Rust test configurations, all five platform wheel builds and installed-wheel
smokes, source distribution and SBOM collection succeeded. Attestation,
GitHub release creation and PyPI publishing were skipped. Subsequent README
edits preserve the same tested code; their Python examples were executed locally.
No merge to main or PyPI publication is part of this delivery.

## Reference sources

- [BorutaPy 0.4.3](https://pypi.org/project/Boruta/0.4.3/) — installed source used
  for classic decision, rough-fix and ranking comparisons.
- [GreedyBoruta 1.0.0](https://pypi.org/project/greedyboruta/1.0.0/) — published
  source used for fixed rounds, hit-once selection and rank 1/2 rules.
- [BankaiRF a15 source](https://github.com/breno-jesus-fernandes/bankai-random-forest/tree/caa46c22ef93434cd87319490513e877d4378564)
  — DenseInput, forest seeds, training order and per-tree gain normalization.
