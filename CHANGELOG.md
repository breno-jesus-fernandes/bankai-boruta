# Changelog

## 0.1.0a8 (unreleased)

- Make a shadow session the default for native exact Extra Trees. Each fit
  sorts rows once into a canonical float32 column bank; each round copies the
  shadow sources into one reusable buffer and Fisher-Yates shuffles the values
  in parallel, one xoshiro256** stream per shadow. No augmented matrix and no
  permutation maps are allocated, and training reads contiguous columns
  through BankaiRF 0.1.0a17's `ExactColumnTable`.
- Industrial 200k x 500 (Mac, 4 threads): a 500-feature round takes 2.3-2.9 s
  instead of 8.0 s (a7 materialized) or 7.1 s (a7 virtual session); five-seed
  full fits run 1.9-4.3x faster than BorutaPy with scikit-learn ExtraTrees at
  the same settings, with lower peak RSS and the same selections.
- Extend the shadow session to every native backend. Binned Extra Trees
  (including the default estimator) bin float32 real columns once per fit and
  shuffle their bin codes for shadows: a 500-feature round drops from 7.7 s to
  2.0 s and the bank is one byte per value. Ordinary exact Random Forests
  (`extra_trees=False`, `max_bins=None`) now take the native path too, with
  BankaiRF's forest-normalized gain. Float64 binned input keeps per-round
  binning; binned Random Forests keep the Python fit.
- Sum per-tree gain shares in fixed point, so importances are identical for
  any worker count and tree completion order.
- Shadow values come from a new stream, so selections for a given
  `random_state` differ from a7 while the statistics are unchanged (60-seed
  round counts: 32.1 +/- 1.4 versus 35.1 +/- 1.2 for BorutaPy + ExtraTrees).
  `BANKAI_BORUTA_EXACT_REFERENCE=1` restores the a7 materialized path and
  `BANKAI_BORUTA_EXACT_SESSION=1` the a7 virtual session.
- Require BankaiRF 0.1.0a17.

## 0.1.0a7

- Add an experimental private per-fit exact session with BankaiRF a16 shared
  real columns and explicit virtual shadow permutations.
- Cache stable real-feature ordering and refine only shadow/label ties; invalidate
  on active-list changes without altering RandomState draws or statistics.
- Preserve a6 public float64 adapter behavior and all binned/default paths.
- Add separate timing/live-buffer/copy counters, verbose round traces, exact
  parity tests and sequential external-process RSS/VmHWM comparisons.
- Keep the materialized path as default pending AMD acceptance.
- Pin Rust and development Python sources to an immutable BankaiRF revision;
  published dependency requires RF a16 to be released first.
- Require explicit `publish=true` dispatch for PyPI upload.
