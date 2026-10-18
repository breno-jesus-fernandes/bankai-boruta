# Experimental exact ExtraTrees session (a7)

The public selector API and default behavior remain compatible with a6. To
exercise the private candidate on native exact float32 fits, set
`BANKAI_BORUTA_EXACT_SESSION=1` before calling `fit`. The switch is experimental
and is not a new estimator parameter. Default/binned/external paths retain
existing behavior. Float64 exact public fits retain the a6 float64 adapter:
converting those fits to float32 changes splits on near-equal values. The private
session accepts f32/f64 and preserves original precision for canonical sorting;
its training contract matches the native materialized f32 reference, not the
float64 external adapter. Internal tests cover both original dtypes.

Each `fit` owns one session, released when fit returns. It snapshots immutable
real columns once in original precision. Trees share the bank. Each round keeps
explicit independently shuffled permutations and the exact legacy RandomState
sequence, including seed draws. Active real columns define the primary stable
row order; only groups tied across all active real features are stably refined
by shadows and labels. Complete ties keep original row order. Changing the
ordered active list invalidates the primary order. Round order/permutation
capacities are reused; no augmented real-plus-shadow matrix is built.

Standard MT19937/u32 fits generate the exact independent Fisher-Yates
permutations directly in the session-owned reusable buffer, exchanging the same
legacy RNG state before the unchanged seed draw. This avoids a second NumPy
permutation matrix and its round copy; it does not replace permutations with an
approximation. Custom RandomState subclasses and int64 indices retain NumPy
shuffle and copy into a reusable native snapshot. That fallback avoids unsafe
lifetimes or reading mutable caller arrays while trees train. The original
input, real bank and explicit permutations still have a memory cost.

## Diagnostics

Existing fields retain their original scope:

| Field | Scope |
|---|---|
| `preparation_seconds` | Sum of shuffle and native preparation; includes session creation when enabled |
| `training_seconds` | Forest training and per-tree importance reduction |
| `copied_bytes`, `copy_seconds` | Python augmented-matrix copying only; zero never meant no internal allocation |
| `input_conversion_bytes` | Python input conversion volume |
| `permutation_bytes` | Maximum live NumPy permutation matrix plus row-index template; zero with owned shuffle |
| `training_storage_rebuilds` | Materialized native storage constructions; zero for the session |

New fields separate known costs:

| Field | Scope |
|---|---|
| `storage_mode` | `virtual_session` or `materialized` |
| `session_creation_seconds` | Real bank construction/validation and immutable label snapshot, once |
| `validation_seconds` | Round map/permutation copying and validation (including bijections) |
| `shuffle_seconds` | NumPy permutation preparation and legacy-equivalent shuffle |
| `sorting_seconds` | Primary order rebuild when needed plus tied-group refinement |
| `storage_seconds` | Native input construction/control validation, excluding round map validation |
| `real_bank_bytes` | Owned original-precision bank; live high water |
| `real_bank_copied_bytes` | One-time bank snapshot volume |
| `native_permutation_bytes` | Retained capacity of the owned Rust permutation buffer; live high water |
| `native_permutation_copied_bytes` | Cumulative bytes copied into that buffer across rounds; zero with owned shuffle |
| `order_cache_bytes` | Retained real/canonical order and tie-range capacities; live high water |
| `sort_cache_rebuilds` | Number of primary-order rebuilds |
| `materialized_bytes` | Known final augmented storage high water; zero for the session |
| `materialized_copied_bytes` | Cumulative known final materialization volume; zero for the session |
| `feature_workspace_peak_bytes_per_worker` | Maximum observed binary split-buffer capacity for one worker |

For the materialized binned path, `materialized_bytes` counts final u8 storage
only, excluding temporary conversion/layout buffers. These are partial buffer
counters, not total heap usage. They exclude tree arenas, masks, node statistics,
some transient row/label/weight buffers, allocator overhead and the caller's
original matrix. Do not sum cumulative-copy volumes with live capacities.
Use external peak RSS and Linux VmHWM for acceptance. The per-worker maximum
cannot be treated as a sum of simultaneous workspaces.

`verbose>=2` adds `diagnostics_['rounds']` with active width, selector seed,
duration, hits and decisions. Only the last round gets the final `stop_reason`;
previous rounds use null. This opt-in trace consumes memory and should be
disabled during timed campaigns. Existing `store_history` controls importances.

## Reproduction and acceptance

`benchmarks/compare_exact_session.py` runs candidates sequentially in fresh
processes and refuses to overwrite an experiment output. The reference interpreter
must contain exactly Boruta a6 / RF a15. It compares data hashes, hits, decisions,
confirmed/tentative features, ranking, rounds and stop reasons exactly; importances
and thresholds use atol=rtol=1e-12. RNG-state and full-tree tests complement fit
comparisons. It measures fit-only RSS externally every 5 ms with a ready/go/done
handshake, attempts Linux VmHWM reset, and explicitly records scope on failure.

```sh
uv venv .venv-reference --python 3.13
uv pip install --python .venv-reference/bin/python bankai-boruta==0.1.0a6 bankai-random-forest==0.1.0a15
uv run --no-sync python benchmarks/compare_exact_session.py --reference-python "$PWD/.venv-reference/bin/python" --candidate-python "$PWD/.venv/bin/python" --output /tmp/a7-fixed-t1.json
uv run --no-sync python benchmarks/compare_exact_session.py --reference-python "$PWD/.venv-reference/bin/python" --candidate-python "$PWD/.venv/bin/python" --rows 2000 --columns 40 --trees 10 --fixed-rounds 0 --output /tmp/a7-complete-t1.json
```

Fixed rounds and complete fits have separate identities and outputs. Run without
concurrent compilers/tests or other benchmark candidates. On AMD, repeat with
`--threads 1`, `4`, `16` and `--seed 42` through `46`. Full Industrial uses
`--benchmark-root /path/to/frozen/bankai-boruta-benchmark --trees 100`, and complete
fits additionally use `--fixed-rounds 0 --max-iter 120`. The frozen benchmark
supplies its existing full generator. Keep repeats (default 3) and new output
paths. Report dispersion and individual observations, not only medians.

The session remains opt-in until it passes parity and reduces peak RSS and median
time in both controlled and full Industrial AMD trials. See
[local evidence](exact-session-validation.md). Hosted Linux CI evidence complements
local tests; it does not substitute for the dedicated AMD campaign. Historical
benchmark a6 artifacts must remain byte-identical.
