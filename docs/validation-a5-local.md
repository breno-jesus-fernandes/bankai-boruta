# a5: local memory validation

This candidate starts from `v0.1.0a4` (`b3f8ee6`) on branch
`feature/a5-memory`. BankaiRF remains pinned to Python `0.1.0a15` and Rust
revision `caa46c22ef93434cd87319490513e877d4378564`. No statistical, forest,
binning, canonical ordering, materialized-bin or public parameter changes are
included. Historical a4 results and article tables are preserved.

## Correctness

The isolated a4 and candidate Python 3.11.11 environments have identical runtime
dependencies. The independent comparison passed **200 fits per version / 1,400
arrays**: exact support, weak support, ranking, round count and stop reason;
importance and shadow threshold histories at `rtol=0`, `atol=1e-14`, with NaNs
preserved. See [machine-readable results](../benchmarks/results/a5-equivalence.json).

Coverage includes classic and greedy, float32 and float64, C and Fortran layouts
and strided/reversed views, binary and multiclass labels, ties, constants, signed
zero, fewer than five features, actual active-feature shrinkage, refits, weights,
constraints, custom getters, external estimators and warm-start fallback.

Validation date: 2026-10-18. The unchanged a4 suite passes 120 tests in its
reference environment. The candidate suite passes 252 tests, including the inherited a4 suite and native /
adapter parity. New checks cover identical uint32/int64 shuffles and final RNG
state, dtype selection around 2**32 rows without allocating those matrices,
platform addressing limits, all four Rust input specializations and invalid
shapes / row indices / feature indices / nonfinite values. Rust tests and Clippy
with warnings denied also pass. Installed-wheel smoke covers both float dtypes.

The standard RandomState MT19937 shuffle uses a Rust implementation of the same
legacy descending Fisher-Yates loop and masked rejection draws. It reads and
restores that RandomState's full state, retaining Gaussian cache and interleaved
shuffle/forest-seed consumption. This specializes uint32 swaps without adding an
int64 scratch vector. Exact comparisons cover four seeds, lengths 0 through
100,000, MT twist boundaries, rejection draws, cached Gaussian draws and
interleaved randint calls. RandomState subclasses, alternative bit generators
and int64 permutations retain NumPy shuffle. Per-column ndarray views resolve
strides once before the Rust accessor; no input or bin matrix is added.

This specialization was necessary for the time gate: NumPy's legacy shuffle
[specializes pointer-sized swaps](https://github.com/numpy/numpy/blob/v2.4.6/numpy/random/mtrand.pyx#L4219-L4226)
and uses a generic memcpy path for uint32. A local 1,500-shuffle microbenchmark
of 100,000 indices took 2.90 s for uint32 versus 1.52 s for int64. Before the
specialization, the candidate's industrial float32 median was 11.9% slower with
eight threads. The final campaign below uses the compatible Rust shuffle.

## Local measurement protocol

Hardware: Apple M1, 8 logical CPUs, 8 GiB RAM, macOS arm64.
Build tools: rustc 1.97.1, Cargo 1.97.1, maturin 1.15.0; release
optimization, fat LTO, one codegen unit (the unchanged a4 profile). Exact OS, Python and
package versions accompany every raw measurement. No remote benchmark hosts are
used.

The industrial input is 100,000 × 500. A seeded RandomState generates normal
values, rounded to float32; float64 runs promote those same values. Chunked data
generation avoids a whole-matrix float64 temporary before the initial RSS
sample. Target depends on the first three columns plus seeded noise.

Each engine uses 20 trees, depth 6, sqrt feature sampling, classic Boruta
`max_iter=4` (**three completed rounds**, not a full convergence campaign).
BankaiRF uses binned Extra Trees, 63 bins and its unchanged a15 defaults;
BorutaPy 0.4.3 uses sklearn ExtraTreesClassifier with its defaults except the
listed tree/depth/thread settings. Thus forest defaults differ between engines.

For each of float32 / float64 and 1 / 8 threads, two discarded warmup fits per
engine precede five measured seeds (1, 7, 42, 100, 123). Engine order rotates
between a4, candidate and BorutaPy + ExtraTrees. Every fit uses a new process.
The parent polls child RSS every 5 ms while Rust may hold the Python GIL.
Initial RSS is taken after imports and data generation; incremental peak RSS is
sampled peak minus that initial RSS. Short peaks can be missed. Times cover the
whole fit, including conversion and shadow preparation.

Explicit input-conversion and permutation buffers are reported separately. a4
buffer sizes are reconstructed from its source (it lacks those diagnostic
fields). `permutation_bytes` describes the largest per-round retained
permutation array plus its row-index vector, not cumulative writes or transient
allocator usage. `copied_bytes` retains its a4 meaning: cumulative augmented
float-matrix writes, zero on the native path. None of these buffer counters is
RSS or total process allocation.

For float32 industrial input, explicit buffers shrink from 800,800,000 bytes to
200,400,000 bytes: **600,400,000 bytes / 572.59 MiB saved**. For float64 they
shrink from 400,800,000 to 200,400,000 bytes (191.12 MiB saved). Materialized bins
and all forest storage remain included in measured RSS.

## Results

All local gates **passed** on this machine and the stated three-round budget.
The campaign completed 84 fresh-process fits: 24 discarded warmups and 60
measured fits. Each median below contains five seeds. Paired a4 / candidate
measurements also retained identical selected features, weak support, ranking,
completed rounds and stop reason, including discarded warmups.

| Input | Threads | Engine | Median fit (s) | Median peak RSS (MiB) | Median incremental peak (MiB) |
| --- | ---: | --- | ---: | ---: | ---: |
| float32 | 1 | a4 native | 23.61 | 1334.3 | 980.1 |
| float32 | 1 | a5 native | 22.47 | 762.9 | 408.3 |
| float32 | 1 | BorutaPy + ExtraTrees | 8.29 | 1315.2 | 960.6 |
| float32 | 8 | a4 native | 8.36 | 1383.1 | 1028.9 |
| float32 | 8 | a5 native | 7.85 | 809.4 | 453.2 |
| float32 | 8 | BorutaPy + ExtraTrees | 4.94 | 1350.7 | 995.9 |
| float64 | 1 | a4 native | 24.00 | 1146.8 | 600.5 |
| float64 | 1 | a5 native | 23.28 | 954.8 | 407.7 |
| float64 | 1 | BorutaPy + ExtraTrees | 8.46 | 2443.3 | 1897.8 |
| float64 | 8 | a4 native | 8.69 | 1192.3 | 647.4 |
| float64 | 8 | a5 native | 8.08 | 1000.6 | 455.2 |
| float64 | 8 | BorutaPy + ExtraTrees | 5.03 | 2489.8 | 1943.6 |

| Input | Threads | a5 / a4 time | Incremental peak RSS reduction vs a4 | Local gate |
| --- | ---: | ---: | ---: | --- |
| float32 | 1 | 0.952× | 58.3% | passed |
| float32 | 8 | 0.939× | 55.9% | passed |
| float64 | 1 | 0.970× | 32.1% | passed |
| float64 | 8 | 0.929× | 29.7% | passed |

The memory gate applies to float32 (at least 25% lower median incremental
peak); the time gate applies to all four settings (at most 1.05× a4). a5's total
sampled peak RSS was also below BorutaPy + ExtraTrees in all four local settings.
ExtraTrees remains faster in these shallow three-round fits; the table reports
that difference alongside memory, without implying equivalent forest algorithms.

The float32 reductions are **58.3% with one thread** and **55.9% with eight
threads** in incremental median peak RSS. Median fit time improves by 4.8% and
6.1%, respectively. Raw timings, initial RSS, peaks, diagnostics and package
versions are in [the local campaign JSON](../benchmarks/results/a5-local.json).
Explicit-buffer savings remain 572.59 MiB; they are not an RSS estimate.

## Reproduction and later Linux campaign

Build isolated environments from the a4 tag and candidate with identical
Python/dependency versions, install release wheels, then run:

```sh
python benchmarks/compare_a5.py \
  --a4-python /absolute/a4/env/bin/python \
  --candidate-python /absolute/a5/env/bin/python \
  --correctness --output benchmarks/results/a5-equivalence.json
python benchmarks/compare_a5.py \
  --a4-python /absolute/a4/env/bin/python \
  --candidate-python /absolute/a5/env/bin/python \
  --rounds 4 --output benchmarks/results/a5-local.json
```

The harness accepts rows, feature count, threads, seeds, trees and round budget
for the later four-profile Linux campaign. Its default round budget is 20;
this local report explicitly uses 4. The JSON retains discarded warmups and
per-fit configuration, versions, timing, initial RSS, sampled peak, incremental
peak, selected features and diagnostics. Final summaries exclude warmups.

These measurements only establish behavior on this local machine and stated
round budget. They do not validate four-profile performance, scalability,
Linux behavior or full-campaign RSS versus ExtraTrees. No release is published
and historical article tables are not replaced.
