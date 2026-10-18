# Exact session a7 validation

Status: implementation and local parity pass; default activation and release
acceptance remain pending. Reference: published Boruta 0.1.0a6 / RF 0.1.0a15.
Candidate: Boruta a7, RF a16 source pinned to
`da52a5bc0e2358911131d3b06d9cea65fc360505`.

## Checks completed locally

- BankaiRF: 269 Python tests; Rust workspace suites: 34 core, 5 integration,
  25 XRF tests, all passed. Exact tree multisets include splits, scores, leaves
  and covers for both bank dtypes, binary/multiclass, bootstrap and gini/entropy.
- Boruta: 309 Python tests and Rust workspace tests passed. This includes
  original dtypes/layouts, repeated values/constants, active-list invalidation,
  duplicate shadows, controls, invalid maps, snapshot isolation, repeated fits,
  custom RandomState fallback, explicit owned-shuffle parity and RNG state,
  including the cached Gaussian value.
- All completed fresh-process comparisons below passed exact hits, decisions,
  selected/tentative features, ranking, rounds, stop reason and data hashes;
  importances/thresholds passed atol=rtol=1e-12.

## Sequential local measurements

Three repetitions per variant/cell; order alternates and no builds/tests run
concurrently with the final measurements. All controlled fixed trials use 25
trees, depth 6, bootstrap, sqrt and five rounds. Complete controlled trials use
10 trees, depth 6, max_iter 120 and patience 100. They are smaller fixtures, not
Industrial or production convergence evidence. Individual observations, quartiles,
baseline RSS, fit RSS delta, hashes and package versions are in the JSON files
under `benchmarks/results/exact-session-local-*.json`.

| Fixture | workers | a6 median s | session median s | a6 peak RSS MiB (median) | session peak RSS MiB (median) |
|---|---:|---:|---:|---:|---:|
| complete 2,000 × 40 | 1 | 0.1116 | 0.0875 | 153.5 | 148.5 |
| complete 2,000 × 40 | 16 | 0.0636 | 0.0391 | 165.6 | 161.3 |
| complete 2,000 × 40 | 4 | 0.0668 | 0.0441 | 165.0 | 160.7 |
| fixed 20,000 × 100 | 1 | 1.2176 | 0.8687 | 198.3 | 203.4 |
| fixed 20,000 × 100 | 16 | 0.3137 | 0.2756 | 247.1 | 236.4 |
| fixed 20,000 × 100 | 4 | 0.5000 | 0.3412 | 228.1 | 208.4 |
| fixed 50,000 × 200 | 1 | 4.5805 | 3.3928 | 378.6 | 336.4 |
| fixed 50,000 × 200 | 4 | 2.2463 | 1.6245 | 384.8 | 357.8 |

Host: macOS-26.6.2-arm64-arm-64bit-Mach-O; 8 logical CPUs; 8 GiB RAM. The 16-worker cells oversubscribe this machine and must not be
interpreted as 16 physical-core scaling. RSS is sampled externally every 5 ms
only during fit; VmHWM is unavailable on macOS. Absolute RSS in the small
one-worker fixed cell did not improve and varies with process baseline. The
buffer counters and RSS deltas do not override that observation.

The larger 50,000 × 200 fixed workload also passed parity. At one worker,
median fit time decreased 25.9% and median peak RSS decreased 11.2%; at four
workers, median time decreased 27.7% and peak RSS decreased 7.0%. These remain
local controlled results, not full Industrial acceptance. Preparation improved
while training itself remains more expensive with indirect reads; the separate
phase counters expose that tradeoff. Complete 2,000 × 40 fits converged in 22
rounds on both paths. Initial designs that retained a second permutation snapshot
or repeatedly dispatched dtype/maps in hot reads are preserved in
`benchmarks/results/exact-session-development/`; they are not pooled with the
final candidate. The final standard MT19937/u32 path keeps one explicit native
permutation buffer and dispatches reads once per feature. Custom-RNG/int64
fallbacks retain the safe owned-copy boundary.

## Linux evidence and pending gates

Hosted Linux controlled validation passed on a four-logical-CPU Azure runner:
[CI job 117783104729](https://github.com/breno-jesus-fernandes/bankai-boruta/actions/runs/39164027968/job/117783104729).
The job used commit `6ebe1ed` with the same native/session implementation and RF
pin as the current candidate. Both cells use one worker and three repetitions.
Raw JSON is in `benchmarks/results/exact-session-linux-*.json`.

| Linux fixture | a6 median s | session median s | a6 peak RSS MiB | session peak RSS MiB |
|---|---:|---:|---:|---:|
| complete_fit | 0.1369 | 0.0974 | 154.9 | 155.7 |
| fixed_rounds | 1.3345 | 1.1356 | 193.4 | 188.6 |

All Linux trials passed numerical/decision parity. VmHWM reset succeeded and its
scope is fit in every observation; raw HWM and external RSS are both recorded.
The fixed workload reduced runtime and RSS, but the small complete-fit workload
increased median RSS by about 0.8 MiB. This fails a universal time-and-memory
acceptance claim, even though its runtime improved. The default therefore remains
materialized. Full cross-platform artifact review is still pending.

Dedicated AMD trials remain pending: workers 1/4/16,
seeds 42–46, repeated fixed rounds AND complete Industrial, same host and
sequential variants. The complete Industrial generator, max_iter, patience,
bootstrap and tree budget are unchanged. Local controlled observations cannot
satisfy this gate.

No AWS capacity was provisioned or budget ledger modified. The dedicated
candidate campaign has not consumed the existing r5 calibration reservation.
Its duration/cost must fit the reconciled authorized balance and verified
automatic shutdown; otherwise the unfinished cells need further authorization.

The session stays opt-in. Local ABI3 wheels for both packages and the Boruta source distribution were
built. A clean environment using those local wheels passed installed-package
binned dtype and exact-session smokes; wheel/sdist metadata requires a7/RF a16,
and the sdist includes the tests and comparison script. This does not substitute
for resolving RF a16 from PyPI after its release. No PyPI publication was performed. Publish RF a16
first, then verify clean public installation of Boruta a7 and review all
platform artifacts before an explicitly authorized publishing dispatch.
