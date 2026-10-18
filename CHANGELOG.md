# Changelog

## 0.1.0a7 (unreleased)

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
