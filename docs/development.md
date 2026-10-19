# Development

```sh
uv sync --locked --no-install-project --extra test
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest -q
cargo test --workspace --locked
cargo fmt --all --check
```

Python 3.11+ is supported. CI builds and smoke-tests ABI3 wheels for Linux
x86_64/ARM64, macOS x86_64/ARM64 and Windows x86_64, and tests Python 3.11–3.14.
Local validation covers only the host platform; CI results must be inspected
before release. Do not publish a7 until compatibility and benchmark results
have been reviewed. Dependency upgrades must align the Python pin and the Rust
revision and rerun numerical parity, algorithm tests and performance campaigns.
The scheduled BankaiRF check reports upstream releases without updating pins.

## Release automation

Merging a version update into `main` runs the test matrix, builds five ABI3
wheels and the source distribution, generates three SBOMs, and validates the
release metadata and mandatory BankaiRF pin. `SHA256SUMS` covers the nine build
assets. GitHub provenance attestation precedes publication.

GitHub release publication is independent of PyPI's version-exists check and
upload job. A new GitHub release remains a draft until all ten assets, including
the checksum manifest, have been uploaded and their GitHub SHA-256 digests
verified. Alpha versions such as `0.1.0a4` publish under `v0.1.0a4` with the
prerelease flag. Existing assets are never silently overwritten.

For an interrupted upload, rerun the failed publication job in the original
Actions run so it reuses the same immutable build artifacts. Completed uploads
are reused. A fresh workflow can also be dispatched on `main`; if rebuilt assets
have different bytes from an existing release, publication fails explicitly
instead of replacing them. Pull requests and `develop` builds do not publish.
PyPI upload still uses the configured Trusted Publisher and only runs for a
version not yet present on PyPI.

## a9 release order

BankaiRF **0.1.0a18** must be published before Bankai Boruta **0.1.0a9**. Development sources use the immutable
RF git revision recorded in pyproject.toml and the Rust manifest; no mutable
branch or local Cargo patch is part of the release. The distribution's mandatory
Python dependency is RF ==0.1.0a18. After RF is published, verify public wheel
installation in a clean environment before authorizing Boruta publication.

CI still builds artifacts automatically. PyPI upload now additionally requires
an authorized manual dispatch with `publish=true` on the default branch. Example
after approval: `gh workflow run ci.yml --ref main -f publish=true`. Opening or
merging the PR does not authorize that dispatch. No publication was executed.

macOS local builds of the Boruta extension need dependency-wide dynamic symbol
lookup because the RF dependency also emits a cdylib:
`RUSTFLAGS='-C link-arg=-undefined -C link-arg=dynamic_lookup' uv run --no-sync maturin develop --release --locked`.
CI configures this flag on macOS. Other platforms retain their normal linker settings.

The Python audit preserves hashes for registry dependencies, including the RF
transitive requirements. Its export excludes only the immutable RF VCS project,
which pip-audit cannot hash; that project's own audit runs in RF PR #21.
Release tests check the full git revision against both manifests and lockfiles.

PR installed-wheel smokes resolve RF from the same immutable development git pin.
Publication builds instead resolve RF alpha16 from PyPI, so their success remains
conditional on publishing RF first. Both paths exercise the installed exact
session as well as the ordinary binned dtype smoke.
