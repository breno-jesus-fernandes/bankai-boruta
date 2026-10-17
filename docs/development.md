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
before release. Do not publish a4 until compatibility and benchmark results
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
