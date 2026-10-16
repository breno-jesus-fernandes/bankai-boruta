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
before release. Do not publish a3 until compatibility and benchmark results
have been reviewed. Dependency upgrades must align the Python pin and the Rust
revision and rerun numerical parity, algorithm tests and performance campaigns.
The scheduled BankaiRF check reports upstream releases without updating pins.
