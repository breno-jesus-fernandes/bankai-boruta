# Bankai Boruta

Native Rust implementation of the Boruta all-relevant feature selection
algorithm with a scikit-learn compatible Python API.

The project is in early development. The public estimator contract and native
state ledger are under development; `BankaiBoruta.fit` is not available until
the native Boruta loop is implemented. See [`docs/ROADMAP.md`](docs/ROADMAP.md)
for the architecture, milestone status, test strategy, and benchmark criteria.

## Development

Python wheels and editable installs use Maturin:

```bash
uv sync --extra test
uv run pytest
```

Rust core tests run with Cargo:

```bash
cargo test --workspace
```
