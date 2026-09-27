# Bankai Boruta

Native Rust implementation of the Boruta all-relevant feature selection
algorithm with a scikit-learn compatible Python API.

The canonical classification selector now runs its complete Boruta iteration
loop inside Rust. `BankaiBoruta.fit(X, y)` accepts C- or F-contiguous NumPy
`float32`/`float64` arrays without copying `X`; incompatible numeric inputs can
use `input_mode="copy"`. Rejected features are pruned between iterations,
confirmed features remain as context, and unresolved features stay tentative
when `max_iter` is reached. Aggressive pruning and Polars support are planned
for later milestones. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for details.

## Development

Python wheels and editable installs use Maturin:

```bash
uv sync --extra test
uv run maturin develop --skip-install
uv run pytest
```

Rust core tests run with Cargo:

```bash
cargo test --workspace
```
