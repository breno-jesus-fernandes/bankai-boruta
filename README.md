# Bankai Boruta

Native Rust implementation of the Boruta all-relevant feature selection
algorithm with a scikit-learn compatible Python API.

The canonical classification selector now runs its complete Boruta iteration
loop inside Rust. `BankaiBoruta.fit(X, y)` accepts C- or F-contiguous NumPy
`float32`/`float64` arrays without copying `X`; incompatible numeric inputs can
use `input_mode="copy"`. Rejected features are pruned between iterations,
confirmed features remain as context in canonical mode, and unresolved
features stay tentative when `max_iter` is reached. `pruning="aggressive"`
removes confirmed features from subsequent training and is experimental; use it
only when the benchmarked trade-off is acceptable for your data. Deterministic
multi-worker training is available with `n_jobs`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for details.

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
