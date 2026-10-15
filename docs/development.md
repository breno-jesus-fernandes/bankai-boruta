# CI and releases

The [GitHub Actions workflow](../.github/workflows/ci.yml) runs on pushes and
pull requests to `main` and `develop`, and can also be started manually.

## CI

Python 3.11 and Rust tests run on Linux x86_64/ARM64, Windows x86_64, and
macOS x86_64/ARM64. Test dependencies include the optional Arrow integration.
After tests pass, Maturin builds one stable-ABI wheel per platform and a source
distribution. The combined `bankai-boruta-release` artifact is retained for
14 days.

## Publish to PyPI

A push to `main` checks the version in `pyproject.toml` against PyPI. If that
version is missing, the workflow publishes after all tests and builds pass.
An existing version is skipped; an unexpected PyPI response fails the check.
Pull requests, `develop` pushes, and manual runs only test and build artifacts.

Configure repository Actions secrets `TWINE_USERNAME` (`__token__` for a PyPI
API token) and `TWINE_PASSWORD` (the token). Upload uses Twine and validates
distribution metadata first. Uploads of the same version run sequentially.

To release, update the Python version in `pyproject.toml` and the Rust version
in the workspace `Cargo.toml`, refresh `uv.lock` and `Cargo.lock`, and merge to
`main`. Keep the Python and Rust versions equivalent (for example, `0.1.0a2`
and `0.1.0-alpha.2`).
