# CI and releases

[CI and release artifacts](../.github/workflows/ci.yml) runs for pull requests
targeting `main` or `develop`, non-documentation pushes to these branches, and
manual runs. Markdown-only pushes skip this workflow; pull requests still run.

## Tests and artifacts

Linux x86_64 tests Python 3.11–3.14. Linux ARM64, Windows x86_64, and macOS
x86_64/ARM64 test Python 3.11. Every job builds the extension and runs Python
and Rust tests with locked dependencies, including the Arrow integration.
The Linux/Python 3.11 job generates Python coverage and submits it to Codecov.
Coverage measures the Python wrapper, not the native Rust engine.

After tests pass, Maturin builds five `cp311-abi3` wheels and a source
distribution. Wheel filenames are checked against ABI and target platform.
The combined `bankai-boruta-release` artifact contains `packages/`, three Rust
CycloneDX SBOMs (one per workspace crate), and a Python CycloneDX SBOM including
optional runtime integrations. Artifacts are retained for 14 days.

## Automatic releases

Only pushes to `main` can publish. The workflow checks the package version in
`pyproject.toml` against PyPI: HTTP 404 enables publication, HTTP 200 skips it,
and any other response fails the check. Tests, builds, SBOM generation, and
provenance attestations must succeed before publication. PRs, `develop` pushes,
and manual runs build artifacts without publishing.

Publication uses PyPI Trusted Publishing (OIDC), with no Twine token secrets.
Uploads of the same version run sequentially. After PyPI publication, a GitHub
Release named `v<VERSION>` is created for the tested commit with generated notes,
all distributions, and SBOMs. Alpha, beta, and RC versions are prereleases.

Update the Python version, workspace Rust version, and internal Rust dependency
version constraints together, keeping them
equivalent (for example `0.1.0a2` and `0.1.0-alpha.2`). Refresh both lockfiles,
validate locally, and merge into `main`. This change does not bump the version.

## External service setup

These settings are external to repository files and must be verified by a
maintainer before relying on automatic releases or coverage badges:

- On PyPI, configure the Trusted Publisher for owner
  `breno-jesus-fernandes`, repository `bankai-boruta`, workflow filename
  `ci.yml`, and environment `pypi`.
- In GitHub, create the `pypi` environment and restrict deployments to `main`.
  Ensure GitHub Actions can request OIDC tokens and create releases/attestations.
- Enable the repository in Codecov. If a token is required, configure the
  `CODECOV_TOKEN` Actions secret. Upload failures are non-blocking; inspect the
  Codecov upload job and dashboard before considering coverage operational.
- Enable GitHub's dependency graph and Dependabot alerts. Configure branch
  protection using the exact check names in [CONTRIBUTING.md](../CONTRIBUTING.md).

The implementation of these workflows does not establish that the external
services or branch protection have already been configured.

## Dependency security

[Dependency security](../.github/workflows/security.yml) runs on pushes and
pull requests to `main`/`develop`, weekly on Monday, and manually. It reviews
PR dependency changes and audits the Rust and Python lockfiles, including Python
extras. See the [security policy](../.github/SECURITY.md) for scanner behavior.
Dependabot proposes weekly updates for Cargo, uv, and GitHub Actions.
