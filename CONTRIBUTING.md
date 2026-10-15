# Contributing

Thanks for considering a contribution to Bankai Boruta. Contributions are welcome through GitHub issues and pull requests. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- For bugs and improvement ideas, use the corresponding [issue form](https://github.com/breno-jesus-fernandes/bankai-boruta/issues/new/choose).
- For security vulnerabilities, follow the private reporting instructions in [SECURITY.md](.github/SECURITY.md); do not open a public issue.
- For substantial changes, open an issue first so maintainers can discuss the approach.
- The public Python estimator API is not expected to change as part of routine contributions. Discuss any proposed API or serialized-model compatibility change in an issue first.

## Development setup

Install Git, Python 3.11 or newer, a stable Rust toolchain, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). On Windows, install the Rust MSVC build tools; on macOS, install the Xcode command-line tools.

Fork the repository on GitHub, then clone your fork and configure the upstream remote:

```bash
git clone https://github.com/<your-user>/bankai-boruta.git
cd bankai-boruta
git remote add upstream https://github.com/breno-jesus-fernandes/bankai-boruta.git
uv python install 3.11
uv sync --locked --no-install-project --extra test --extra arrow-test
uv run --no-sync maturin develop --release --locked
```

Create a focused branch from the current `main` and keep it up to date with upstream:

```bash
git fetch upstream
git switch -c descriptive-change upstream/main
```

## Checks

Run both suites from the repository root before opening a pull request:

```bash
uv run --no-sync pytest -q
uv run --no-sync cargo test --workspace --locked
```

The CI matrix also builds the native extension and wheels on Linux, Windows, and macOS. A local pass does not replace the required CI checks.

## Dependencies and lockfiles

Keep `uv.lock` and `Cargo.lock` reproducible. If dependencies change, update the relevant manifest and lockfile intentionally, explain the reason in the pull request, and run the checks above. Do not allow installs or tooling to update lockfiles implicitly. Dependency changes are reviewed by CI's Dependency Review, Python dependency audit, and Rust dependency audit checks. Read the [dependency security policy](.github/SECURITY.md) before changing scanner configuration or adding an exception.

## Pull requests and review

Push your branch to your fork and open a pull request against `main` in the upstream repository. Use the pull request template, describe the user-visible impact, link related issues, and include tests and documentation updates where relevant. Keep changes focused and respond to review feedback; maintainers may ask for revisions or additional checks before merging.

There is no additional Contributor License Agreement (CLA) or Developer Certificate of Origin (DCO) requirement. Contributions are covered by the project's [GPL-3.0-or-later license](COPYING).

## Maintainer GitHub settings checklist

GitHub repository settings must enforce the following for `main` (these settings cannot be enabled by files in the repository; apply equivalent checks to `develop`):

- Require a pull request before merging and require **one approval** from a maintainer.
- Require these status checks before merging:
  - `Test (Linux x86_64 / Python 3.11)`
  - `Test (Linux x86_64 / Python 3.12)`
  - `Test (Linux x86_64 / Python 3.13)`
  - `Test (Linux x86_64 / Python 3.14)`
  - `Test (Linux ARM64 / Python 3.11)`
  - `Test (Windows x86_64 / Python 3.11)`
  - `Test (macOS x86_64 / Python 3.11)`
  - `Test (macOS ARM64 / Python 3.11)`
  - `Rust dependency audit`
  - `Python dependency audit`
  - `Dependency Review`
- Allow an administrator bypass so the maintainer can merge their own pull request when needed.

The check names follow the CI matrix and security jobs documented in [SECURITY.md](.github/SECURITY.md). Verify the active ruleset or branch protection on GitHub after configuring it.
