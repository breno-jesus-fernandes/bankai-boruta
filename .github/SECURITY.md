# Dependency security policy

Cargo and uv lockfiles are the reproducible dependency inputs. CI must use locked installs and must not update either lockfile implicitly.

Dependency Review blocks pull requests that introduce high or critical severity vulnerabilities. `pip-audit` audits the complete pinned export of `uv.lock` without installing
packages or re-resolving dependencies; `cargo-deny` audits Rust advisories, licenses, duplicate versions, and dependency sources. The pinned cargo-deny version blocks all RustSec vulnerability advisories; Dependency Review applies the high-or-critical threshold to dependency changes in PRs. Dependency Review reports lower severities in its PR summary without failing its check; the lockfile audits run independently.

Exceptions must be narrowly scoped and recorded beside the corresponding scanner configuration with the advisory or crate, rationale, responsible maintainer, and review date. Rust advisory exceptions belong in `.cargo/deny.toml`. Do not suppress unresolved Python vulnerabilities; update the lockfile or track the exception in a dedicated issue and document its identifier and review date in the security workflow.

GitHub branch rules must require pull requests and these checks before merging into `main`:

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

Configure the PyPI Trusted Publisher for this repository, workflow filename
`ci.yml` (display name `CI and release artifacts`), and the `pypi` environment. Enable GitHub's dependency graph and Dependabot alerts in repository settings.

## Reporting vulnerabilities

Report vulnerabilities privately to
[breno.de.jesus-fernandes@gmail.com](mailto:breno.de.jesus-fernandes@gmail.com),
the maintainer contact used by the Bankai project. Include affected versions,
a minimal reproduction, and the potential impact. Do not open a public issue
with exploit details.

## Boruta-specific dependency inputs

The forest engine comes from the Bankai Random Forest Git repository at the
revision pinned in `crates/boruta-core/Cargo.toml`. Only that Git origin is
allowed in `.cargo/deny.toml`; no vulnerability advisories are ignored.
The license allowlist includes CC0-1.0 for Arrow's transitive `tiny-keccak`
dependency. Python audits include every optional integration and development
group. See [release setup](../docs/development.md#external-service-setup) for
Trusted Publisher and Codecov configuration.
