"""Validate the complete a3 release set before attestation or publication."""
from email.parser import BytesParser
import hashlib
from pathlib import Path
import re
import sys
import tarfile
import tomllib
import zipfile
import json

PLATFORMS = {'linux-x86_64', 'linux-aarch64', 'macos-x86_64', 'macos-arm64', 'windows-amd64'}
SBOMS = {'bankai-boruta-boruta-core-rust.cdx.json', 'bankai-boruta-python-rust.cdx.json', 'bankai-boruta-python.cdx.json'}


def project_release(root):
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    version = project['version']
    rust_version = re.sub(r'(a|b|rc)(\d+)$', lambda m: {'a': '-alpha.', 'b': '-beta.', 'rc': '-rc.'}[m[1]] + m[2], version)
    cargo = tomllib.loads((root / 'Cargo.toml').read_text())
    if cargo['workspace']['package']['version'] != rust_version:
        raise ValueError('Python and Rust release versions differ')
    pin = next(dep for dep in project['dependencies'] if dep.startswith('bankai-random-forest=='))
    return version, pin


def check_metadata(raw, version, pin):
    metadata = BytesParser().parsebytes(raw)
    if metadata['Name'] != 'bankai-boruta' or metadata['Version'] != version:
        raise ValueError('Distribution name/version does not match the release')
    requirements = [re.sub(r'\s+', '', r) for r in metadata.get_all('Requires-Dist', [])]
    if pin not in requirements:
        raise ValueError('Distribution is missing the mandatory BankaiRF pin')


def wheel_platform(name):
    platform = name.removesuffix('.whl').rsplit('-', 1)[-1]
    for prefix, suffix, key in [('manylinux', 'x86_64', 'linux-x86_64'), ('manylinux', 'aarch64', 'linux-aarch64'),
                                ('macosx', 'x86_64', 'macos-x86_64'), ('macosx', 'arm64', 'macos-arm64'),
                                ('win_', 'amd64', 'windows-amd64')]:
        if platform.startswith(prefix) and platform.endswith(suffix):
            return key
    raise ValueError(f'Unexpected wheel platform: {name}')


def verify_artifacts(directory, version, pin):
    wheels = sorted((directory / 'packages').glob('*.whl'))
    sources = sorted((directory / 'packages').glob('*.tar.gz'))
    sboms = sorted(directory.glob('*.cdx.json'))
    if len(wheels) != 5 or len(sources) != 1 or {p.name for p in sboms} != SBOMS:
        raise ValueError('Release requires five wheels, one source distribution and all three SBOMs')
    platforms = set()
    for path in wheels:
        if not path.name.startswith(f'bankai_boruta-{version}-cp311-abi3-'):
            raise ValueError(f'Unexpected wheel name or ABI: {path.name}')
        platforms.add(wheel_platform(path.name))
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(entries) != 1:
                raise ValueError(f'Wheel must have one metadata record: {path.name}')
            check_metadata(archive.read(entries[0]), version, pin)
    if platforms != PLATFORMS:
        raise ValueError('Wheel set is missing a required platform or contains duplicates')
    if sources[0].name != f'bankai_boruta-{version}.tar.gz':
        raise ValueError('Unexpected source distribution version')
    with tarfile.open(sources[0]) as archive:
        records = [m for m in archive.getmembers() if m.name.count('/') == 1 and m.name.endswith('/PKG-INFO')]
        if len(records) != 1:
            raise ValueError('Source distribution must have one package metadata record')
        check_metadata(archive.extractfile(records[0]).read(), version, pin)
    for path in sboms:
        if json.loads(path.read_text()).get('bomFormat') != 'CycloneDX':
            raise ValueError(f'Invalid CycloneDX SBOM: {path.name}')
    paths = sorted(wheels + sources + sboms, key=lambda p: p.name)
    checksum = directory / 'SHA256SUMS'
    checksum.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in paths))
    return paths + [checksum]


if __name__ == '__main__':
    version, pin = project_release(Path(__file__).resolve().parents[1])
    paths = verify_artifacts(Path(sys.argv[1] if len(sys.argv) > 1 else 'dist'), version, pin)
    print(f'Validated release {version}: {len(paths)} assets including SHA256SUMS')
