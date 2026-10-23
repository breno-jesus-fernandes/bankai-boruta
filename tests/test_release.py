import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile
import pytest

from scripts.release_artifacts import SBOMS, project_release, verify_artifacts
from scripts.publish_github_release import ensure_release

VERSION = '0.1.0a11'
PIN = 'bankai-random-forest==0.1.0a18'


def metadata(version=VERSION, pin=PIN):
    return f'Metadata-Version: 2.4\nName: bankai-boruta\nVersion: {version}\nRequires-Dist: {pin}\n\n'.encode()


@pytest.fixture
def artifacts(tmp_path):
    packages = tmp_path / 'packages'
    packages.mkdir()
    for platform in ['manylinux_2_28_x86_64', 'manylinux_2_28_aarch64', 'macosx_11_0_x86_64', 'macosx_11_0_arm64', 'win_amd64']:
        path = packages / f'bankai_boruta-{VERSION}-cp311-abi3-{platform}.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'bankai_boruta-{VERSION}.dist-info/METADATA', metadata())
    with tarfile.open(packages / f'bankai_boruta-{VERSION}.tar.gz', 'w:gz') as archive:
        raw = metadata()
        member = tarfile.TarInfo(f'bankai_boruta-{VERSION}/PKG-INFO')
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    for name in SBOMS:
        (tmp_path / name).write_text(json.dumps({'bomFormat': 'CycloneDX'}))
    return tmp_path


def test_complete_release_manifest(artifacts):
    paths = verify_artifacts(artifacts, VERSION, PIN)
    assert len(paths) == 10
    lines = (artifacts / 'SHA256SUMS').read_text().splitlines()
    assert len(lines) == 9
    for path in paths[:-1]:
        assert f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}' in lines


@pytest.mark.parametrize('corruption', ['missing-platform', 'duplicate-platform', 'wrong-version', 'wrong-pin', 'optional-pin', 'missing-sbom'])
def test_incomplete_or_mismatched_release_is_rejected(artifacts, corruption):
    wheels = sorted((artifacts / 'packages').glob('*.whl'))
    if corruption == 'missing-platform':
        wheels[0].unlink()
    elif corruption == 'duplicate-platform':
        linux = next(p for p in wheels if 'manylinux' in p.name)
        linux.rename(linux.with_name(f'bankai_boruta-{VERSION}-cp311-abi3-macosx_12_0_arm64.whl'))
    elif corruption == 'missing-sbom':
        next(artifacts.glob('*.cdx.json')).unlink()
    else:
        raw = metadata(version='0.1.0a2') if corruption == 'wrong-version' else metadata(pin='bankai-random-forest==0.1.0a14' if corruption == 'wrong-pin' else PIN + '; extra == "bankai-rf"')
        with zipfile.ZipFile(wheels[0], 'w') as archive:
            archive.writestr(f'bankai_boruta-{VERSION}.dist-info/METADATA', raw)
    with pytest.raises(ValueError):
        verify_artifacts(artifacts, VERSION, PIN)


def test_version_alignment(tmp_path):
    (tmp_path / 'pyproject.toml').write_text(f'[project]\nversion="{VERSION}"\ndependencies=["{PIN}"]\n')
    (tmp_path / 'Cargo.toml').write_text('[workspace.package]\nversion="0.1.0-alpha.11"\n')
    assert project_release(tmp_path) == (VERSION, PIN)
    (tmp_path / 'Cargo.toml').write_text('[workspace.package]\nversion="0.1.0-alpha.2"\n')
    with pytest.raises(ValueError, match='versions differ'):
        project_release(tmp_path)


class FakeGitHub:
    def __init__(self):
        self.release = None
        self.creates = 0
        self.publishes = 0
        self.fail_upload = None

    def request(self, method, path, data=None):
        if method == 'GET':
            return copy.deepcopy(self.release)
        if method == 'PATCH':
            self.publishes += 1
            self.release.update(data)
            return copy.deepcopy(self.release)
        if path.endswith('/releases'):
            self.creates += 1
            assert data['draft'] is True
            self.release = dict(data, id=1, assets=[], upload_url='https://uploads.github.com/test{?name,label}', html_url='https://github.com/test/releases/v0.1.0a9')
            return copy.deepcopy(self.release)
        from urllib.parse import parse_qs, urlparse
        name = parse_qs(urlparse(path).query)['name'][0]
        if name == self.fail_upload:
            raise RuntimeError('simulated upload failure')
        asset = {'name': name, 'state': 'uploaded', 'digest': 'sha256:' + hashlib.sha256(data).hexdigest()}
        self.release['assets'].append(asset)
        return copy.deepcopy(asset)


def test_release_publishes_only_complete_prerelease_and_retries_safely(artifacts):
    paths = verify_artifacts(artifacts, VERSION, PIN)
    client = FakeGitHub()
    client.fail_upload = paths[2].name
    with pytest.raises(RuntimeError, match='upload failure'):
        ensure_release(client, 'owner/repo', VERSION, 'abc123', paths)
    assert client.release['draft'] is True
    assert client.publishes == 0
    assert len(client.release['assets']) == 2
    client.fail_upload = None
    result = ensure_release(client, 'owner/repo', VERSION, 'abc123', paths)
    assert result['draft'] is False
    assert result['prerelease'] is True
    assert result['target_commitish'] == 'abc123'
    assert len(result['assets']) == 10
    ensure_release(client, 'owner/repo', VERSION, 'abc123', paths)
    assert client.creates == client.publishes == 1
    assert len(client.release['assets']) == 10


def test_existing_asset_is_not_silently_overwritten(artifacts):
    paths = verify_artifacts(artifacts, VERSION, PIN)
    client = FakeGitHub()
    ensure_release(client, 'owner/repo', VERSION, 'abc123', paths)
    client.release['assets'][0]['digest'] = 'sha256:wrong'
    with pytest.raises(ValueError, match='differs'):
        ensure_release(client, 'owner/repo', VERSION, 'abc123', paths)
    assert client.publishes == 1


def test_current_release_versions_are_consistent():
    import tomllib
    from bankai_boruta import __version__

    root = Path(__file__).resolve().parents[1]
    version, _ = project_release(root)
    assert __version__ == version == VERSION
    rust_version = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']['package']['version']
    boundary = tomllib.loads((root / 'crates/python/Cargo.toml').read_text())
    assert boundary['dependencies']['boruta-core']['version'] == '=' + rust_version
    python_lock = tomllib.loads((root / 'uv.lock').read_text())
    assert next(p['version'] for p in python_lock['package'] if p['name'] == 'bankai-boruta') == version
    rust_lock = tomllib.loads((root / 'Cargo.lock').read_text())
    for name in ('boruta-core', 'bankai-boruta-python'):
        assert next(p['version'] for p in rust_lock['package'] if p['name'] == name) == rust_version


def test_immutable_forest_source_alignment():
    import re
    import tomllib
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root/'pyproject.toml').read_text())
    source = project['tool']['uv']['sources']['bankai-random-forest']
    assert re.fullmatch('[0-9a-f]{40}', source['rev'])
    dependencies = tomllib.loads((root/'crates/boruta-core/Cargo.toml').read_text())['dependencies']
    for name in ('bankai-random-forest', 'xrf'):
        assert dependencies[name]['git'] == source['git']
        assert dependencies[name]['rev'] == source['rev']
    for lock in ('uv.lock','Cargo.lock'):
        package = next(p for p in tomllib.loads((root/lock).read_text())['package'] if p['name']=='bankai-random-forest')
        assert source['rev'] in str(package['source'])
        assert package['version'] == ('0.1.0a18' if lock=='uv.lock' else '0.1.0-alpha.18')
