"""Report newer BankaiRF versions without changing dependency pins."""
import json
from pathlib import Path
import tomllib
import urllib.request
from packaging.version import Version

project = tomllib.loads(Path('pyproject.toml').read_text())
pinned = next(dep.split('==')[1] for dep in project['project']['dependencies']
              if dep.startswith('bankai-random-forest=='))
with urllib.request.urlopen('https://pypi.org/pypi/bankai-random-forest/json', timeout=30) as response:
    upstream = json.load(response)
newer = sorted(Version(version) for version, files in upstream['releases'].items()
               if files and any(not item.get('yanked', False) for item in files)
               and Version(version) > Version(pinned))
print(f'Validated BankaiRF: {pinned}; newer published versions: {newer}')
if newer:
    print('::warning::Align Python/Rust pins and validate numerical parity and performance before upgrading.')
