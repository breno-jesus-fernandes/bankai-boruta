"""Publish a complete GitHub release, independently of PyPI publication state.

New releases remain drafts until all assets are uploaded and verified. Retries
reuse existing assets and reject changed bytes instead of overwriting a release.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from .release_artifacts import project_release, verify_artifacts
else:
    from release_artifacts import project_release, verify_artifacts


class GitHub:
    def __init__(self, token):
        self.token = token

    def request(self, method, path, data=None):
        url = path if path.startswith('https://uploads.github.com/') else 'https://api.github.com' + path
        headers = {'Authorization': 'Bearer ' + self.token, 'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28'}
        if isinstance(data, bytes):
            headers['Content-Type'] = 'application/octet-stream'
            body = data
        else:
            headers['Content-Type'] = 'application/json'
            body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if method == 'GET' and error.code == 404:
                return None
            raise RuntimeError(f'GitHub {method} failed with HTTP {error.code}') from None


def ensure_release(client, repository, version, commit, paths):
    base = '/repos/' + repository
    tag = 'v' + version
    prerelease = bool(re.search(r'(a|b|rc)\d+$', version))
    release = client.request('GET', base + '/releases/tags/' + urllib.parse.quote(tag))
    if release is None:
        release = client.request('POST', base + '/releases', {
            'tag_name': tag, 'target_commitish': commit, 'name': tag,
            'draft': True, 'prerelease': prerelease, 'generate_release_notes': True,
        })
    existing = {asset['name']: asset for asset in release['assets']}
    upload_url = release['upload_url'].split('{', 1)[0]
    for path in paths:
        digest = 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()
        asset = existing.get(path.name)
        if asset is not None:
            if asset.get('digest') != digest or asset.get('state') != 'uploaded':
                raise ValueError(f'Existing release asset differs: {path.name}; retry the original publication run')
        else:
            asset = client.request('POST', upload_url + '?name=' + urllib.parse.quote(path.name), path.read_bytes())
            if asset.get('digest') != digest or asset.get('state') != 'uploaded':
                raise ValueError(f'Uploaded release asset failed checksum verification: {path.name}')
    complete = client.request('GET', base + '/releases/' + str(release['id']))
    uploaded = {a['name']: a for a in complete['assets']}
    for path in paths:
        asset = uploaded.get(path.name, {})
        if asset.get('state') != 'uploaded' or asset.get('digest') != 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Incomplete release asset: {path.name}')
    if complete['draft'] or complete['prerelease'] != prerelease:
        complete = client.request('PATCH', base + '/releases/' + str(release['id']), {'draft': False, 'prerelease': prerelease})
    print(f'Published {tag}: {complete["html_url"]} ({len(paths)} verified assets)')
    return complete


if __name__ == '__main__':
    version, pin = project_release(Path(__file__).resolve().parents[1])
    paths = verify_artifacts(Path('dist'), version, pin)
    token = os.environ.get('GH_TOKEN') or os.environ['GITHUB_TOKEN']
    ensure_release(GitHub(token), os.environ['GITHUB_REPOSITORY'], version, os.environ['GITHUB_SHA'], paths)
