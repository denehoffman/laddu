"""
Release classification and gates shared by the generated GitHub workflows.

Only the Python standard library is needed. Git revisions come from event payloads,
never from shell interpolation. An unknown change takes the full-CI path.
"""

# CI invokes fixed tools using shell-free argument lists.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

RELEASE_BRANCH = 'release-please--branches--main--components--laddu'
STABLE_VERSION = re.compile(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z')
WRAPPERS = {'laddu-python', 'laddu-python-local', 'laddu-python-mpi'}
RELEASE_FILES = {
    'Cargo.toml',
    'Cargo.lock',
    'python/laddu/pyproject.toml',
    '.release-please-manifest.json',
    'CITATION.cff',
    'CHANGELOG.md',
    'version.txt',
}


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True)


def version_tuple(version: str) -> tuple[int, ...]:
    if not STABLE_VERSION.fullmatch(version):
        msg = f'Expected a stable release version, got {version!r}'
        raise ValueError(msg)
    return tuple(map(int, version.split('.')))


def normalize_metadata(path: str, content: str) -> object:  # noqa: PLR0912 -- explicit release file formats
    """Discard only release-managed fields, preserving all executable configuration."""
    if path == 'CHANGELOG.md':
        return '<changelog>'
    if path == 'version.txt':
        version_tuple(content.strip())
        return '<version>'
    if path == 'CITATION.cff':
        return re.sub(r'(?m)^(version|date-released):.*$', r'\1: <release>', content)
    if path == '.release-please-manifest.json':
        data = json.loads(content)
        version_tuple(data['.'])
        data['.'] = '<version>'
        return data
    data = tomllib.loads(content)
    if path == 'Cargo.toml':
        data['workspace']['package']['version'] = '<version>'
        for dependency in data['workspace']['dependencies'].values():
            if isinstance(dependency, dict) and 'path' in dependency and 'version' in dependency:
                dependency['version'] = '<version>'
    elif path == 'Cargo.lock':
        local = {package['name'] for package in data['package'] if 'source' not in package}
        for package in data['package']:
            if package['name'] in local and 'source' not in package:
                package['version'] = '<version>'
            for i, dependency in enumerate(package.get('dependencies', [])):
                parts = dependency.split()
                if len(parts) == 2 and parts[0] in local:  # noqa: PLR2004 -- Cargo lock dependency name/version pair
                    package['dependencies'][i] = f'{parts[0]} <version>'
    elif path == 'python/laddu/pyproject.toml':
        requirements = data['project']['optional-dependencies']['mpi']
        data['project']['optional-dependencies']['mpi'] = [
            re.sub(r'^laddu-mpi==\d+\.\d+\.\d+$', 'laddu-mpi==<version>', item) for item in requirements
        ]
    else:
        msg = f'Not a release-managed file: {path}'
        raise ValueError(msg)
    return data


def metadata_only(changes: dict[str, tuple[str | None, str | None]]) -> bool:
    if not changes:
        return False
    for path, (before, after) in changes.items():
        if before is None or after is None:
            return False
        try:
            if normalize_metadata(path, before) != normalize_metadata(path, after):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def release_pr(event: dict) -> bool:
    pr = event.get('pull_request', {})
    body = pr.get('body') or ''
    # GitHub adds issue labels after PR creation. The generated body identifies
    # the initial opened event too, so it cannot miss release validation.
    generated_body = body.startswith(':robot: I have created a release') and (
        'This PR was generated with [Release Please](https://github.com/googleapis/release-please).' in body
    )
    return (
        pr.get('base', {}).get('ref') == 'main'
        and pr.get('head', {}).get('ref') == RELEASE_BRANCH
        and pr.get('head', {}).get('repo', {}).get('full_name') == event['repository']['full_name']
        and ('autorelease: pending' in {label['name'] for label in pr.get('labels', [])} or generated_body)
    )


def read_changes(base: str, head: str) -> dict[str, tuple[str | None, str | None]]:
    # --no-renames makes file additions/removals visible to the conservative classifier.
    paths = git('diff', '--no-renames', '--name-only', '-z', base, head, '--').split('\0')
    changes = {}
    for path in filter(None, paths):
        if path not in RELEASE_FILES:
            # Unknown paths already require full CI; do not read potentially
            # large binary fixtures or source trees merely to classify them.
            changes[path] = (None, None)
            continue
        contents = []
        for revision in (base, head):
            result = subprocess.run(['git', 'show', f'{revision}:{path}'], capture_output=True, check=False)
            contents.append(result.stdout.decode('utf-8', errors='replace') if result.returncode == 0 else None)
        changes[path] = (contents[0], contents[1])
    return changes


def metrics_relevant(paths: list[str]) -> bool:
    return any(
        path.endswith(('.rs', '/Cargo.toml'))
        or path in {'Cargo.toml', 'Cargo.lock', '.yamloom.py'}
        or path.startswith(('benchmarks/', 'scripts/', '.github/workflows/'))
        for path in paths
    )


def classify(event: dict, event_name: str) -> dict[str, str]:
    if event_name == 'workflow_dispatch':
        return {'full': 'true', 'release': 'false', 'metrics': 'true'}
    if event_name == 'pull_request':
        base = event['pull_request']['base']['sha']
        head = event['pull_request']['head']['sha']
    elif event_name == 'push':
        base, head = event['before'], event['after']
        if set(base) == {'0'}:
            return {'full': 'true', 'release': 'false', 'metrics': 'true'}
    else:
        msg = f'Unsupported event: {event_name}'
        raise ValueError(msg)
    changes = read_changes(base, head)
    only_metadata = metadata_only(changes)
    is_release = event_name == 'pull_request' and release_pr(event)
    return {
        'full': 'true',
        'release': str(is_release).lower(),
        'metrics': str(not only_metadata and metrics_relevant(list(changes))).lower(),
    }


def previous_tag(target_version: str, tags: list[str]) -> str:
    target = version_tuple(target_version)
    candidates = [tag for tag in tags if tag.startswith('v') and STABLE_VERSION.fullmatch(tag[1:])]
    candidates = [tag for tag in candidates if version_tuple(tag[1:]) < target]
    if not candidates:
        msg = f'No preceding stable release tag for {target_version}'
        raise ValueError(msg)
    return max(candidates, key=lambda tag: version_tuple(tag[1:]))


def release_baseline(event: dict, event_name: str, version: str) -> str:
    if event_name == 'pull_request':
        base = event['pull_request']['base']['sha']
        old_version = json.loads(git('show', f'{base}:.release-please-manifest.json'))['.']
        if version_tuple(version) <= version_tuple(old_version):
            msg = f'Release version {version} must exceed {old_version}'
            raise ValueError(msg)
        baseline = f'v{old_version}'
    else:
        baseline = previous_tag(version, git('tag', '--merged', 'HEAD').splitlines())
    git('rev-parse', '--verify', f'refs/tags/{baseline}^{{commit}}')
    subprocess.run(['git', 'merge-base', '--is-ancestor', baseline, 'HEAD'], check=True)
    return baseline


def validate_versions(root: Path, metadata: dict) -> str:
    version = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']['package']['version']
    version_tuple(version)
    if any(package['version'] != version for package in metadata['packages']):
        msg = 'Workspace package versions are not in lockstep'
        raise ValueError(msg)
    for package in metadata['packages']:
        for dependency in package['dependencies']:
            if dependency.get('path') and dependency['req'] not in {version, f'^{version}', f'={version}'}:
                msg = f'Internal dependency {dependency["name"]} does not require {version}'
                raise ValueError(msg)
    manifest = json.loads((root / '.release-please-manifest.json').read_text())
    if manifest['.'] != version:
        msg = 'Release manifest and Cargo version disagree'
        raise ValueError(msg)
    citation = (root / 'CITATION.cff').read_text()
    if not re.search(rf'(?m)^version:\s*[\"\']?{re.escape(version)}[\"\']?(?:\s|$)', citation):
        msg = 'Citation and Cargo version disagree'
        raise ValueError(msg)
    changelog = (root / 'CHANGELOG.md').read_text()
    first_heading = re.search(r'(?m)^##\s+(.+)$', changelog)
    if not first_heading or not first_heading[1].startswith(f'[{version}]'):
        msg = 'First changelog entry does not match the release version'
        raise ValueError(msg)
    python = tomllib.loads((root / 'python/laddu/pyproject.toml').read_text())
    if python['project']['optional-dependencies']['mpi'] != [f'laddu-mpi=={version}']:
        msg = 'Python MPI extra does not match the release version'
        raise ValueError(msg)
    for package in ('laddu', 'laddu-local', 'laddu-mpi'):
        config = tomllib.loads((root / f'python/{package}/pyproject.toml').read_text())
        if 'version' not in config['project'].get('dynamic', []):
            msg = f'{package} must derive its version from Cargo'
            raise ValueError(msg)
    version_file = root / 'version.txt'
    if version_file.exists() and version_file.read_text().strip() != version:
        msg = 'version.txt and Cargo version disagree'
        raise ValueError(msg)
    return version


def validate_release(event: dict, event_name: str, ref: str) -> tuple[str, str]:
    metadata = json.loads(
        subprocess.check_output(
            ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1'],
            text=True,
        )
    )
    version = validate_versions(Path(), metadata)
    if event_name == 'push':
        if ref != f'refs/tags/v{version}':
            msg = 'Tag does not match the release version'
            raise ValueError(msg)
        subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', 'origin/main'], check=True)
    elif event_name != 'pull_request' or not release_pr(event):
        msg = 'Release validation requires a release PR or a version tag push'
        raise ValueError(msg)
    baseline = release_baseline(event, event_name, version)
    print(f'Validate {version} against {baseline}')
    return version, baseline


def semver(baseline: str) -> None:
    current = json.loads(
        subprocess.check_output(
            ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1'],
            text=True,
        )
    )
    baseline_lock = tomllib.loads(git('show', f'{baseline}:Cargo.lock'))
    old_packages = {package['name'] for package in baseline_lock['package'] if 'source' not in package}
    packages = []
    for package in current['packages']:
        if package['publish'] == [] or package['name'] in WRAPPERS:
            continue
        if package['name'] not in old_packages:
            print(f'New crate {package["name"]}: no API compatibility baseline')
        else:
            packages.extend(['--package', package['name']])
    if packages:
        subprocess.run(
            [
                'cargo',
                'semver-checks',
                *packages,
                '--baseline-rev',
                baseline,
                '--all-features',
            ],
            check=True,
        )


def check_gate(needs: dict) -> None:
    classification = needs['changes']
    if classification['result'] != 'success':
        msg = 'Change classification failed or was cancelled'
        raise ValueError(msg)
    outputs = classification['outputs']
    if outputs.get('full') not in {'true', 'false'} or outputs.get('release') not in {'true', 'false'}:
        msg = 'Missing or invalid classification outputs'
        raise ValueError(msg)
    if outputs['full'] != 'true':
        msg = 'Full correctness checks are required for every CI run'
        raise ValueError(msg)
    expected = {
        'workflow-check': True,
        'rust': True,
        'python': True,
        'free-threaded': True,
        'release-check': outputs['release'] == 'true',
        'release-smoke': outputs['release'] == 'true',
    }
    if set(needs) != {'changes', *expected}:
        msg = 'Unexpected CI dependency set'
        raise ValueError(msg)
    for job, required in expected.items():
        expected_result = 'success' if required else 'skipped'
        if needs[job]['result'] != expected_result:
            msg = f'{job}: expected {expected_result}, got {needs[job]["result"]}'
            raise ValueError(msg)


def output(values: dict[str, str]) -> None:
    print(json.dumps(values, sort_keys=True))
    if path := os.environ.get('GITHUB_OUTPUT'):
        with Path(path).open('a') as stream:
            stream.writelines(f'{key}={value}\n' for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('classify', 'release', 'semver', 'gate'))
    parser.add_argument('--baseline')
    args = parser.parse_args()
    if args.command == 'gate':
        check_gate(json.loads(os.environ['CI_NEEDS']))
    elif args.command == 'semver':
        if not args.baseline:
            parser.error('semver requires --baseline')
        semver(args.baseline)
    else:
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        event_name = os.environ['GITHUB_EVENT_NAME']
        if args.command == 'classify':
            output(classify(event, event_name))
        else:
            version, baseline = validate_release(event, event_name, os.environ['GITHUB_REF'])
            output({'version': version, 'baseline': baseline})


if __name__ == '__main__':
    main()
