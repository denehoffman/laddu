"""Guard the decisions that allow CI work to be skipped or releases published."""

from __future__ import annotations

# These bootstrap checks deliberately use unittest, without third-party dependencies.
# ruff: noqa: PT009, PT027
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci


def event() -> dict:
    return {
        'repository': {'full_name': 'denehoffman/laddu'},
        'pull_request': {
            'base': {'ref': 'main', 'sha': 'base'},
            'head': {'ref': ci.RELEASE_BRANCH, 'sha': 'head', 'repo': {'full_name': 'denehoffman/laddu'}},
            'labels': [{'name': 'autorelease: pending'}],
        },
    }


def needs(*, full: bool, release: bool) -> dict:
    result = {'changes': {'result': 'success', 'outputs': {'full': str(full).lower(), 'release': str(release).lower()}}}
    for name in ('workflow-check', 'rust', 'python', 'free-threaded', 'release-check', 'release-smoke'):
        required = name == 'workflow-check' or (full if name in {'rust', 'python', 'free-threaded'} else release)
        result[name] = {'result': 'success' if required else 'skipped'}
    return result


class ClassificationTests(unittest.TestCase):
    def test_opened_release_pr_is_detected_before_labels_are_added(self):
        pr = event()
        pr['pull_request']['labels'] = []
        pr['pull_request']['body'] = (
            ':robot: I have created a release *beep* *boop*\n'
            'This PR was generated with [Release Please](https://github.com/googleapis/release-please).'
        )
        self.assertTrue(ci.release_pr(pr))
        pr['pull_request']['head']['repo']['full_name'] = 'fork/laddu'
        self.assertFalse(ci.release_pr(pr))

    def test_only_versions_are_ignored_in_cargo_manifest(self):
        before = '[workspace.package]\nversion="0.26.0"\nrust-version="1.96.0"\n[workspace.dependencies]\nladdu={path="crates/laddu",version="0.26.0"}\nnumpy="0.29.0"\n'
        bump = before.replace('0.26.0', '0.27.0')
        self.assertTrue(ci.metadata_only({'Cargo.toml': (before, bump)}))
        for changed in (
            bump.replace('1.96.0', '1.97.0'),
            bump.replace('0.29.0', '0.30.0'),
            bump + '\n[profile.release]\nlto=true\n',
        ):
            self.assertFalse(ci.metadata_only({'Cargo.toml': (before, changed)}))

    def test_registry_lock_changes_require_full_ci(self):
        before = 'version=4\n[[package]]\nname="laddu"\nversion="0.26.0"\n[[package]]\nname="numpy"\nversion="0.29.0"\nsource="registry"\nchecksum="abc"\n'
        bump = before.replace('0.26.0', '0.27.0')
        self.assertTrue(ci.metadata_only({'Cargo.lock': (before, bump)}))
        self.assertFalse(ci.metadata_only({'Cargo.lock': (before, bump.replace('abc', 'def'))}))

    def test_python_configuration_is_preserved(self):
        before = '[project.optional-dependencies]\nmpi=["laddu-mpi==0.26.0"]\n[build-system]\nrequires=["maturin>=1.13,<2"]\n'
        self.assertTrue(ci.metadata_only({'python/laddu/pyproject.toml': (before, before.replace('0.26.0', '0.27.0'))}))
        self.assertFalse(ci.metadata_only({'python/laddu/pyproject.toml': (before, before.replace('1.13', '1.14'))}))

    def test_unknown_deleted_added_and_invalid_files_are_conservative(self):
        for changes in (
            {},
            {'src/lib.rs': ('old', 'new')},
            {'Cargo.toml': ('invalid', 'invalid')},
            {'CHANGELOG.md': (None, 'new')},
            {'CHANGELOG.md': ('old', None)},
        ):
            self.assertFalse(ci.metadata_only(changes))

    def test_release_identity_requires_repository_branch_target_and_label(self):
        self.assertTrue(ci.release_pr(event()))
        for field, value in (('branch', 'feature'), ('repo', 'fork/laddu'), ('target', 'development'), ('labels', [])):
            modified = event()
            if field == 'branch':
                modified['pull_request']['head']['ref'] = value
            elif field == 'repo':
                modified['pull_request']['head']['repo']['full_name'] = value
            elif field == 'target':
                modified['pull_request']['base']['ref'] = value
            else:
                modified['pull_request']['labels'] = value
            self.assertFalse(ci.release_pr(modified))

    @patch.object(ci, 'read_changes')
    def test_release_changes_choose_full_or_targeted_checks(self, changes):
        changes.return_value = {'.release-please-manifest.json': ('{".":"0.26.0"}', '{".":"0.27.0"}')}
        self.assertEqual(ci.classify(event(), 'pull_request'), {'full': 'false', 'release': 'true', 'metrics': 'false'})
        changes.return_value['crates/laddu/src/lib.rs'] = ('old', 'new')
        self.assertEqual(ci.classify(event(), 'pull_request'), {'full': 'true', 'release': 'true', 'metrics': 'true'})
        ordinary = event()
        ordinary['pull_request']['head']['ref'] = 'feature'
        self.assertEqual(ci.classify(ordinary, 'pull_request')['release'], 'false')
        self.assertEqual(ci.classify(ordinary, 'pull_request')['full'], 'true')

    def test_lockfiles_and_ci_tools_trigger_metrics(self):
        for path in (
            'Cargo.lock',
            '.yamloom.py',
            'scripts/ci.py',
            '.github/workflows/metrics.yml',
            'crates/a/Cargo.toml',
        ):
            self.assertTrue(ci.metrics_relevant([path]))
        self.assertFalse(ci.metrics_relevant(['README.md', 'docs/index.rst']))

    def test_manual_checks_are_full_and_not_releases(self):
        self.assertEqual(ci.classify({}, 'workflow_dispatch'), {'full': 'true', 'release': 'false', 'metrics': 'true'})


class GateTests(unittest.TestCase):
    def test_all_supported_routes_pass(self):
        for full, release in ((True, False), (False, True), (True, True)):
            ci.check_gate(needs(full=full, release=release))

    def test_failures_cancellations_and_unexpected_skips_cannot_pass(self):
        for full, release in ((True, False), (False, True), (True, True)):
            valid = needs(full=full, release=release)
            for job, item in valid.items():
                if item['result'] != 'success':
                    continue
                for result in ('failure', 'cancelled', 'skipped'):
                    modified = copy.deepcopy(valid)
                    modified[job]['result'] = result
                    with self.assertRaises(ValueError):
                        ci.check_gate(modified)

    def test_missing_outputs_jobs_and_no_route_fail(self):
        modified = needs(full=True, release=False)
        del modified['changes']['outputs']['full']
        with self.assertRaises(ValueError):
            ci.check_gate(modified)
        modified = needs(full=True, release=False)
        del modified['rust']
        with self.assertRaises(ValueError):
            ci.check_gate(modified)
        with self.assertRaises(ValueError):
            ci.check_gate(needs(full=False, release=False))


class ReleaseTests(unittest.TestCase):
    @patch.object(ci, 'release_baseline', return_value='v0.26.0')
    @patch.object(ci, 'validate_versions', return_value='0.27.0')
    @patch.object(ci.subprocess, 'check_output', return_value='{}')
    @patch.object(ci.subprocess, 'run')
    def test_tag_version_and_main_ancestry_are_required(self, run, metadata, versions, baseline):
        self.assertEqual(ci.validate_release({}, 'push', 'refs/tags/v0.27.0'), ('0.27.0', 'v0.26.0'))
        run.assert_called_once_with(['git', 'merge-base', '--is-ancestor', 'HEAD', 'origin/main'], check=True)
        with self.assertRaises(ValueError):
            ci.validate_release({}, 'push', 'refs/tags/v0.28.0')
        with self.assertRaises(ValueError):
            ci.validate_release({}, 'workflow_dispatch', 'refs/tags/v0.27.0')
        run.side_effect = subprocess.CalledProcessError(1, 'git')
        with self.assertRaises(subprocess.CalledProcessError):
            ci.validate_release({}, 'push', 'refs/tags/v0.27.0')
        self.assertEqual(baseline.call_count, 1)
        self.assertEqual(versions.call_count, 4)
        self.assertEqual(metadata.call_count, 4)

    def test_previous_tag_uses_semver_order_and_excludes_current(self):
        self.assertEqual(
            ci.previous_tag('0.27.0', ['v0.9.0', 'v0.26.0', 'v0.27.0', 'v1.0.0', 'v0.26.1-rc.1']), 'v0.26.0'
        )
        with self.assertRaises(ValueError):
            ci.previous_tag('0.1.0', ['v0.1.0'])

    @patch.object(ci.subprocess, 'run')
    @patch.object(ci, 'git')
    def test_release_pr_baseline_comes_from_base_manifest(self, git, run):
        git.return_value = '{".":"0.26.0"}'
        self.assertEqual(ci.release_baseline(event(), 'pull_request', '0.27.0'), 'v0.26.0')
        git.assert_any_call('show', 'base:.release-please-manifest.json')
        run.assert_called_once_with(['git', 'merge-base', '--is-ancestor', 'v0.26.0', 'HEAD'], check=True)
        with self.assertRaises(ValueError):
            ci.release_baseline(event(), 'pull_request', '0.26.0')

    @patch.object(ci.subprocess, 'check_output')
    @patch.object(ci, 'git')
    @patch.object(ci.subprocess, 'run')
    def test_semver_checks_existing_publishable_crates_and_reports_new_ones(self, run, git, metadata):
        metadata.return_value = json.dumps(
            {
                'packages': [
                    {'name': name, 'publish': publish}
                    for name, publish in (
                        ('laddu', None),
                        ('laddu-new', None),
                        ('laddu-python', []),
                    )
                ]
            }
        )
        git.return_value = '[[package]]\nname="laddu"\nversion="0.26.0"\n'
        ci.semver('v0.26.0')
        run.assert_called_once_with(
            ['cargo', 'semver-checks', '--package', 'laddu', '--baseline-rev', 'v0.26.0', '--all-features'], check=True
        )

    def test_version_consistency_rejects_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Cargo.toml').write_text('[workspace.package]\nversion="0.27.0"\n')
            (root / '.release-please-manifest.json').write_text('{".":"0.27.0"}')
            (root / 'CITATION.cff').write_text('version: 0.27.0 # marker\n')
            (root / 'CHANGELOG.md').write_text('# Changelog\n\n## [0.27.0](link)\n')
            for package in ('laddu', 'laddu-local', 'laddu-mpi'):
                path = root / f'python/{package}/pyproject.toml'
                path.parent.mkdir(parents=True)
                path.write_text(
                    '[project]\ndynamic=["version"]\n[project.optional-dependencies]\nmpi=["laddu-mpi==0.27.0"]\n'
                )
            metadata = {
                'packages': [
                    {'version': '0.27.0', 'dependencies': [{'name': 'laddu', 'path': 'crates/laddu', 'req': '^0.27.0'}]}
                ]
            }
            self.assertEqual(ci.validate_versions(root, metadata), '0.27.0')
            for path, content in (
                ('CITATION.cff', 'version: 0.26.0'),
                ('.release-please-manifest.json', '{".":"0.26.0"}'),
                ('CHANGELOG.md', '## [0.26.0]'),
            ):
                original = (root / path).read_text()
                (root / path).write_text(content)
                with self.assertRaises(ValueError):
                    ci.validate_versions(root, metadata)
                (root / path).write_text(original)
            metadata['packages'][0]['dependencies'][0]['req'] = '*'
            with self.assertRaises(ValueError):
                ci.validate_versions(root, metadata)


if __name__ == '__main__':
    unittest.main()
