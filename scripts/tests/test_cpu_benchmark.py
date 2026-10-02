"""Contracts for the local CPU workflow benchmark commands."""
# ruff: noqa: S101, PLR2004

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / 'cpu_benchmark.py'


def command(*args):
    return subprocess.run([sys.executable, str(RUNNER), *args], capture_output=True, text=True, check=False)  # noqa: S603


def test_reference_plan_freezes_three_workflows_and_real_replica_counts():
    result = command('run', '--plan', '--sizes', 'small,reference', '--repeats', '2', '--variation')
    assert result.returncode == 0, result.stderr

    plan = json.loads(result.stdout)
    reference = [case for case in plan['cases'] if case['size'] == 'reference' and case['threads'] == 4]
    assert {case['workflow'] for case in reference} == {'likelihood-5', 'likelihood-12', 'bootstrap', 'parquet'}
    assert all(case['data_events'] == 20_000 and case['mc_events'] == 200_000 for case in reference)
    assert all(case['replicas'] == 8 and case['seed'] == 20261002 for case in reference)
    assert any(case['threads'] == 1 for case in plan['cases'])
    stress = command('run', '--plan', '--sizes', 'reference', '--stress', '--repeats', '1')
    assert stress.returncode == 0, stress.stderr
    assert json.loads(stress.stdout)['cases'][0]['replicas'] == 200


def test_comparison_never_accepts_an_empty_baseline(tmp_path):
    baseline = artifact(100.0, 2.0)
    baseline['runs'] = []
    path = tmp_path / 'empty.json'
    path.write_text(json.dumps(baseline))
    result = command('compare', str(path), str(path))
    assert result.returncode == 2
    assert 'empty' in result.stderr


def test_large_repeat_drift_cannot_widen_scientific_tolerances(tmp_path):
    baseline = artifact(100.0, 2.0)
    different_repeat = artifact(101.0, 2.0)['runs'][0]
    different_repeat['config']['repeat'] = 1
    baseline['runs'].append(different_repeat)
    path = tmp_path / 'nondeterministic.json'
    path.write_text(json.dumps(baseline))
    result = command('compare', str(path), str(path))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report['scientific_equivalence'] is False
    assert report['cases'][0]['failures'][0]['field'] == 'repeat_variation'


def test_parallel_roundoff_is_accepted_using_only_baseline_variation(tmp_path):
    baseline = artifact(100.0, 2.0)
    repeated = artifact(100.00000000000003, 2.0)['runs'][0]
    repeated['config']['repeat'] = 1
    baseline['runs'].append(repeated)
    before = tmp_path / 'roundoff.json'
    after = tmp_path / 'after.json'
    before.write_text(json.dumps(baseline))
    after.write_text(json.dumps(artifact(100.00000000000004, 1.0)))
    result = command('compare', str(before), str(after))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['scientific_equivalence'] is True
    after.write_text(json.dumps(artifact(100.000001, 1.0)))
    result = command('compare', str(before), str(after))
    assert result.returncode == 1
    assert json.loads(result.stdout)['scientific_equivalence'] is False


def artifact(value, seconds):
    return {
        'schema_version': 1,
        'machine': {'platform': 'test', 'cpu': 'test', 'cpu_count': 4},
        'build': {'profile': 'release', 'features': 'fit', 'rustc': 'test'},
        'runs': [
            {
                'status': 'ok',
                'config': {'size': 'small', 'workflow': 'bootstrap', 'threads': 4, 'repeat': 0, 'seed': 17},
                'science': {'cross_section': {'value': value, 'bootstrap_std': 0.25}},
                'stages_seconds': {'cross_section_total': seconds},
                'total_seconds': seconds,
                'peak_rss_bytes': 1024,
            }
        ],
    }


def test_comparison_rejects_scientific_drift_and_reports_stage_improvement(tmp_path):
    baseline = artifact(100.0, 2.0)
    baseline['runs'].append({**baseline['runs'][0], 'config': {**baseline['runs'][0]['config'], 'repeat': 1}})
    before = tmp_path / 'before.json'
    after = tmp_path / 'after.json'
    before.write_text(json.dumps(baseline))
    after.write_text(json.dumps(artifact(101.0, 1.0)))
    result = command('compare', str(before), str(after))
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report['scientific_equivalence'] is False
    assert report['cases'][0]['failures'][0]['field'] == 'cross_section.value'
    assert report['cases'][0]['performance']['cross_section_total']['ratio'] == 0.5
    after.write_text(json.dumps(artifact(100.0, 1.0)))
    result = command('compare', str(before), str(after))
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif('LADDU_CPU_WORKER' not in os.environ, reason='opt-in native workflow correctness check')
def test_native_quick_workflows_keep_paired_replicas_and_mass_bin_rows(tmp_path):
    output = tmp_path / 'quick.json'
    result = command(
        'run',
        '--sizes',
        'quick',
        '--repeats',
        '1',
        '--no-build',
        '--binary',
        os.environ['LADDU_CPU_WORKER'],
        '--output',
        str(output),
    )
    assert result.returncode == 0, result.stderr
    runs = json.loads(output.read_text())['runs']
    assert all(run['status'] == 'ok' and run['peak_rss_bytes'] > 0 for run in runs)
    bootstrap = next(run for run in runs if run['config']['workflow'] == 'bootstrap')
    assert bootstrap['diagnostics']['pairing'] == {'replicas': 8, 'draws': 8, 'bootstrap_seed': 20261102}
    assert len(bootstrap['science']['replica_parameters']) == 8
    assert len(bootstrap['science']['cross_section']['draws']) == 8
    assert bootstrap['science']['cross_section']['bootstrap_std'] > 0
    assert len(bootstrap['science']['projection_mass_wave_11']['values']) == 16
    parquet = next(run for run in runs if run['config']['workflow'] == 'parquet')
    assert [sum(parquet['science'][f'period_{period}_bin_events']) for period in range(4)] == [512] * 4
    limited = command(
        'run',
        '--sizes',
        'quick',
        '--repeats',
        '1',
        '--no-build',
        '--binary',
        os.environ['LADDU_CPU_WORKER'],
        '--memory-limit-gb',
        '0.001',
        '--output',
        str(tmp_path / 'limited.json'),
    )
    assert limited.returncode == 1
    assert all(run['status'] == 'memory_limit' for run in json.loads((tmp_path / 'limited.json').read_text())['runs'])
