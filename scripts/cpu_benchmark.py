"""
Run isolated generated CPU workflows and compare scientific baseline outputs.

The default command builds the native worker once, outside the measurements.
Linux /proc supplies per-process peak RSS and a sampled RSS watchdog.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = {'quick': (128, 512), 'small': (2_000, 20_000), 'reference': (20_000, 200_000)}
WORKFLOWS = ('likelihood-5', 'likelihood-12', 'bootstrap', 'parquet')
# Calibration must not turn materially unstable repeats into a broad allowance.
MAX_REPEAT_ROUNDOFF = 1e-12


def invalid(message):
    raise ValueError(message)


def load_artifact(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def save_artifact(path, artifact):
    data = (json.dumps(artifact, indent=None if path.suffix == '.gz' else 2, allow_nan=False) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    temporary.replace(path)


def capture(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()  # noqa: S603


def numeric_fields(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for name, child in sorted(value.items()):
            result.update(numeric_fields(child, f'{prefix}.{name}' if prefix else name))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(numeric_fields(child, f'{prefix}[{index}]'))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        invalid(f'nonfinite or nonnumeric scientific output at {prefix}')
    return {prefix: float(value)}


def identity(config, *, include_threads=True):
    excluded = {'repeat'} if include_threads else {'repeat', 'threads'}
    return json.dumps({key: value for key, value in config.items() if key not in excluded}, sort_keys=True)


def groups(artifact):
    grouped = {}
    for run in artifact['runs']:
        grouped.setdefault(identity(run['config']), []).append(run)
    return grouped


def calibrate(runs):
    """Calibrate measured baseline variation across repeats and thread counts."""
    flattened = [numeric_fields(run['science']) for run in runs]
    if any(set(row) != set(flattened[0]) for row in flattened):
        invalid('baseline scientific output shapes differ')
    return {
        field: {
            'spread': max(row[field] for row in flattened) - min(row[field] for row in flattened),
            'absolute_tolerance': max(
                4 * (max(row[field] for row in flattened) - min(row[field] for row in flattened)),
                32 * math.ulp(max(abs(row[field]) for row in flattened)),
            ),
        }
        for field in flattened[0]
    }


def performance_metrics(successful, measured):
    performance = {}
    metrics = {'total_seconds', 'peak_rss_bytes'} | set(successful[0]['stages_seconds'])
    for metric in sorted(metrics):

        def metric_value(run, name=metric):
            return run[name] if name in ('total_seconds', 'peak_rss_bytes') else run['stages_seconds'][name]

        old = statistics.median(metric_value(run) for run in successful)
        new = statistics.median(metric_value(run) for run in measured)
        performance[metric] = {'before': old, 'after': new, 'ratio': new / old if old else None}
    return performance


def validate_comparison(before, after):
    if before['schema_version'] != 1 or after['schema_version'] != 1:
        invalid('unsupported benchmark schema')
    if not before['runs'] or not after['runs']:
        invalid('empty benchmark artifact')
    for key in ('machine', 'build'):
        if before[key] != after[key]:
            invalid(f'{key} differs; compare on the same machine and build settings')
    before_groups, after_groups = groups(before), groups(after)
    if before_groups.keys() != after_groups.keys():
        invalid('case configurations differ; retain an identical smaller-size comparison after a memory failure')
    return before_groups, after_groups


def repeat_failures(runs, artifact_name):
    if not runs:
        return []
    fields = [numeric_fields(run['science']) for run in runs]
    if any(row.keys() != fields[0].keys() for row in fields):
        return [{'field': 'science', 'artifact': artifact_name, 'reason': 'repeat output shape differs'}]
    failures = []
    for field in fields[0]:
        values = [row[field] for row in fields]
        spread = max(values) - min(values)
        bound = MAX_REPEAT_ROUNDOFF * max(1.0, *(abs(value) for value in values))
        if spread > bound:
            failures.append(
                {
                    'field': 'repeat_variation',
                    'quantity': field,
                    'artifact': artifact_name,
                    'spread': spread,
                    'roundoff_bound': bound,
                }
            )
    return failures


def summarize(artifact):
    cases = []
    for rows in groups(artifact).values():
        successful = [run for run in rows if run['status'] == 'ok']
        case = {'config': rows[0]['config'], 'statuses': [run['status'] for run in rows]}
        if successful:
            fields = [numeric_fields(run['science']) for run in successful]
            case.update(
                {
                    'median_total_seconds': statistics.median(run['total_seconds'] for run in successful),
                    'max_peak_rss_bytes': max(run['peak_rss_bytes'] for run in successful),
                    'median_stages_seconds': {
                        name: statistics.median(run['stages_seconds'][name] for run in successful)
                        for name in successful[0]['stages_seconds']
                    },
                    'fixed_configuration_exact': all(run['science'] == successful[0]['science'] for run in successful)
                    if len(successful) > 1
                    else None,
                    'repeat_roundoff_acceptable': not repeat_failures(successful, 'summary')
                    if len(successful) > 1
                    else None,
                    'max_repeat_absolute_spread': max(
                        max(row[field] for row in fields) - min(row[field] for row in fields) for field in fields[0]
                    ),
                }
            )
        cases.append(case)
    return {'schema_version': 1, 'machine': artifact['machine'], 'build': artifact['build'], 'cases': cases}


def science_failures(reference, measured, tolerances):
    failures = []
    for run in measured:
        values = numeric_fields(run['science'])
        if values.keys() != reference.keys():
            failures.append({'field': 'science', 'reason': 'output shape differs'})
            continue
        for field, value in values.items():
            delta = abs(value - reference[field])
            if delta > tolerances[field]['absolute_tolerance']:
                failures.append({'field': field, 'difference': delta, **tolerances[field]})
    return failures


def compare(before, after):
    before_groups, after_groups = validate_comparison(before, after)
    cases = []
    equivalent = True
    for key, baseline in before_groups.items():
        candidate = after_groups[key]
        failures = []
        successful = [run for run in baseline if run['status'] == 'ok']
        measured = [run for run in candidate if run['status'] == 'ok']
        if len(successful) != len(baseline) or len(measured) != len(candidate):
            failures.append({'field': 'status', 'reason': 'incomplete or memory-limited workflow'})
        failures.extend(repeat_failures(successful, 'before'))
        failures.extend(repeat_failures(measured, 'after'))
        if successful and measured:
            config = successful[0]['config']
            variations = [
                run
                for run in before['runs']
                if run['status'] == 'ok'
                and identity(run['config'], include_threads=False) == identity(config, include_threads=False)
            ]
            tolerances = calibrate(variations)
            reference = numeric_fields(successful[0]['science'])
            failures.extend(science_failures(reference, measured, tolerances))
            performance = performance_metrics(successful, measured)
        else:
            performance = {}
        equivalent &= not failures
        cases.append({'config': baseline[0]['config'], 'failures': failures, 'performance': performance})
    return {
        'schema_version': 1,
        'comparison_policy': 'baseline-roundoff',
        'max_repeat_roundoff': MAX_REPEAT_ROUNDOFF,
        'scientific_equivalence': equivalent,
        'cases': cases,
    }


def read_peak(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
    except FileNotFoundError:
        return 0
    return max((int(line.split()[1]) * 1024 for line in lines if line.startswith(('VmRSS:', 'VmHWM:'))), default=0)


def execute(binary, config, memory_limit):
    with tempfile.TemporaryDirectory(prefix='laddu-cpu-') as temporary:
        directory = Path(temporary)
        config_path = directory / 'config.json'
        config_path.write_text(json.dumps(config))
        # File-backed output cannot deadlock the worker on a filled pipe. Preserve
        # completed stage records when the RSS watchdog interrupts a workflow.
        with (directory / 'stdout').open('w+') as stdout, (directory / 'stderr').open('w+') as stderr:
            started = time.perf_counter()
            process = subprocess.Popen(  # noqa: S603 - explicitly selected local benchmark worker
                [str(binary), str(config_path), str(directory / 'parquet')],
                stdout=stdout,
                stderr=stderr,
                cwd=ROOT,
            )
            peak = 0
            limited = False
            try:
                while process.poll() is None:
                    peak = max(peak, read_peak(process.pid))
                    if peak > memory_limit:
                        limited = True
                        process.kill()
                        break
                    time.sleep(0.01)
                process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            elapsed = time.perf_counter() - started
            stdout.seek(0)
            # SIGKILL can leave a partial final JSON line; earlier completed
            # stages remain usable in the controlled-failure artifact.
            records = [json.loads(line) for line in stdout if line.endswith('\n') and line.strip()]
            stderr.seek(0)
            errors = stderr.read()[-4000:]
        stages = [record for record in records if 'stage' in record]
        results = [record['result'] for record in records if 'result' in record]
        if process.returncode == 0 and len(results) == 1:
            result = results[0]
            peak = max(peak, result['peak_rss_bytes'])
            limited |= peak > memory_limit
        else:
            result = {'config': config}
        return {
            **result,
            'status': 'memory_limit' if limited else 'ok' if results and process.returncode == 0 else 'failed',
            'peak_rss_bytes': peak,
            'process_seconds': elapsed,
            'stage_records': stages,
            'returncode': process.returncode,
            'stderr': errors,
        }


def run_matrix(args, plan):
    if sys.platform != 'linux':
        invalid('RSS monitoring currently requires Linux /proc')
    binary = args.binary.resolve()
    if not args.no_build:
        subprocess.run(
            [  # noqa: S607 - use the developer's configured Cargo toolchain
                'cargo',
                'build',
                '--release',
                '-p',
                'laddu',
                '--example',
                'cpu_workflow',
                '--no-default-features',
                '--features',
                'fit',
            ],
            cwd=ROOT,
            check=True,
        )
    if not binary.is_file():
        invalid(f'worker missing: {binary}')
    artifact = {
        'schema_version': 1,
        'machine': {
            'platform': platform.platform(),
            'hostname': platform.node(),
            'cpu': next(
                (
                    line.split(':', 1)[1].strip()
                    for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')
                ),
                platform.processor(),
            ),
            'cpu_count': os.cpu_count(),
        },
        'build': {'profile': 'release', 'features': 'fit', 'rustc': capture('rustc', '--version')},
        'revision': capture('git', 'rev-parse', 'HEAD'),
        'working_tree': capture('git', 'status', '--porcelain'),
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan': plan,
        'runs': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for config in plan['cases']:
        print(
            f'{config["size"]} {config["workflow"]} threads={config["threads"]} repeat={config["repeat"]}',
            file=sys.stderr,
        )
        artifact['runs'].append(execute(binary, config, plan['memory_limit_bytes']))
        # Save after every case so interruption doesn't discard smaller results.
        save_artifact(args.output, artifact)
    return all(run['status'] == 'ok' for run in artifact['runs'])


def make_plan(args):
    cases = []
    for size in args.sizes.split(','):
        data, mc = SIZES[size]
        for workflow in ('bootstrap',) if args.stress else WORKFLOWS:
            for threads in (4, 1) if args.variation else (4,):
                cases.extend(
                    {
                        'size': size,
                        'workflow': workflow,
                        'data_events': data,
                        'mc_events': mc,
                        'waves': 5 if workflow == 'likelihood-5' else 12,
                        'replicas': 200 if args.stress else 8,
                        'threads': threads,
                        'repeat': repeat,
                        'seed': 20261002,
                        'evaluations': 3 if size == 'quick' else 10,
                        'fit_steps': 4 if size == 'quick' else 30,
                        'periods': 4 if workflow == 'parquet' else 1,
                        'shards_per_period': 4,
                        'bins': 16,
                    }
                    for repeat in range(args.repeats)
                )
    return {'schema_version': 1, 'memory_limit_bytes': int(args.memory_limit_gb * 1_000_000_000), 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', action='store_true', help='print frozen cases without building or running')
    run.add_argument('--sizes', default='small,reference')
    run.add_argument('--repeats', type=int, default=3)
    run.add_argument('--variation', action='store_true', help='also measure one CPU thread')
    run.add_argument('--stress', action='store_true', help='only the 200-replica bootstrap workflow')
    run.add_argument('--memory-limit-gb', type=float, default=28.0)
    run.add_argument('--output', type=Path, default=ROOT / 'target/cpu-benchmark/results.json')
    run.add_argument('--binary', type=Path, default=ROOT / 'target/release/examples/cpu_workflow')
    run.add_argument('--no-build', action='store_true', help='reuse a worker already built by this checkout')
    comparison = commands.add_parser('compare')
    comparison.add_argument('before', type=Path)
    comparison.add_argument('after', type=Path)
    summary = commands.add_parser('summary')
    summary.add_argument('artifact', type=Path)
    args = parser.parse_args()
    if args.command == 'summary':
        try:
            print(json.dumps(summarize(load_artifact(args.artifact)), indent=2, allow_nan=False))
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        return 0
    if args.command == 'compare':
        try:
            report = compare(load_artifact(args.before), load_artifact(args.after))
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['scientific_equivalence'] else 1
    if (
        args.repeats < 1
        or not math.isfinite(args.memory_limit_gb)
        or args.memory_limit_gb <= 0
        or any(size not in SIZES for size in args.sizes.split(','))
    ):
        parser.error('use positive repeats/memory limit and sizes from quick,small,reference')
    plan = make_plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        return 0 if run_matrix(args, plan) else 1
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
