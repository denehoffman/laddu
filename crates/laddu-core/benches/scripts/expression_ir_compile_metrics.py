#!/usr/bin/env python3
"""
Collect wall-clock benchmark timing together with staged expression-IR compile metrics.

This script joins Criterion timing for the `expression_ir_compile_costs` benchmark group with
JSON emitted by the `expression_ir_compile_probe` helper binary so compile-stage regressions can
be reviewed separately from steady-state runtime wins.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

DEFAULT_CASES: Final[list[str]] = [
    'initial_load/separable',
    'initial_load/partial',
    'initial_load/non_separable',
    'specialization_cache_miss/partial',
    'specialization_cache_hit_restore/partial',
]


@dataclass
class CompileRun:
    benchmark_filter: str
    wall_clock_sec: float
    command: str
    compile_metrics: dict[str, int]
    specialization_metrics: dict[str, int]
    probe_command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect laddu-core expression-IR compile benchmark metrics.'
    )
    parser.add_argument(
        '--bench',
        default='expression_ir_benchmarks',
        help='Criterion bench target name (default: expression_ir_benchmarks).',
    )
    parser.add_argument(
        '--cases',
        default=','.join(DEFAULT_CASES),
        help='Comma-separated benchmark filters to run.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/laddu_core_expression_ir_compile_metrics.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/laddu_core_expression_ir_compile_metrics.md',
        help='Output Markdown path.',
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_cases(raw: str) -> list[str]:
    cases = [token.strip() for token in raw.split(',') if token.strip()]
    if not cases:
        msg = 'at least one compile benchmark case is required'
        raise ValueError(msg)
    return cases


def parse_probe_metrics(stdout: str) -> float:
    for line in reversed(stdout.splitlines()):
        if line.startswith('__METRICS__'):
            payload = json.loads(line.removeprefix('__METRICS__'))
            return float(payload['wall_clock_sec'])
    msg = 'failed to parse wall-clock probe metrics from benchmark output'
    raise RuntimeError(msg)


def parse_case(case: str) -> tuple[str, str]:
    try:
        operation, scenario = case.split('/', maxsplit=1)
    except ValueError as exc:
        msg = f'invalid case format: {case}'
        raise ValueError(msg) from exc
    return operation, scenario


def run_benchmark_case(
    root: Path, bench: str, benchmark_filter: str
) -> tuple[float, str]:
    cargo_cmd = [
        'cargo',
        'bench',
        '-p',
        'laddu-core',
        '--features',
        'expression-ir',
        '--bench',
        bench,
        '--',
        benchmark_filter,
        '--noplot',
    ]
    probe = (
        'import json, subprocess, sys, time\n'
        'cmd = sys.argv[1:]\n'
        'start = time.perf_counter()\n'
        'subprocess.run(cmd, check=True)\n'
        'wall = time.perf_counter() - start\n'
        "print('__METRICS__' + json.dumps({'wall_clock_sec': wall}))\n"
    )
    cmd = [sys.executable, '-c', probe, *cargo_cmd]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_probe_metrics(proc.stdout), ' '.join(cargo_cmd)


def run_compile_probe(
    root: Path, operation: str, scenario: str
) -> tuple[dict[str, int], dict[str, int], str]:
    cargo_cmd = [
        'cargo',
        'run',
        '-q',
        '-p',
        'laddu-core',
        '--features',
        'expression-ir',
        '--bin',
        'expression_ir_compile_probe',
        '--',
        operation,
        scenario,
    ]
    proc = subprocess.run(
        cargo_cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    return (
        dict(payload['compile_metrics']),
        dict(payload['specialization_metrics']),
        ' '.join(cargo_cmd),
    )


def run_single(root: Path, bench: str, benchmark_filter: str) -> CompileRun:
    operation, scenario = parse_case(benchmark_filter)
    wall_clock_sec, command = run_benchmark_case(root, bench, benchmark_filter)
    compile_metrics, specialization_metrics, probe_command = run_compile_probe(
        root, operation, scenario
    )
    return CompileRun(
        benchmark_filter=benchmark_filter,
        wall_clock_sec=wall_clock_sec,
        command=command,
        compile_metrics=compile_metrics,
        specialization_metrics=specialization_metrics,
        probe_command=probe_command,
    )


def write_json(path: Path, runs: list[CompileRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'runs': [
            {
                'benchmark_filter': run.benchmark_filter,
                'wall_clock_sec': run.wall_clock_sec,
                'command': run.command,
                'probe_command': run.probe_command,
                'compile_metrics': run.compile_metrics,
                'specialization_metrics': run.specialization_metrics,
            }
            for run in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, runs: list[CompileRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# laddu-core Expression-IR Compile Metrics',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '| Benchmark Filter | Wall Clock (s) | IR Compile (ns) | Cached Integrals (ns) | Lowering (ns) | Cache Hits | Cache Misses | Lowering Cache Hits | Lowering Cache Misses |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for run in runs:
        metrics = run.compile_metrics
        lines.append(
            f'| `{run.benchmark_filter}` | {run.wall_clock_sec:.6f} | '
            f'{metrics.get("initial_ir_compile_nanos", 0) + metrics.get("specialization_ir_compile_nanos", 0)} | '
            f'{metrics.get("initial_cached_integrals_nanos", 0) + metrics.get("specialization_cached_integrals_nanos", 0)} | '
            f'{metrics.get("initial_lowering_nanos", 0) + metrics.get("specialization_lowering_nanos", 0)} | '
            f'{metrics.get("specialization_cache_hits", 0)} | '
            f'{metrics.get("specialization_cache_misses", 0)} | '
            f'{metrics.get("specialization_lowering_cache_hits", 0)} | '
            f'{metrics.get("specialization_lowering_cache_misses", 0)} |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    root = repo_root()
    cases = parse_cases(args.cases)
    runs = [run_single(root, args.bench, case) for case in cases]
    write_json(Path(args.json_out), runs)
    write_markdown(Path(args.md_out), runs)


if __name__ == '__main__':
    main()
