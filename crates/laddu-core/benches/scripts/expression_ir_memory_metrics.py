#!/usr/bin/env python3
"""
Run laddu-core expression-IR memory workloads and capture wall-clock + max RSS metrics.

Each benchmark filter is run via `cargo bench` under a Python probe that records wall-clock
and `ru_maxrss`. The report also includes an explicit allocation proxy per benchmark filter so
memory-trend reports can distinguish measured peak RSS from workload-sized allocation pressure.
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

DEFAULT_FILTERS: Final[list[str]] = [
    'gradient_local_large/large_gradient@4096x32',
    'value_gradient_local_large/large_gradient@4096x32',
    'activation_cycle_large/partial@32768',
]

COMPLEX64_BYTES: Final[int] = 16


@dataclass
class BenchRun:
    benchmark_filter: str
    wall_clock_sec: float
    max_rss_kb: int
    allocation_proxy_bytes_per_iter: int
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect laddu-core expression-IR memory workload metrics.'
    )
    parser.add_argument(
        '--bench',
        default='expression_ir_benchmarks',
        help='Criterion bench target name (default: expression_ir_benchmarks).',
    )
    parser.add_argument(
        '--filters',
        default=','.join(DEFAULT_FILTERS),
        help='Comma-separated benchmark filters to run.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/laddu_core_expression_ir_memory_metrics.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/laddu_core_expression_ir_memory_metrics.md',
        help='Output Markdown path.',
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_filters(raw: str) -> list[str]:
    filters = [token.strip() for token in raw.split(',') if token.strip()]
    if not filters:
        msg = 'at least one benchmark filter is required'
        raise ValueError(msg)
    return filters


def build_benchmark_target(root: Path, bench: str) -> None:
    cargo_cmd = [
        'cargo',
        'bench',
        '-p',
        'laddu-core',
        '--bench',
        bench,
        '--no-run',
    ]
    subprocess.run(
        cargo_cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def parse_probe_metrics(stdout: str) -> tuple[float, int]:
    for line in reversed(stdout.splitlines()):
        if line.startswith('__METRICS__'):
            payload = json.loads(line.removeprefix('__METRICS__'))
            return float(payload['wall_clock_sec']), int(payload['max_rss_kb'])
    msg = 'failed to parse probe metrics from benchmark output'
    raise RuntimeError(msg)


def allocation_proxy_bytes_per_iter(benchmark_filter: str) -> int:
    if benchmark_filter == 'gradient_local_large/large_gradient@4096x32':
        return 4096 * 32 * COMPLEX64_BYTES
    if benchmark_filter == 'value_gradient_local_large/large_gradient@4096x32':
        return 4096 * (32 + 1) * COMPLEX64_BYTES
    if benchmark_filter == 'activation_cycle_large/partial@32768':
        return 0
    msg = f'unknown benchmark filter for allocation proxy: {benchmark_filter}'
    raise ValueError(msg)


def run_single(root: Path, bench: str, benchmark_filter: str) -> BenchRun:
    cargo_cmd = [
        'cargo',
        'bench',
        '-p',
        'laddu-core',
        '--bench',
        bench,
        '--',
        benchmark_filter,
        '--noplot',
    ]
    probe = (
        'import json, resource, subprocess, sys, time\n'
        'cmd = sys.argv[1:]\n'
        'start = time.perf_counter()\n'
        'subprocess.run(cmd, check=True)\n'
        'wall = time.perf_counter() - start\n'
        'rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n'
        "print('__METRICS__' + json.dumps({'wall_clock_sec': wall, 'max_rss_kb': rss}))\n"
    )
    cmd = [sys.executable, '-c', probe, *cargo_cmd]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wall_clock, max_rss_kb = parse_probe_metrics(proc.stdout)
    return BenchRun(
        benchmark_filter=benchmark_filter,
        wall_clock_sec=wall_clock,
        max_rss_kb=max_rss_kb,
        allocation_proxy_bytes_per_iter=allocation_proxy_bytes_per_iter(benchmark_filter),
        command=' '.join(cargo_cmd),
    )


def write_json(path: Path, runs: list[BenchRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'allocation_proxy_note': (
            'allocation_proxy_bytes_per_iter is a workload-sized logical output proxy, not a '
            'measured allocator statistic.'
        ),
        'runs': [
            {
                'benchmark_filter': run.benchmark_filter,
                'wall_clock_sec': run.wall_clock_sec,
                'max_rss_kb': run.max_rss_kb,
                'allocation_proxy_bytes_per_iter': run.allocation_proxy_bytes_per_iter,
                'command': run.command,
            }
            for run in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, runs: list[BenchRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# laddu-core Expression-IR Memory Workload Metrics',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        'Allocation proxy note: `allocation_proxy_bytes_per_iter` is a workload-sized logical '
        'output proxy, not a measured allocator statistic.',
        '',
        '| Benchmark Filter | Wall Clock (s) | Max RSS (kB) | Allocation Proxy (bytes/iter) |',
        '| --- | ---: | ---: | ---: |',
    ]
    lines.extend(
        f'| `{run.benchmark_filter}` | {run.wall_clock_sec:.6f} | {run.max_rss_kb} | '
        f'{run.allocation_proxy_bytes_per_iter} |'
        for run in runs
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    root = repo_root()
    filters = parse_filters(args.filters)
    build_benchmark_target(root, args.bench)
    runs = [
        run_single(root, args.bench, benchmark_filter) for benchmark_filter in filters
    ]
    write_json(Path(args.json_out), runs)
    write_markdown(Path(args.md_out), runs)


if __name__ == '__main__':
    main()
