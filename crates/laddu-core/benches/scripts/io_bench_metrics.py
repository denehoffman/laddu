#!/usr/bin/env python3
"""
Run laddu-core I/O microbenchmarks and capture wall-clock + max RSS metrics.

Each benchmark filter is run via `cargo bench` under `/usr/bin/time -v`, and
the resulting metrics are written to JSON and Markdown outputs.
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
    'read_full/parquet_f32_small',
    'read_full/parquet_f64_small',
    'read_full/root_f32_small',
    'read_full/root_f64_small',
    'read_chunked_chunk_10/parquet_f32_small',
    'read_chunked_chunk_10/parquet_f64_small',
]


@dataclass
class BenchRun:
    benchmark_filter: str
    wall_clock_sec: float
    max_rss_kb: int
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect laddu-core open_benchmark wall-clock and max RSS metrics.'
    )
    parser.add_argument(
        '--bench',
        default='open_benchmark',
        help='Criterion bench target name (default: open_benchmark).',
    )
    parser.add_argument(
        '--filters',
        default=','.join(DEFAULT_FILTERS),
        help='Comma-separated benchmark filters to run.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/laddu_core_io_metrics.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/laddu_core_io_metrics.md',
        help='Output Markdown path.',
    )
    return parser.parse_args()


def repo_root() -> Path:
    # crates/laddu-core/benches/scripts -> repo root
    return Path(__file__).resolve().parents[4]


def parse_filters(raw: str) -> list[str]:
    filters = [token.strip() for token in raw.split(',') if token.strip()]
    if not filters:
        msg = 'at least one benchmark filter is required'
        raise ValueError(msg)
    return filters


def parse_probe_metrics(stdout: str) -> tuple[float, int]:
    for line in reversed(stdout.splitlines()):
        if line.startswith('__METRICS__'):
            payload = json.loads(line.removeprefix('__METRICS__'))
            return float(payload['wall_clock_sec']), int(payload['max_rss_kb'])
    msg = 'failed to parse probe metrics from benchmark output'
    raise RuntimeError(msg)


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
        command=' '.join(cargo_cmd),
    )


def write_json(path: Path, runs: list[BenchRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'runs': [
            {
                'benchmark_filter': run.benchmark_filter,
                'wall_clock_sec': run.wall_clock_sec,
                'max_rss_kb': run.max_rss_kb,
                'command': run.command,
            }
            for run in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, runs: list[BenchRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# laddu-core I/O Benchmark Metrics',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '| Benchmark Filter | Wall Clock (s) | Max RSS (kB) |',
        '| --- | ---: | ---: |',
    ]
    lines.extend(
        f'| `{run.benchmark_filter}` | {run.wall_clock_sec:.6f} | {run.max_rss_kb} |'
        for run in runs
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    root = repo_root()
    filters = parse_filters(args.filters)

    runs = [
        run_single(root, args.bench, benchmark_filter) for benchmark_filter in filters
    ]
    write_json(Path(args.json_out), runs)
    write_markdown(Path(args.md_out), runs)


if __name__ == '__main__':
    main()
