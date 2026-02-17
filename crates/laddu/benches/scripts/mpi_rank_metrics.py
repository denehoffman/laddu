#!/usr/bin/env python3
"""
Collect MPI rank-scaling wall-clock and per-rank RSS metrics.

This script launches MPI benchmark runs for one or more rank counts, captures:
- total wall-clock time for the full MPI run
- per-rank max RSS (kB) using `/usr/bin/time`

It emits both JSON and Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RankRun:
    rank_count: int
    total_wall_clock_sec: float
    max_rss_kb_by_rank: dict[int, int]
    max_rss_kb_min: int
    max_rss_kb_max: int
    max_rss_kb_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run MPI benchmarks for rank-count sweeps and capture wall-clock + per-rank RSS.'
        )
    )
    parser.add_argument(
        '--rank-counts',
        default='2,4',
        help='Comma-separated MPI rank counts (for example: 2,4,8).',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/mpi_rank_metrics.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/mpi_rank_metrics.md',
        help='Output Markdown path.',
    )
    parser.add_argument(
        '--bench-command',
        default=(
            'cargo bench -p laddu --features mpi '
            '--bench workflow_behavior_mpi_benchmarks -- '
            'kmatrix_nll_mpi_rank_parameterized/value_only --noplot'
        ),
        help='Command run on each MPI rank.',
    )
    return parser.parse_args()


def repo_root() -> Path:
    # crates/laddu/benches/scripts -> repo root
    return Path(__file__).resolve().parents[4]


def parse_rank_counts(raw: str) -> list[int]:
    counts = []
    for token in raw.split(','):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            msg = f'rank count must be > 0, got {value}'
            raise ValueError(msg)
        counts.append(value)
    if not counts:
        msg = 'at least one rank count is required'
        raise ValueError(msg)
    return counts


def run_single_rank_count(root: Path, rank_count: int, bench_command: str) -> RankRun:
    run_dir = root / 'target' / 'criterion' / 'mpi_metrics' / f'ranks_{rank_count}'
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_script = root / 'crates' / 'laddu' / 'benches' / 'scripts' / 'mpi_rank_probe.py'
    cmd = [
        'mpirun',
        '-n',
        str(rank_count),
        'python3',
        str(probe_script),
        '--run-dir',
        str(run_dir),
        '--command',
        bench_command,
    ]

    start = time.perf_counter()
    subprocess.run(cmd, cwd=root, check=True)
    total_wall = time.perf_counter() - start

    max_rss_kb_by_rank: dict[int, int] = {}
    for rank in range(rank_count):
        rank_path = run_dir / f'rank_{rank}.json'
        if not rank_path.exists():
            msg = f'missing per-rank RSS file for rank={rank}: {rank_path}'
            raise RuntimeError(msg)
        payload = json.loads(rank_path.read_text(encoding='utf-8'))
        max_rss_kb_by_rank[rank] = int(payload['max_rss_kb'])

    rss_values = list(max_rss_kb_by_rank.values())
    return RankRun(
        rank_count=rank_count,
        total_wall_clock_sec=total_wall,
        max_rss_kb_by_rank=max_rss_kb_by_rank,
        max_rss_kb_min=min(rss_values),
        max_rss_kb_max=max(rss_values),
        max_rss_kb_mean=statistics.fmean(rss_values),
    )


def write_json(path: Path, runs: list[RankRun], bench_command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'bench_command': bench_command,
        'runs': [
            {
                'rank_count': run.rank_count,
                'total_wall_clock_sec': run.total_wall_clock_sec,
                'max_rss_kb_by_rank': run.max_rss_kb_by_rank,
                'max_rss_kb_min': run.max_rss_kb_min,
                'max_rss_kb_max': run.max_rss_kb_max,
                'max_rss_kb_mean': run.max_rss_kb_mean,
            }
            for run in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, runs: list[RankRun], bench_command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# MPI Rank Metrics',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        f'Command: `{bench_command}`',
        '',
        '| Ranks | Total Wall Clock (s) | Min RSS (kB) | Mean RSS (kB) | Max RSS (kB) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    lines.extend(
        [
            '| '
            f'{run.rank_count} | '
            f'{run.total_wall_clock_sec:.6f} | '
            f'{run.max_rss_kb_min} | '
            f'{run.max_rss_kb_mean:.2f} | '
            f'{run.max_rss_kb_max} |'
            for run in runs
        ]
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    counts = parse_rank_counts(args.rank_counts)
    root = repo_root()

    runs = [
        run_single_rank_count(
            root=root, rank_count=count, bench_command=args.bench_command
        )
        for count in counts
    ]
    runs.sort(key=lambda run: run.rank_count)

    write_json(Path(args.json_out), runs, args.bench_command)
    write_markdown(Path(args.md_out), runs, args.bench_command)


if __name__ == '__main__':
    main()
