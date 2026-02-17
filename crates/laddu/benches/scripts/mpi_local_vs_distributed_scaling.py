#!/usr/bin/env python3
"""Capture local-vs-distributed k-matrix scaling across dataset-size tiers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DistributedRun:
    ranks: int
    wall_clock_sec: float
    speedup_vs_local: float


@dataclass
class SizeTierResult:
    label: str
    max_events: int
    local_wall_clock_sec: float
    distributed: list[DistributedRun]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run local and MPI scaling sweeps for representative dataset sizes.'
    )
    parser.add_argument(
        '--size-tiers',
        default='small:2000,medium:5000,large:10000',
        help='Comma-separated tiers: label:max_events,label:max_events.',
    )
    parser.add_argument(
        '--rank-counts',
        default='2,4',
        help='Comma-separated MPI rank counts.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/mpi_local_vs_distributed_scaling.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/mpi_local_vs_distributed_scaling.md',
        help='Output Markdown path.',
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_size_tiers(raw: str) -> list[tuple[str, int]]:
    tiers: list[tuple[str, int]] = []
    for token in raw.split(','):
        part = token.strip()
        if not part:
            continue
        label, raw_events = part.split(':', maxsplit=1)
        max_events = int(raw_events)
        if max_events <= 0:
            msg = f'max_events must be > 0 for tier {label}'
            raise ValueError(msg)
        tiers.append((label.strip(), max_events))
    if not tiers:
        msg = 'at least one dataset-size tier is required'
        raise ValueError(msg)
    return tiers


def parse_rank_counts(raw: str) -> list[int]:
    counts = [int(token.strip()) for token in raw.split(',') if token.strip()]
    if not counts:
        msg = 'at least one rank count is required'
        raise ValueError(msg)
    if any(count <= 1 for count in counts):
        msg = 'rank counts must be > 1 for distributed comparison'
        raise ValueError(msg)
    return counts


def timed_run(cmd: list[str], cwd: Path, env: dict[str, str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return time.perf_counter() - start


def run_scaling(
    root: Path, size_tiers: list[tuple[str, int]], rank_counts: list[int]
) -> list[SizeTierResult]:
    results: list[SizeTierResult] = []
    for label, max_events in size_tiers:
        env = dict(**os.environ, LADDU_BENCH_MAX_EVENTS=str(max_events))

        local_cmd = [
            'cargo',
            'bench',
            '-p',
            'laddu',
            '--bench',
            'workflow_behavior_cpu_benchmarks',
            '--',
            'kmatrix_nll_thread_scaling/value_only/1',
            '--noplot',
        ]
        local_wall = timed_run(local_cmd, cwd=root, env=env)

        distributed_runs: list[DistributedRun] = []
        for rank_count in rank_counts:
            mpi_cmd = [
                'mpirun',
                '-n',
                str(rank_count),
                'cargo',
                'bench',
                '-p',
                'laddu',
                '--features',
                'mpi',
                '--bench',
                'workflow_behavior_mpi_benchmarks',
                '--',
                'kmatrix_nll_mpi_rank_parameterized/value_only',
                '--noplot',
            ]
            distributed_wall = timed_run(mpi_cmd, cwd=root, env=env)
            distributed_runs.append(
                DistributedRun(
                    ranks=rank_count,
                    wall_clock_sec=distributed_wall,
                    speedup_vs_local=(local_wall / distributed_wall),
                )
            )

        results.append(
            SizeTierResult(
                label=label,
                max_events=max_events,
                local_wall_clock_sec=local_wall,
                distributed=distributed_runs,
            )
        )

    return results


def write_json(path: Path, results: list[SizeTierResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'results': [
            {
                'label': result.label,
                'max_events': result.max_events,
                'local_wall_clock_sec': result.local_wall_clock_sec,
                'distributed': [
                    {
                        'ranks': run.ranks,
                        'wall_clock_sec': run.wall_clock_sec,
                        'speedup_vs_local': run.speedup_vs_local,
                    }
                    for run in result.distributed
                ],
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, results: list[SizeTierResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# MPI Local vs Distributed Scaling',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '| Dataset Tier | Max Events | Local Wall Clock (s) | Ranks | Distributed Wall Clock (s) | Speedup vs Local |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    lines.extend(
        [
            '| '
            f'{result.label} | '
            f'{result.max_events} | '
            f'{result.local_wall_clock_sec:.6f} | '
            f'{run.ranks} | '
            f'{run.wall_clock_sec:.6f} | '
            f'{run.speedup_vs_local:.3f} |'
            for result in results
            for run in result.distributed
        ]
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    size_tiers = parse_size_tiers(args.size_tiers)
    rank_counts = parse_rank_counts(args.rank_counts)
    root = repo_root()

    results = run_scaling(root=root, size_tiers=size_tiers, rank_counts=rank_counts)
    write_json(Path(args.json_out), results)
    write_markdown(Path(args.md_out), results)


if __name__ == '__main__':
    main()
