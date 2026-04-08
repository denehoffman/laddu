#!/usr/bin/env python3
"""Generate consolidated MPI rank-scaling report with bottleneck notes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Bottleneck:
    title: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Synthesize rank-scaling report from MPI memory/wall-clock and local-vs-distributed artifacts.'
        )
    )
    parser.add_argument(
        '--rank-metrics-json',
        default='target/criterion/summary/mpi_rank_metrics.json',
        help='Path to mpi_rank_metrics.json',
    )
    parser.add_argument(
        '--scaling-json',
        default='target/criterion/summary/mpi_local_vs_distributed_scaling.json',
        help='Path to mpi_local_vs_distributed_scaling.json',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/mpi_rank_scaling_report.json',
        help='Output JSON report path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/mpi_rank_scaling_report.md',
        help='Output Markdown report path.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def analyze_bottlenecks(
    rank_metrics: dict[str, Any], scaling: dict[str, Any]
) -> list[Bottleneck]:
    findings: list[Bottleneck] = []

    rank_runs = rank_metrics.get('runs', [])
    if len(rank_runs) >= 2:
        sorted_runs = sorted(rank_runs, key=lambda run: int(run['rank_count']))
        first = sorted_runs[0]
        last = sorted_runs[-1]
        first_wall = float(first['total_wall_clock_sec'])
        last_wall = float(last['total_wall_clock_sec'])
        if first_wall > 0.0:
            ratio = last_wall / first_wall
            if ratio >= 1.05:
                findings.append(
                    Bottleneck(
                        title='Limited wall-clock scaling with added ranks',
                        evidence=(
                            f'Total wall-clock increased from {first_wall:.3f}s at '
                            f'{first["rank_count"]} ranks to {last_wall:.3f}s at '
                            f'{last["rank_count"]} ranks (x{ratio:.3f}).'
                        ),
                    )
                )

        first_rss = float(first['max_rss_kb_mean'])
        last_rss = float(last['max_rss_kb_mean'])
        if first_rss > 0.0:
            rss_ratio = last_rss / first_rss
            if rss_ratio >= 1.05:
                findings.append(
                    Bottleneck(
                        title='Per-rank memory growth across rank counts',
                        evidence=(
                            f'Mean per-rank max RSS increased from {first_rss:.1f} kB at '
                            f'{first["rank_count"]} ranks to {last_rss:.1f} kB at '
                            f'{last["rank_count"]} ranks (x{rss_ratio:.3f}).'
                        ),
                    )
                )

    scaling_results = scaling.get('results', [])
    low_speedups = []
    for tier in scaling_results:
        label = str(tier.get('label', 'unknown'))
        for run in tier.get('distributed', []):
            speedup = float(run['speedup_vs_local'])
            ranks = int(run['ranks'])
            if speedup < 1.0:
                low_speedups.append((label, ranks, speedup))
    if low_speedups:
        example = low_speedups[0]
        findings.append(
            Bottleneck(
                title='Distributed run slower than local baseline for some tiers',
                evidence=(
                    f"Example: tier '{example[0]}' at {example[1]} ranks had "
                    f'speedup {example[2]:.3f}x vs local (< 1.0x).'
                ),
            )
        )

    if not findings:
        findings.append(
            Bottleneck(
                title='No clear bottleneck flags from current thresholds',
                evidence='Current artifacts did not cross the heuristic bottleneck thresholds.',
            )
        )
    return findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# MPI Rank Scaling Report',
        '',
        f'Generated: {payload["generated_at_utc"]}',
        '',
        '## Known Bottlenecks',
        '',
    ]
    for item in payload['known_bottlenecks']:
        lines.extend([f'- **{item["title"]}**: {item["evidence"]}'])

    lines.extend(
        [
            '',
            '## Inputs',
            '',
            f'- Rank metrics: `{payload["inputs"]["rank_metrics_json"]}`',
            f'- Local-vs-distributed scaling: `{payload["inputs"]["scaling_json"]}`',
        ]
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    rank_metrics_path = Path(args.rank_metrics_json)
    scaling_path = Path(args.scaling_json)
    rank_metrics = load_json(rank_metrics_path)
    scaling = load_json(scaling_path)
    bottlenecks = analyze_bottlenecks(rank_metrics=rank_metrics, scaling=scaling)

    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'rank_metrics_json': str(rank_metrics_path),
            'scaling_json': str(scaling_path),
        },
        'known_bottlenecks': [
            {'title': bottleneck.title, 'evidence': bottleneck.evidence}
            for bottleneck in bottlenecks
        ],
    }

    write_json(Path(args.json_out), payload)
    write_markdown(Path(args.md_out), payload)


if __name__ == '__main__':
    main()
