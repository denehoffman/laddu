#!/usr/bin/env python3
"""
Compare laddu-core I/O benchmark metric artifacts against a baseline.

Inputs are JSON files produced by `io_bench_metrics.py`.
Outputs include:
- machine-readable comparison JSON
- human-readable Markdown summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkDelta:
    benchmark_filter: str
    baseline_wall_clock_sec: float
    candidate_wall_clock_sec: float
    wall_clock_change_pct: float
    baseline_max_rss_kb: int
    candidate_max_rss_kb: int
    max_rss_change_pct: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare laddu-core I/O benchmark metrics to a baseline.'
    )
    parser.add_argument(
        '--baseline',
        required=True,
        help='Baseline metrics JSON path.',
    )
    parser.add_argument(
        '--candidate',
        required=True,
        help='Candidate metrics JSON path.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/laddu_core_io_regression.json',
        help='Output JSON report path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/laddu_core_io_regression.md',
        help='Output Markdown report path.',
    )
    parser.add_argument(
        '--wall-regress-threshold-pct',
        type=float,
        default=5.0,
        help='Regression threshold for wall-clock percent increase.',
    )
    parser.add_argument(
        '--rss-regress-threshold-pct',
        type=float,
        default=5.0,
        help='Regression threshold for RSS percent increase.',
    )
    parser.add_argument(
        '--improve-threshold-pct',
        type=float,
        default=5.0,
        help='Improvement threshold for wall-clock percent decrease.',
    )
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with code 1 if any benchmark is regressed.',
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    runs = payload.get('runs', [])
    result: dict[str, dict[str, float | int]] = {}
    for run in runs:
        bench = str(run['benchmark_filter'])
        result[bench] = {
            'wall_clock_sec': float(run['wall_clock_sec']),
            'max_rss_kb': int(run['max_rss_kb']),
        }
    return result


def pct_change(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0 if new == 0.0 else 100.0
    return ((new - old) / old) * 100.0


def classify_status(
    wall_change_pct: float,
    rss_change_pct: float,
    wall_regress_threshold_pct: float,
    rss_regress_threshold_pct: float,
    improve_threshold_pct: float,
) -> str:
    if (
        wall_change_pct >= wall_regress_threshold_pct
        or rss_change_pct >= rss_regress_threshold_pct
    ):
        return 'regressed'
    if wall_change_pct <= -improve_threshold_pct:
        return 'improved'
    return 'unchanged'


def compare(
    baseline: dict[str, dict[str, float | int]],
    candidate: dict[str, dict[str, float | int]],
    wall_regress_threshold_pct: float,
    rss_regress_threshold_pct: float,
    improve_threshold_pct: float,
) -> list[BenchmarkDelta]:
    all_ids = sorted(set(baseline.keys()) | set(candidate.keys()))
    deltas: list[BenchmarkDelta] = []
    for bench_id in all_ids:
        if bench_id not in baseline or bench_id not in candidate:
            continue
        base_wall = float(baseline[bench_id]['wall_clock_sec'])
        cand_wall = float(candidate[bench_id]['wall_clock_sec'])
        base_rss = int(baseline[bench_id]['max_rss_kb'])
        cand_rss = int(candidate[bench_id]['max_rss_kb'])
        wall_change = pct_change(base_wall, cand_wall)
        rss_change = pct_change(float(base_rss), float(cand_rss))
        status = classify_status(
            wall_change,
            rss_change,
            wall_regress_threshold_pct,
            rss_regress_threshold_pct,
            improve_threshold_pct,
        )
        deltas.append(
            BenchmarkDelta(
                benchmark_filter=bench_id,
                baseline_wall_clock_sec=base_wall,
                candidate_wall_clock_sec=cand_wall,
                wall_clock_change_pct=wall_change,
                baseline_max_rss_kb=base_rss,
                candidate_max_rss_kb=cand_rss,
                max_rss_change_pct=rss_change,
                status=status,
            )
        )
    return deltas


def write_json(
    path: Path, deltas: list[BenchmarkDelta], args: argparse.Namespace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline': args.baseline,
        'candidate': args.candidate,
        'wall_regress_threshold_pct': args.wall_regress_threshold_pct,
        'rss_regress_threshold_pct': args.rss_regress_threshold_pct,
        'improve_threshold_pct': args.improve_threshold_pct,
        'deltas': [delta.__dict__ for delta in deltas],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(
    path: Path, deltas: list[BenchmarkDelta], args: argparse.Namespace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# laddu-core I/O Benchmark Regression Report',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        f'Baseline: `{args.baseline}`',
        f'Candidate: `{args.candidate}`',
        '',
        '| Benchmark | Baseline Wall (s) | Candidate Wall (s) | Wall Δ (%) | Baseline RSS (kB) | Candidate RSS (kB) | RSS Δ (%) | Status |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    lines.extend(
        (
            '| '
            f'`{delta.benchmark_filter}` | '
            f'{delta.baseline_wall_clock_sec:.6f} | '
            f'{delta.candidate_wall_clock_sec:.6f} | '
            f'{delta.wall_clock_change_pct:.3f} | '
            f'{delta.baseline_max_rss_kb} | '
            f'{delta.candidate_max_rss_kb} | '
            f'{delta.max_rss_change_pct:.3f} | '
            f'{delta.status} |'
        )
        for delta in deltas
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    baseline = load_metrics(Path(args.baseline))
    candidate = load_metrics(Path(args.candidate))
    deltas = compare(
        baseline,
        candidate,
        wall_regress_threshold_pct=args.wall_regress_threshold_pct,
        rss_regress_threshold_pct=args.rss_regress_threshold_pct,
        improve_threshold_pct=args.improve_threshold_pct,
    )

    write_json(Path(args.json_out), deltas, args)
    write_markdown(Path(args.md_out), deltas, args)

    has_regression = any(delta.status == 'regressed' for delta in deltas)
    if args.fail_on_regression and has_regression:
        print('Detected benchmark regressions', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
