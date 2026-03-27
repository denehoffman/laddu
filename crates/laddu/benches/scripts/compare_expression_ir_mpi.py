#!/usr/bin/env python3
"""Compare workflow MPI benchmark summaries with and without `expression-ir`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare MPI workflow benchmark summaries with and without expression-ir.'
    )
    parser.add_argument(
        '--baseline-json',
        default='target/criterion/summary/benchmark_summary_mpi.json',
        help='Path to baseline MPI workflow benchmark summary JSON.',
    )
    parser.add_argument(
        '--candidate-json',
        default='target/criterion/summary/benchmark_summary_mpi_expression_ir.json',
        help='Path to expression-ir MPI workflow benchmark summary JSON.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/benchmark_summary_mpi_expression_ir_compare.json',
        help='Path to write joined comparison JSON.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/benchmark_summary_mpi_expression_ir_compare.md',
        help='Path to write joined comparison Markdown.',
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('benchmarks')
    if not isinstance(rows, list):
        message = f'{path} does not contain a "benchmarks" list'
        raise TypeError(message)
    return rows


def benchmark_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        bench_id = row.get('benchmark_id')
        if isinstance(bench_id, str):
            result[bench_id] = row
    return result


def format_ns(value: float | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.3f} ns'


def compare_rows(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_map = benchmark_map(baseline_rows)
    candidate_map = benchmark_map(candidate_rows)
    ids = sorted(set(baseline_map) | set(candidate_map))
    rows: list[dict[str, Any]] = []
    for bench_id in ids:
        baseline = baseline_map.get(bench_id)
        candidate = candidate_map.get(bench_id)
        baseline_estimate = (
            baseline.get('typical_estimate_ns') if baseline is not None else None
        )
        candidate_estimate = (
            candidate.get('typical_estimate_ns') if candidate is not None else None
        )
        delta_pct = None
        if (
            isinstance(baseline_estimate, (int, float))
            and isinstance(candidate_estimate, (int, float))
            and baseline_estimate != 0
        ):
            delta_pct = (
                (candidate_estimate - baseline_estimate) / baseline_estimate
            ) * 100.0
        rows.append(
            {
                'benchmark_id': bench_id,
                'baseline_typical_estimate_ns': baseline_estimate,
                'candidate_typical_estimate_ns': candidate_estimate,
                'delta_pct': delta_pct,
            }
        )
    return rows


def write_json(
    path: Path, rows: list[dict[str, Any]], baseline_json: str, candidate_json: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'baseline_json': baseline_json,
        'candidate_json': candidate_json,
        'benchmarks': rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(
    path: Path, rows: list[dict[str, Any]], baseline_json: str, candidate_json: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# MPI Workflow Comparison With `expression-ir`',
        '',
        f'- Baseline: `{baseline_json}`',
        f'- Candidate: `{candidate_json}`',
        '',
        '| Benchmark | Baseline | Candidate | Delta |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in rows:
        delta_pct = row['delta_pct']
        delta_text = 'n/a' if delta_pct is None else f'{delta_pct:+.2f}%'
        lines.append(
            '| '
            f'{row["benchmark_id"]} | '
            f'{format_ns(row["baseline_typical_estimate_ns"])} | '
            f'{format_ns(row["candidate_typical_estimate_ns"])} | '
            f'{delta_text} |'
        )
    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_json)
    candidate_path = Path(args.candidate_json)
    baseline_rows = load_rows(baseline_path)
    candidate_rows = load_rows(candidate_path)
    rows = compare_rows(baseline_rows, candidate_rows)
    write_json(Path(args.json_out), rows, args.baseline_json, args.candidate_json)
    write_markdown(Path(args.md_out), rows, args.baseline_json, args.candidate_json)


if __name__ == '__main__':
    main()
