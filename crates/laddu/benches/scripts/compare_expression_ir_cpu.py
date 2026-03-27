#!/usr/bin/env python3
"""Compare workflow CPU benchmark summaries with and without `expression-ir`."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BenchEntry:
    benchmark_id: str
    typical_estimate: float | None
    typical_unit: str | None
    events_per_sec: float | None
    gradient_evals_per_sec: float | None
    regression_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare CPU workflow benchmark summaries with and without expression-ir.'
    )
    parser.add_argument(
        '--baseline-json',
        default='target/criterion/summary/benchmark_summary_cpu.json',
        help='Path to baseline summary JSON without expression-ir.',
    )
    parser.add_argument(
        '--candidate-json',
        default='target/criterion/summary/benchmark_summary_cpu_expression_ir.json',
        help='Path to candidate summary JSON with expression-ir.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/benchmark_summary_cpu_expression_ir_compare.json',
        help='Output JSON comparison path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/benchmark_summary_cpu_expression_ir_compare.md',
        help='Output Markdown comparison path.',
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, BenchEntry]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    benchmarks = {}
    for item in payload.get('benchmarks', []):
        bench = BenchEntry(
            benchmark_id=item['benchmark_id'],
            typical_estimate=item.get('typical_estimate'),
            typical_unit=item.get('typical_unit'),
            events_per_sec=item.get('events_per_sec'),
            gradient_evals_per_sec=item.get('gradient_evals_per_sec'),
            regression_status=item.get('regression_status', 'unknown'),
        )
        benchmarks[bench.benchmark_id] = bench
    return benchmarks


def pct_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0.0) or candidate is None:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def fmt_float(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.6g}'


def write_json(
    path: Path, rows: list[dict[str, Any]], baseline_json: str, candidate_json: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_json': baseline_json,
        'candidate_json': candidate_json,
        'rows': rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(
    path: Path, rows: list[dict[str, Any]], baseline_json: str, candidate_json: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# CPU Workflow Benchmark Comparison',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        f'- Baseline: `{baseline_json}`',
        f'- Candidate: `{candidate_json}`',
        '',
        '| Benchmark ID | Baseline Time | Candidate Time | Time Delta % | Baseline Events/s | Candidate Events/s | Events/s Delta % |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    lines.extend(
        (
            f'| `{row["benchmark_id"]}` | '
            f'{fmt_float(row["baseline_typical_estimate"])} {row["baseline_typical_unit"] or ""}'.rstrip()
            + f' | {fmt_float(row["candidate_typical_estimate"])} {row["candidate_typical_unit"] or ""}'.rstrip()
            + f' | {fmt_float(row["typical_estimate_delta_pct"])} | '
            f'{fmt_float(row["baseline_events_per_sec"])} | '
            f'{fmt_float(row["candidate_events_per_sec"])} | '
            f'{fmt_float(row["events_per_sec_delta_pct"])} |'
        )
        for row in rows
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_json)
    candidate_path = Path(args.candidate_json)
    baseline = load_summary(baseline_path)
    candidate = load_summary(candidate_path)
    benchmark_ids = sorted(set(baseline) & set(candidate))

    rows = []
    for benchmark_id in benchmark_ids:
        base = baseline[benchmark_id]
        cand = candidate[benchmark_id]
        rows.append(
            {
                'benchmark_id': benchmark_id,
                'baseline_typical_estimate': base.typical_estimate,
                'baseline_typical_unit': base.typical_unit,
                'candidate_typical_estimate': cand.typical_estimate,
                'candidate_typical_unit': cand.typical_unit,
                'typical_estimate_delta_pct': pct_delta(
                    base.typical_estimate, cand.typical_estimate
                ),
                'baseline_events_per_sec': base.events_per_sec,
                'candidate_events_per_sec': cand.events_per_sec,
                'events_per_sec_delta_pct': pct_delta(
                    base.events_per_sec, cand.events_per_sec
                ),
                'baseline_gradient_evals_per_sec': base.gradient_evals_per_sec,
                'candidate_gradient_evals_per_sec': cand.gradient_evals_per_sec,
                'gradient_evals_per_sec_delta_pct': pct_delta(
                    base.gradient_evals_per_sec, cand.gradient_evals_per_sec
                ),
                'baseline_regression_status': base.regression_status,
                'candidate_regression_status': cand.regression_status,
            }
        )

    write_json(Path(args.json_out), rows, args.baseline_json, args.candidate_json)
    write_markdown(Path(args.md_out), rows, args.baseline_json, args.candidate_json)


if __name__ == '__main__':
    main()
