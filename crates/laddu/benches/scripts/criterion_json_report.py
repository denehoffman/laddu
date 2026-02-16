#!/usr/bin/env python3
"""
Generate benchmark summary artifacts from Criterion JSON-line messages.

The input is expected to contain one JSON object per line from a benchmark runner
that emits Criterion-compatible machine messages (for example, `benchmark-complete`
and `group-complete`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkSummary:
    benchmark_id: str
    report_directory: str | None
    typical_estimate: float | None
    typical_unit: str | None
    throughput_per_iteration: float | None
    throughput_unit: str | None
    change_state: str | None
    change_mean_pct: float | None
    change_median_pct: float | None
    regression_status: str
    events_per_sec: float | None
    gradient_evals_per_sec: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert Criterion JSON-line output into summary JSON and Markdown artifacts.'
        )
    )
    parser.add_argument(
        '--input',
        default='-',
        help="Input JSONL file path. Use '-' to read from stdin.",
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/benchmark_summary.json',
        help='Output JSON summary path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/benchmark_summary.md',
        help='Output Markdown summary path.',
    )
    parser.add_argument(
        '--policy',
        choices=['off', 'warn', 'fail'],
        default='off',
        help='Regression policy mode.',
    )
    parser.add_argument(
        '--regression-threshold-pct',
        type=float,
        default=3.0,
        help='Percentage threshold above which a benchmark is treated as regressed.',
    )
    parser.add_argument(
        '--improvement-threshold-pct',
        type=float,
        default=3.0,
        help='Percentage threshold below which a benchmark is treated as improved.',
    )
    return parser.parse_args()


def read_lines(input_path: str) -> list[str]:
    if input_path == '-':
        import sys

        return [line.rstrip('\n') for line in sys.stdin]
    return Path(input_path).read_text(encoding='utf-8').splitlines()


def typical_seconds(estimate: float | None, unit: str | None) -> float | None:
    if estimate is None or unit is None:
        return None
    factors = {
        's': 1.0,
        'ms': 1e-3,
        'us': 1e-6,
        'ns': 1e-9,
        'ps': 1e-12,
    }
    factor = factors.get(unit)
    if factor is None:
        return None
    return estimate * factor


def classify_regression_status(
    change_mean_pct: float | None,
    change_state: str | None,
    regression_threshold_pct: float,
    improvement_threshold_pct: float,
) -> str:
    if change_mean_pct is not None:
        if change_mean_pct >= regression_threshold_pct:
            return 'regressed'
        if change_mean_pct <= -improvement_threshold_pct:
            return 'improved'
        return 'unchanged'

    if change_state == 'Regressed':
        return 'regressed'
    if change_state == 'Improved':
        return 'improved'
    if change_state == 'NoChange':
        return 'unchanged'
    return 'unknown'


def summarize_benchmark(
    message: dict[str, Any],
    regression_threshold_pct: float,
    improvement_threshold_pct: float,
) -> BenchmarkSummary:
    benchmark_id = str(message.get('id', ''))
    report_directory = message.get('report_directory')

    typical = message.get('typical') or {}
    typical_estimate = typical.get('estimate')
    typical_unit = typical.get('unit')

    throughput = message.get('throughput') or []
    throughput_per_iteration = None
    throughput_unit = None
    if throughput:
        first = throughput[0]
        throughput_per_iteration = first.get('per_iteration')
        throughput_unit = first.get('unit')

    change_state = None
    change_mean_pct = None
    change_median_pct = None
    change = message.get('change')
    if isinstance(change, dict):
        change_state = change.get('change')
        mean_change = change.get('mean')
        if isinstance(mean_change, dict):
            change_mean_pct = mean_change.get('estimate')
        median_change = change.get('median')
        if isinstance(median_change, dict):
            change_median_pct = median_change.get('estimate')

    regression_status = classify_regression_status(
        change_mean_pct=change_mean_pct,
        change_state=change_state,
        regression_threshold_pct=regression_threshold_pct,
        improvement_threshold_pct=improvement_threshold_pct,
    )

    seconds = typical_seconds(typical_estimate, typical_unit)
    events_per_sec = None
    if (
        seconds is not None
        and seconds > 0.0
        and throughput_per_iteration is not None
        and throughput_unit == 'elements'
    ):
        events_per_sec = float(throughput_per_iteration) / seconds

    gradient_evals_per_sec = None
    if seconds is not None and seconds > 0.0:
        if 'gradient' in benchmark_id or 'value_and_gradient' in benchmark_id:
            gradient_evals_per_sec = 1.0 / seconds

    return BenchmarkSummary(
        benchmark_id=benchmark_id,
        report_directory=report_directory,
        typical_estimate=typical_estimate,
        typical_unit=typical_unit,
        throughput_per_iteration=throughput_per_iteration,
        throughput_unit=throughput_unit,
        change_state=change_state,
        change_mean_pct=change_mean_pct,
        change_median_pct=change_median_pct,
        regression_status=regression_status,
        events_per_sec=events_per_sec,
        gradient_evals_per_sec=gradient_evals_per_sec,
    )


def write_json_output(
    benchmarks: list[BenchmarkSummary],
    groups: list[dict[str, Any]],
    json_out: Path,
    policy: str,
    regression_threshold_pct: float,
    improvement_threshold_pct: float,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    regressed_count = sum(
        1 for benchmark in benchmarks if benchmark.regression_status == 'regressed'
    )
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'policy': policy,
        'regression_threshold_pct': regression_threshold_pct,
        'improvement_threshold_pct': improvement_threshold_pct,
        'benchmark_count': len(benchmarks),
        'group_count': len(groups),
        'regressed_count': regressed_count,
        'benchmarks': [benchmark.__dict__ for benchmark in benchmarks],
        'groups': groups,
    }
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def fmt_float(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.6g}'


def write_markdown_output(
    benchmarks: list[BenchmarkSummary],
    md_out: Path,
    policy: str,
    regression_threshold_pct: float,
    improvement_threshold_pct: float,
) -> None:
    md_out.parent.mkdir(parents=True, exist_ok=True)
    regressed_count = sum(
        1 for benchmark in benchmarks if benchmark.regression_status == 'regressed'
    )
    lines = [
        '# Benchmark Summary',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        f'Policy: {policy}',
        (
            f'Thresholds: regress >= {regression_threshold_pct:.3g}%, '
            f'improve <= -{improvement_threshold_pct:.3g}%'
        ),
        f'Regressed benchmarks: {regressed_count}',
        '',
        '| Benchmark ID | Typical | Throughput | Events/s | Grad eval/s | Change | Change Mean % | Status |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for benchmark in benchmarks:
        typical = (
            f'{fmt_float(benchmark.typical_estimate)} {benchmark.typical_unit or ""}'.strip()
            if benchmark.typical_estimate is not None
            else '-'
        )
        throughput = '-'
        if benchmark.throughput_per_iteration is not None:
            throughput = (
                f'{fmt_float(float(benchmark.throughput_per_iteration))} '
                f'{benchmark.throughput_unit or ""}'
            ).strip()
        change_state = benchmark.change_state or '-'
        lines.append(
            f'| {benchmark.benchmark_id} | {typical} | {throughput} | '
            f'{fmt_float(benchmark.events_per_sec)} | '
            f'{fmt_float(benchmark.gradient_evals_per_sec)} | {change_state} | '
            f'{fmt_float(benchmark.change_mean_pct)} | {benchmark.regression_status} |'
        )
    md_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    input_lines = read_lines(args.input)
    benchmark_messages: list[dict[str, Any]] = []
    group_messages: list[dict[str, Any]] = []
    for line in input_lines:
        if not line.strip():
            continue
        message = json.loads(line)
        reason = message.get('reason')
        if reason == 'benchmark-complete':
            benchmark_messages.append(message)
        elif reason == 'group-complete':
            group_messages.append(message)

    benchmark_summaries = sorted(
        (
            summarize_benchmark(
                message,
                regression_threshold_pct=args.regression_threshold_pct,
                improvement_threshold_pct=args.improvement_threshold_pct,
            )
            for message in benchmark_messages
        ),
        key=lambda summary: summary.benchmark_id,
    )

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    write_json_output(
        benchmark_summaries,
        group_messages,
        json_out,
        policy=args.policy,
        regression_threshold_pct=args.regression_threshold_pct,
        improvement_threshold_pct=args.improvement_threshold_pct,
    )
    write_markdown_output(
        benchmark_summaries,
        md_out,
        policy=args.policy,
        regression_threshold_pct=args.regression_threshold_pct,
        improvement_threshold_pct=args.improvement_threshold_pct,
    )

    regressed = [
        benchmark
        for benchmark in benchmark_summaries
        if benchmark.regression_status == 'regressed'
    ]
    if regressed:
        details = ', '.join(benchmark.benchmark_id for benchmark in regressed)
        if args.policy == 'warn':
            print(
                f'Warning: detected {len(regressed)} regressed benchmark(s): {details}',
                file=sys.stderr,
            )
        elif args.policy == 'fail':
            message = f'Detected {len(regressed)} regressed benchmark(s): {details}'
            raise SystemExit(message)


if __name__ == '__main__':
    main()
