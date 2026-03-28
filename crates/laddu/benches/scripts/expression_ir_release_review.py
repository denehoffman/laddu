#!/usr/bin/env python3
"""Assemble a release-review report for expression-ir benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Assemble a release-review report for expression-ir benchmark artifacts.'
    )
    parser.add_argument(
        '--compile-json',
        default='target/criterion/summary/laddu_core_expression_ir_compile_metrics.json',
        help='Path to laddu-core compile-metrics JSON artifact.',
    )
    parser.add_argument(
        '--memory-json',
        default='target/criterion/summary/laddu_core_expression_ir_memory_metrics.json',
        help='Path to laddu-core memory-metrics JSON artifact.',
    )
    parser.add_argument(
        '--cpu-json',
        default='target/criterion/summary/benchmark_summary_cpu_expression_ir_compare.json',
        help='Path to CPU workflow comparison JSON artifact.',
    )
    parser.add_argument(
        '--mpi-json',
        default='target/criterion/summary/benchmark_summary_mpi_expression_ir_compare.json',
        help='Path to MPI workflow comparison JSON artifact.',
    )
    parser.add_argument(
        '--json-out',
        default='target/criterion/summary/expression_ir_release_review.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--md-out',
        default='target/criterion/summary/expression_ir_release_review.md',
        help='Output Markdown path.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def fmt_pct(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:+.2f}%'


def compile_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get('runs', []))


def memory_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get('runs', []))


def cpu_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get('rows', []))


def mpi_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get('benchmarks', []))


def build_summary(
    compile_payload: dict[str, Any],
    memory_payload: dict[str, Any],
    cpu_payload: dict[str, Any],
    mpi_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'compile_json': args.compile_json,
            'memory_json': args.memory_json,
            'cpu_json': args.cpu_json,
            'mpi_json': args.mpi_json,
        },
        'compile_time': {
            'runs': compile_rows(compile_payload),
        },
        'steady_state': {
            'cpu_workflow': cpu_rows(cpu_payload),
            'mpi_workflow': mpi_rows(mpi_payload),
        },
        'memory': {
            'allocation_proxy_note': memory_payload.get('allocation_proxy_note'),
            'runs': memory_rows(memory_payload),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Expression-IR Release Review',
        '',
        f'Generated: {payload["generated_at_utc"]}',
        '',
        '## Compile-Time Costs',
        '',
        '| Benchmark Filter | Wall Clock (s) | IR Compile (ns) | Cached Integrals (ns) | Lowering (ns) | Cache Hits | Cache Misses |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for run in payload['compile_time']['runs']:
        metrics = run.get('compile_metrics', {})
        ir_compile = metrics.get('initial_ir_compile_nanos', 0) + metrics.get(
            'specialization_ir_compile_nanos', 0
        )
        cached_integrals = metrics.get('initial_cached_integrals_nanos', 0) + metrics.get(
            'specialization_cached_integrals_nanos', 0
        )
        lowering = metrics.get('initial_lowering_nanos', 0) + metrics.get(
            'specialization_lowering_nanos', 0
        )
        lines.append(
            f'| `{run["benchmark_filter"]}` | {fmt_float(run.get("wall_clock_sec"), 6)} | '
            f'{ir_compile} | {cached_integrals} | {lowering} | '
            f'{metrics.get("specialization_cache_hits", 0)} | '
            f'{metrics.get("specialization_cache_misses", 0)} |'
        )

    lines.extend(
        [
            '',
            '## Steady-State CPU Workflow Deltas',
            '',
            '| Benchmark ID | Time Delta % | Events/s Delta % | Gradient Evals/s Delta % |',
            '| --- | ---: | ---: | ---: |',
        ]
    )
    lines.extend(
        (
            f'| `{row["benchmark_id"]}` | '
            f'{fmt_pct(row.get("typical_estimate_delta_pct"))} | '
            f'{fmt_pct(row.get("events_per_sec_delta_pct"))} | '
            f'{fmt_pct(row.get("gradient_evals_per_sec_delta_pct"))} |'
        )
        for row in payload['steady_state']['cpu_workflow']
    )

    lines.extend(
        [
            '',
            '## Steady-State MPI Workflow Deltas',
            '',
            '| Benchmark ID | Time Delta % |',
            '| --- | ---: |',
        ]
    )
    lines.extend(
        f'| `{row["benchmark_id"]}` | {fmt_pct(row.get("delta_pct"))} |'
        for row in payload['steady_state']['mpi_workflow']
    )

    lines.extend(
        [
            '',
            '## Peak RSS And Allocation Proxies',
            '',
            f'Allocation proxy note: {payload["memory"].get("allocation_proxy_note", "-")}',
            '',
            '| Benchmark Filter | Wall Clock (s) | Max RSS (kB) | Allocation Proxy (bytes/iter) |',
            '| --- | ---: | ---: | ---: |',
        ]
    )
    lines.extend(
        (
            f'| `{run["benchmark_filter"]}` | {fmt_float(run.get("wall_clock_sec"), 6)} | '
            f'{run.get("max_rss_kb", "-")} | '
            f'{run.get("allocation_proxy_bytes_per_iter", "-")} |'
        )
        for run in payload['memory']['runs']
    )

    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    compile_payload = load_json(Path(args.compile_json))
    memory_payload = load_json(Path(args.memory_json))
    cpu_payload = load_json(Path(args.cpu_json))
    mpi_payload = load_json(Path(args.mpi_json))
    summary = build_summary(
        compile_payload,
        memory_payload,
        cpu_payload,
        mpi_payload,
        args,
    )
    write_json(Path(args.json_out), summary)
    write_markdown(Path(args.md_out), summary)


if __name__ == '__main__':
    main()
