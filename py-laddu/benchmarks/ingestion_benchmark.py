from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import subprocess
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypedDict

import laddu.io as ldio
import numpy as np
import pandas as pd
import polars as pl


class CaseResult(TypedDict):
    case: str
    events: int
    repeat: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    peak_rss_kib: int
    peak_tracemalloc_bytes: int
    copy_diag: dict[str, int | float]


def _test_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / 'tests' / 'data_files'


def _default_paths() -> dict[str, Path]:
    test_dir = _test_data_dir()
    return {
        'parquet': test_dir / 'data_f32.parquet',
        'root': test_dir / 'data_f32.root',
        'amptools': test_dir / 'data_amptools.root',
    }


def _make_columns(n_events: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12345)
    return {
        'beam_px': rng.normal(0.0, 1.0, n_events).astype(np.float64),
        'beam_py': rng.normal(0.0, 1.0, n_events).astype(np.float64),
        'beam_pz': rng.normal(8.0, 0.5, n_events).astype(np.float64),
        'beam_e': rng.normal(8.2, 0.5, n_events).astype(np.float64),
        'proton_px': rng.normal(0.0, 0.5, n_events).astype(np.float64),
        'proton_py': rng.normal(0.0, 0.5, n_events).astype(np.float64),
        'proton_pz': rng.normal(1.0, 0.3, n_events).astype(np.float64),
        'proton_e': rng.normal(1.4, 0.2, n_events).astype(np.float64),
        'pol_magnitude': rng.uniform(0.0, 1.0, n_events).astype(np.float64),
        'pol_angle': rng.uniform(-math.pi, math.pi, n_events).astype(np.float64),
        'weight': rng.uniform(0.5, 1.5, n_events).astype(np.float64),
    }


def _copy_diag(columns: dict[str, np.ndarray]) -> dict[str, int | float]:
    total_bytes = sum(int(array.nbytes) for array in columns.values())
    non_contig = sum(0 if array.flags.c_contiguous else 1 for array in columns.values())
    non_native_float = sum(
        0 if array.dtype in (np.float32, np.float64) else 1 for array in columns.values()
    )
    return {
        'columns': len(columns),
        'total_input_bytes': total_bytes,
        'non_contiguous_columns': non_contig,
        'non_native_float_columns': non_native_float,
    }


def _build_cases(
    n_events: int,
    paths: dict[str, Path],
) -> dict[str, tuple[Callable[[], Any], dict[str, int | float]]]:
    columns = _make_columns(n_events)
    np_columns = {name: np.asarray(values) for name, values in columns.items()}
    pandas_data = pd.DataFrame(np_columns)
    polars_data = pl.DataFrame(np_columns)

    return {
        'from_dict': (
            lambda: ldio.from_dict(columns),
            _copy_diag(np_columns),
        ),
        'from_numpy': (
            lambda: ldio.from_numpy(np_columns),
            _copy_diag(np_columns),
        ),
        'from_pandas': (
            lambda: ldio.from_pandas(pandas_data),
            _copy_diag(np_columns),
        ),
        'from_polars': (
            lambda: ldio.from_polars(polars_data),
            _copy_diag(np_columns),
        ),
        'read_parquet': (
            lambda: ldio.read_parquet(paths['parquet']),
            {
                'columns': 0,
                'total_input_bytes': 0,
                'non_contiguous_columns': 0,
                'non_native_float_columns': 0,
            },
        ),
        'read_root_oxyroot': (
            lambda: ldio.read_root(paths['root'], backend='oxyroot'),
            {
                'columns': 0,
                'total_input_bytes': 0,
                'non_contiguous_columns': 0,
                'non_native_float_columns': 0,
            },
        ),
        'read_root_uproot': (
            lambda: ldio.read_root(paths['root'], backend='uproot'),
            {
                'columns': 0,
                'total_input_bytes': 0,
                'non_contiguous_columns': 0,
                'non_native_float_columns': 0,
            },
        ),
        'read_amptools': (
            lambda: ldio.read_amptools(paths['amptools']),
            {
                'columns': 0,
                'total_input_bytes': 0,
                'non_contiguous_columns': 0,
                'non_native_float_columns': 0,
            },
        ),
    }


def _run_case(
    case: str, repeat: int, n_events: int, paths: dict[str, Path]
) -> CaseResult:
    cases = _build_cases(n_events, paths)
    if case not in cases:
        msg = f"Unknown case '{case}'"
        raise KeyError(msg)

    build, copy_diag = cases[case]
    durations_ms: list[float] = []

    # Warmup
    _ = build()

    peak_tracemalloc = 0
    events = 0
    tracemalloc.start()
    for _ in range(repeat):
        start = perf_counter()
        dataset = build()
        elapsed_ms = (perf_counter() - start) * 1000.0
        durations_ms.append(elapsed_ms)
        events = int(dataset.n_events)
        _, current_peak = tracemalloc.get_traced_memory()
        if current_peak > peak_tracemalloc:
            peak_tracemalloc = current_peak
    tracemalloc.stop()

    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'case': case,
        'events': events,
        'repeat': repeat,
        'mean_ms': statistics.mean(durations_ms),
        'median_ms': statistics.median(durations_ms),
        'min_ms': min(durations_ms),
        'max_ms': max(durations_ms),
        'peak_rss_kib': int(peak_rss_kib),
        'peak_tracemalloc_bytes': int(peak_tracemalloc),
        'copy_diag': copy_diag,
    }


def _worker_main(args: argparse.Namespace) -> None:
    paths = _default_paths()
    result = _run_case(args.worker_case, args.repeat, args.events, paths)
    print(json.dumps(result))


def _load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _summarize_with_baseline(
    results: list[CaseResult],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_cases = baseline.get('cases', {}) if isinstance(baseline, dict) else {}
    case_map: dict[str, Any] = {}
    for result in results:
        base = baseline_cases.get(result['case'])
        delta_ms_pct = None
        delta_rss_pct = None
        if isinstance(base, dict):
            base_median = float(base.get('median_ms', 0.0) or 0.0)
            base_rss = float(base.get('peak_rss_kib', 0.0) or 0.0)
            if base_median > 0.0:
                delta_ms_pct = ((result['median_ms'] - base_median) / base_median) * 100.0
            if base_rss > 0.0:
                delta_rss_pct = ((result['peak_rss_kib'] - base_rss) / base_rss) * 100.0

        case_map[result['case']] = {
            **result,
            'delta_vs_baseline_median_ms_pct': delta_ms_pct,
            'delta_vs_baseline_peak_rss_pct': delta_rss_pct,
        }

    return {
        'meta': {
            'events': results[0]['events'] if results else 0,
            'repeat': results[0]['repeat'] if results else 0,
            'baseline_loaded': bool(baseline_cases),
        },
        'cases': case_map,
    }


def _main(args: argparse.Namespace) -> None:
    paths = _default_paths()
    cases = list(_build_cases(args.events, paths).keys())
    if args.case != 'all':
        cases = [args.case]

    results: list[CaseResult] = []
    for case in cases:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--worker-case',
            case,
            '--repeat',
            str(args.repeat),
            '--events',
            str(args.events),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        results.append(json.loads(completed.stdout.strip()))

    baseline = _load_baseline(Path(args.baseline) if args.baseline else None)
    summary = _summarize_with_baseline(results, baseline)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark Python ingestion readers')
    parser.add_argument(
        '--case',
        default='all',
        choices=[
            'all',
            'from_dict',
            'from_numpy',
            'from_pandas',
            'from_polars',
            'read_parquet',
            'read_root_oxyroot',
            'read_root_uproot',
            'read_amptools',
        ],
        help='Benchmark case to run',
    )
    parser.add_argument('--repeat', type=int, default=5, help='Repetitions per case')
    parser.add_argument(
        '--events',
        type=int,
        default=5000,
        help='Synthetic event count for in-memory ingestion cases',
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Optional baseline JSON for delta comparison',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='target/benchmarks/python_ingestion_summary.json',
        help='Output JSON path',
    )
    parser.add_argument('--worker-case', type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = parse_args()
    if parsed.worker_case:
        _worker_main(parsed)
    else:
        _main(parsed)
