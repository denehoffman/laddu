#!/usr/bin/env python3
"""Run a command on a single MPI rank and emit per-rank metrics."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run command and capture per-rank metrics.'
    )
    parser.add_argument(
        '--run-dir', required=True, help='Output directory for per-rank files.'
    )
    parser.add_argument('--command', required=True, help='Shell command to execute.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    stdout_path = run_dir / f'rank_{rank}.stdout'
    stderr_path = run_dir / f'rank_{rank}.stderr'
    metrics_path = run_dir / f'rank_{rank}.json'

    start = time.perf_counter()
    with stdout_path.open('w', encoding='utf-8') as stdout_f, stderr_path.open(
        'w', encoding='utf-8'
    ) as stderr_f:
        proc = subprocess.run(
            args.command,
            shell=True,
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
        )
    elapsed = time.perf_counter() - start

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics = {
        'rank': rank,
        'elapsed_sec': elapsed,
        'max_rss_kb': int(usage.ru_maxrss),
        'return_code': proc.returncode,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8'
    )

    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
