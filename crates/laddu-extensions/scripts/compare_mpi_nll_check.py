#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    mode: str
    value: float
    gradient: list[float]


def parse_json_line(output: str, mode: str) -> CheckResult:
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        obj = json.loads(line)
        if obj.get('mode') == mode:
            return CheckResult(
                mode=mode, value=float(obj['value']), gradient=list(obj['gradient'])
            )
    msg = f"did not find JSON output for mode '{mode}'"
    raise RuntimeError(msg)


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout


def close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def compare(local: CheckResult, mpi: CheckResult, rel_tol: float, abs_tol: float) -> None:
    if len(local.gradient) != len(mpi.gradient):
        msg = (
            f'gradient length mismatch: local={len(local.gradient)} '
            f'mpi={len(mpi.gradient)}'
        )
        raise RuntimeError(msg)

    if not close(local.value, mpi.value, rel_tol, abs_tol):
        msg = f'value mismatch: local={local.value} mpi={mpi.value}'
        raise RuntimeError(msg)

    for i, (lv, mv) in enumerate(zip(local.gradient, mpi.gradient)):
        if not close(lv, mv, rel_tol, abs_tol):
            msg = f'gradient[{i}] mismatch: local={lv} mpi={mv}'
            raise RuntimeError(msg)


def main() -> int:
    rel_tol = 1e-8
    abs_tol = 1e-10

    local_cmd = ['cargo', 'run', '-p', 'laddu-extensions', '--bin', 'mpi_nll_check']
    mpi_cmd = [
        'mpirun',
        '-n',
        '2',
        'cargo',
        'run',
        '-p',
        'laddu-extensions',
        '--features',
        'mpi',
        '--bin',
        'mpi_nll_check',
    ]

    local_output = run_command(local_cmd)
    mpi_output = run_command(mpi_cmd)

    local = parse_json_line(local_output, 'local')
    mpi = parse_json_line(mpi_output, 'mpi')
    compare(local, mpi, rel_tol=rel_tol, abs_tol=abs_tol)

    print('mpi_nll_check comparison passed')
    print(f'local value={local.value:.17e}')
    print(f'mpi value={mpi.value:.17e}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end='', file=sys.stdout)
        print(exc.stderr, end='', file=sys.stderr)
        raise
