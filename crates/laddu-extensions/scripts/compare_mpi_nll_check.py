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
    case: str
    value: float
    gradient: list[float]


def parse_json_lines(output: str, mode: str) -> dict[str, CheckResult]:
    cases: dict[str, CheckResult] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        obj = json.loads(line)
        if obj.get('mode') != mode:
            continue
        case = str(obj['case'])
        cases[case] = CheckResult(
            mode=mode,
            case=case,
            value=float(obj['value']),
            gradient=list(obj['gradient']),
        )
    if not cases:
        msg = f"did not find JSON output for mode '{mode}'"
        raise RuntimeError(msg)
    return cases


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout


def close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def compare(
    local_cases: dict[str, CheckResult],
    mpi_cases: dict[str, CheckResult],
    rel_tol: float,
    abs_tol: float,
) -> None:
    local_names = sorted(local_cases.keys())
    mpi_names = sorted(mpi_cases.keys())
    if local_names != mpi_names:
        msg = f'case mismatch: local={local_names} mpi={mpi_names}'
        raise RuntimeError(msg)

    for name in local_names:
        local = local_cases[name]
        mpi = mpi_cases[name]
        if len(local.gradient) != len(mpi.gradient):
            msg = (
                f"gradient length mismatch for case '{name}': "
                f'local={len(local.gradient)} mpi={len(mpi.gradient)}'
            )
            raise RuntimeError(msg)
        if not close(local.value, mpi.value, rel_tol, abs_tol):
            msg = f"value mismatch for case '{name}': local={local.value} mpi={mpi.value}"
            raise RuntimeError(msg)
        for i, (lv, mv) in enumerate(zip(local.gradient, mpi.gradient, strict=False)):
            if not close(lv, mv, rel_tol, abs_tol):
                msg = f"gradient[{i}] mismatch for case '{name}': local={lv} mpi={mv}"
                raise RuntimeError(msg)


def main() -> int:
    rel_tol = 1e-8
    abs_tol = 1e-10

    local_cmd = [
        'cargo',
        'run',
        '-p',
        'laddu-extensions',
        '--bin',
        'mpi_nll_check',
    ]
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

    local_cases = parse_json_lines(local_output, 'local')
    mpi_cases = parse_json_lines(mpi_output, 'mpi')
    compare(local_cases, mpi_cases, rel_tol=rel_tol, abs_tol=abs_tol)

    print('mpi_nll_check comparison passed')
    for case in sorted(local_cases.keys()):
        local = local_cases[case]
        mpi = mpi_cases[case]
        print(
            f'{case}: local={local.value:.17e} mpi={mpi.value:.17e} '
            f'grad_dim={len(local.gradient)}'
        )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end='', file=sys.stdout)
        print(exc.stderr, end='', file=sys.stderr)
        raise
