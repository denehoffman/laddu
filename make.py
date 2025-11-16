#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["rich>=13.7"]
# ///
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import webbrowser
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent
PY_LADDU = REPO_ROOT / 'py-laddu'
PY_LADDU_MPI = REPO_ROOT / 'py-laddu-mpi'
VENV_PATH = REPO_ROOT / '.venv'
SCRIPT_NAME = Path(__file__).name
BIN_DIR = 'Scripts' if os.name == 'nt' else 'bin'
DOCKER_FLAG = 'LADDU_INSIDE_DOCKER'
DOCKER_IMAGE = 'laddu:latest'
DOCKER_WORKDIR = Path('/work')
DOCKER_SRC = Path('/src')
CONTAINER_UV_PYTHON = Path('/root/.local/share/uv/python')
VERBOSE = False

console = Console()


class Target(str, Enum):

    """Available Python frontends."""

    laddu = 'laddu'
    mpi = 'mpi'


def _venv_python() -> Path:
    return VENV_PATH / BIN_DIR / ('python.exe' if os.name == 'nt' else 'python')


def _build_env(*, use_venv: bool = False, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if use_venv:
        env['VIRTUAL_ENV'] = str(VENV_PATH)
        bin_path = _venv_python().parent
        env['PATH'] = f'{bin_path}{os.pathsep}{env.get("PATH", "")}'
    if extra:
        env.update(extra)
    return env


def _run(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    safe_cmd = [str(part) for part in cmd]
    display = shlex.join(safe_cmd)
    console.print(f'[bold green]$[/] {display}')
    subprocess.run(safe_cmd, check=True, cwd=cwd or REPO_ROOT, env=env)  # noqa: S603


def _python_version_of(python_exe: Path) -> str | None:
    if not python_exe.exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [python_exe, '-c', 'import platform; print(platform.python_version())'],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _maybe_remove_venv() -> None:
    if VENV_PATH.exists():
        console.print(f'[yellow]Removing {VENV_PATH}[/]')
        shutil.rmtree(VENV_PATH)


def _version_matches(current: str | None, expected: str) -> bool:
    if current is None:
        return False
    if '.' in expected:
        return current.startswith(expected)
    return current.startswith(f'{expected}.')


def ensure_venv(python_version: str | None = None) -> None:
    expected = python_version or f'{sys.version_info.major}.{sys.version_info.minor}'
    current = _python_version_of(_venv_python())
    if not _version_matches(current, expected):
        if current is not None:
            console.print(f'[yellow].venv uses Python {current}, recreating for {expected}[/]')
        _maybe_remove_venv()
    if not VENV_PATH.exists():
        console.print(f'[cyan]Creating venv at {VENV_PATH}[/]')
        python_arg = python_version or sys.executable
        _run(['uv', 'venv', str(VENV_PATH), '-p', python_arg])


def _run_maturin(
    project_dir: Path,
    extras: Sequence[str] | None = None,
    python_version: str | None = None,
    profile: str | None = None,
) -> None:
    ensure_venv(python_version)
    ensure_dir(project_dir)
    cmd = ['maturin', 'develop', '--uv']
    if profile:
        cmd.extend(['--profile', profile])
    if extras:
        unique = sorted(set(extras))
        cmd.append(f'--extras={",".join(unique)}')
    env = _build_env(use_venv=True, extra={'CARGO_INCREMENTAL': 'true'})
    target_label = project_dir.name
    extras_label = f' [{",".join(extras)}]' if extras else ''
    _run_quiet(
        cmd,
        message=f'Maturin develop: {target_label}{extras_label}',
        cwd=project_dir,
        env=env,
    )


def _ensure_laddu_uninstalled(package: str) -> None:
    python_exe = _venv_python()
    if not python_exe.exists():
        return
    _run_quiet(
        [
            'uv',
            'pip',
            'uninstall',
            '--python',
            str(python_exe),
            package,
        ],
        message=f'Removing existing {package} installation (if any)',
        check=False,
    )


def _ensure_nextest_available() -> bool:
    try:
        subprocess.run(
            ['cargo', 'nextest', '--version'],  # noqa: S607
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print(
            '[yellow]cargo-nextest not found. Install it with '
            '`cargo install cargo-nextest --locked` to run Rust tests.[/]'
        )
        return False
    return True


def _confirm_mpi_variant(*, expected: bool, label: str) -> None:
    env = _build_env(use_venv=True)
    script = (
        'import laddu, sys; '
        "value = bool(getattr(laddu, 'mpi', None) and laddu.mpi.is_mpi_available()); "
        'print(value); '
        f'sys.exit(0 if value == {expected!s} else 1)'
    )
    result = subprocess.run(  # noqa: S603
        [str(_venv_python()), '-c', script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''
    actual = output.lower() == 'true'
    if result.returncode != 0 or actual != expected:
        detail = output or result.stderr.strip() or 'unknown'
        console.print(f'[red]{label}: Expected MPI availability {expected}, observed {detail}[/]')
        msg = 'MPI availability mismatch'
        raise RuntimeError(msg)
    state = 'enabled' if actual else 'disabled'
    console.print(f'[green]{label}[/]: MPI support {state}')


def _run_pytest(project_dir: Path, python_version: str | None = None) -> None:
    ensure_venv(python_version)
    ensure_dir(project_dir)
    env = _build_env(use_venv=True)
    _run([str(_venv_python()), '-m', 'pytest'], cwd=project_dir, env=env)


def _build_docs(*, clean: bool, python_version: str | None = None) -> None:
    _run_maturin(PY_LADDU, extras=['docs'], python_version=python_version)
    docs_cmd = ['make', '-C', 'docs']
    if clean:
        _run([*docs_cmd, 'clean'], cwd=PY_LADDU)
    _run([*docs_cmd, 'html'], cwd=PY_LADDU)


class CLIError(RuntimeError):

    """Raised for predictable CLI failures."""


def ensure_dir(path: Path) -> None:
    if not path.exists():
        msg = f'Expected directory {path} is missing'
        raise CLIError(msg)


def _forward_cli_args() -> list[str]:
    cleaned: list[str] = []
    for arg in sys.argv[1:]:
        if arg == '--docker':
            continue
        cleaned.append(arg)
    return cleaned


def _run_inside_docker(image: str) -> int:
    args = _forward_cli_args()
    cmd = [
        'docker',
        'run',
        '--rm',
        '-v',
        f'{REPO_ROOT}:{DOCKER_SRC}:ro',
        '--tmpfs',
        f'{DOCKER_WORKDIR}:rw,exec,nosuid,size=8g',
        '--tmpfs',
        f'{DOCKER_WORKDIR / ".venv"}:rw,exec,nosuid,size=2g',
        '-w',
        str(DOCKER_WORKDIR),
        '-e',
        f'{DOCKER_FLAG}=1',
        '-e',
        f'LD_LIBRARY_PATH={DOCKER_WORKDIR}/.venv/lib:$LD_LIBRARY_PATH',
    ]
    if sys.stdin.isatty():
        cmd.append('-i')
    if sys.stdout.isatty():
        cmd.append('-t')
    quoted_args = ' '.join(shlex.quote(a) for a in args)
    rsync_excludes = ' '.join(f"--exclude '{pattern}'" for pattern in ('target/', '.venv/', '__pycache__/'))
    inner_cmd = (
        'set -euo pipefail; '
        'mkdir -p /work; '
        f'rsync -a --delete {rsync_excludes} {DOCKER_SRC}/ {DOCKER_WORKDIR}/; '
        f'uv venv {DOCKER_WORKDIR}/.venv; '
        'PYTHON_LIB=$(find /root/.local/share/uv/python -maxdepth 2 -type d -name "cpython-*-linux-*-gnu" | head -n 1)/lib; '
        'export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH; '
        f'cd {DOCKER_WORKDIR}; '
        f'uv run {DOCKER_WORKDIR}/{SCRIPT_NAME}'
    )
    if quoted_args:
        inner_cmd += f' {quoted_args}'
    cmd.extend([image, 'bash', '-lc', inner_cmd])
    process = subprocess.run(cmd, check=False)  # noqa: S603
    return process.returncode


def _cargo_with_features(cmd: list[str], *, use_mpi: bool) -> list[str]:
    if not use_mpi:
        return cmd
    enriched = list(cmd)
    enriched.extend(['--features', 'mpi'])
    return enriched


def _run_quiet(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    message: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    safe_cmd = [str(part) for part in cmd]
    if VERBOSE:
        console.print(f'[cyan]{message}[/]')
        result = subprocess.run(safe_cmd, check=False, cwd=cwd or REPO_ROOT, env=env)  # noqa: S603
    else:
        with console.status(message):
            result = subprocess.run(  # noqa: S603
                safe_cmd,
                check=False,
                cwd=cwd or REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
    if result.returncode != 0:
        if check:
            console.print(f'[red]Command failed:[/] {shlex.join(safe_cmd)}')
            if result.stdout:
                console.print(result.stdout)
            if result.stderr:
                console.print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, safe_cmd, output=result.stdout, stderr=result.stderr)
        warning = result.stderr or result.stdout or ''
        if warning:
            console.print(f'[yellow]{message} (ignored): {warning.strip()}[/]')
    return result


def _cmd_develop(args: argparse.Namespace) -> None:
    extras: list[str] = []
    if args.tests or args.all_extras:
        extras.append('tests')
    if args.docs or args.all_extras:
        extras.append('docs')
    project = PY_LADDU_MPI if args.mpi else PY_LADDU
    package_to_remove = 'laddu' if args.mpi else 'laddu-mpi'
    _ensure_laddu_uninstalled(package_to_remove)
    try:
        _run_maturin(project, extras, python_version=args.python_version)
        _confirm_mpi_variant(expected=bool(args.mpi), label='develop')
    except RuntimeError:
        package_to_remove = 'laddu' if args.mpi else 'laddu-mpi'
        console.print(
            '[yellow]Detected mismatched laddu build during develop; removing conflicting package and retrying.[/]'
        )
        _ensure_laddu_uninstalled(package_to_remove)
        _run(['cargo', 'clean'])
        _run_maturin(project, extras, python_version=args.python_version)
        _confirm_mpi_variant(expected=bool(args.mpi), label='develop')


def _cmd_test(args: argparse.Namespace) -> None:
    rust = args.rust
    python = args.python
    mpi = args.mpi
    if not any([rust, python]):
        rust = True
        python = True

    if rust:
        build_cmd = _cargo_with_features(['cargo', 'build', '--all-targets', '--quiet'], use_mpi=mpi)
        feature_label = ' with MPI features' if mpi else ''
        _run_quiet(build_cmd, message=f'Building Rust workspace{feature_label}')
        if _ensure_nextest_available():
            nextest_cmd = _cargo_with_features(['cargo', 'nextest', 'run'], use_mpi=mpi)
            _run(nextest_cmd)
        doc_cmd = _cargo_with_features(['cargo', 'test', '--doc'], use_mpi=mpi)
        console.print(f'[cyan]Running Rust doc tests{feature_label}[/]')
        _run(doc_cmd)
        console.print(f'[green]Rust doc tests completed{feature_label}[/]')
    if python:
        project = PY_LADDU_MPI if mpi else PY_LADDU
        label = 'py-laddu-mpi tests' if mpi else 'py-laddu tests'
        try:
            _run_maturin(
                project,
                extras=['tests'],
                python_version=args.python_version,
                profile='dev',
            )
            _confirm_mpi_variant(expected=bool(mpi), label=label)
        except RuntimeError:
            console.print('[yellow]Detected mismatched laddu build; removing conflicting package and retrying.[/]')
            package_to_remove = 'laddu' if mpi else 'laddu-mpi'
            _ensure_laddu_uninstalled(package_to_remove)
            _run(['cargo', 'clean'])
            _run_maturin(
                project,
                extras=['tests'],
                python_version=args.python_version,
                profile='dev',
            )
            _confirm_mpi_variant(expected=bool(mpi), label=label)
        _run_pytest(project, python_version=args.python_version)


def _cmd_docs(args: argparse.Namespace) -> None:
    _run(['cargo', 'doc'], env=None)
    _build_docs(clean=args.clean, python_version=args.python_version)
    _open_docs(open_python=args.open in {'python', 'all'}, open_rust=args.open in {'rust', 'all'})


def _cmd_clean(args: argparse.Namespace) -> None:
    _run(['cargo', 'clean'])
    if args.python and VENV_PATH.exists():
        _maybe_remove_venv()


def _cmd_venv(args: argparse.Namespace) -> None:
    if args.recreate:
        _maybe_remove_venv()
    ensure_venv(args.python_version)
    console.print(f'.venv ready with Python {_python_version_of(_venv_python())}')


def _open_docs(*, open_python: bool, open_rust: bool) -> None:
    if open_rust:
        rust_index = REPO_ROOT / 'target' / 'doc' / 'laddu' / 'index.html'
        if rust_index.exists():
            console.print(f'[cyan]Opening Rust docs at {rust_index}[/]')
            webbrowser.open(rust_index.as_uri())
        else:
            console.print('[yellow]Rust docs not found. Run `cargo doc` first.[/]')
    if open_python:
        html_index = PY_LADDU / 'docs' / 'build' / 'html' / 'index.html'
        if not html_index.exists():
            msg = f'Python docs not found at {html_index}. Run `{SCRIPT_NAME} docs` first.'
            raise CLIError(msg)
        console.print(f'[cyan]Opening {html_index}[/]')
        webbrowser.open(html_index.as_uri())


def _cmd_docker(args: argparse.Namespace) -> None:
    docker_dir = REPO_ROOT
    if args.build:
        _run(['docker', 'build', '-t', DOCKER_IMAGE, '.'], cwd=docker_dir)
        return
    cmd = [
        'docker',
        'run',
        '--rm',
        '-v',
        f'{REPO_ROOT}:{DOCKER_SRC}:ro',
        '--tmpfs',
        f'{DOCKER_WORKDIR}:rw,exec,nosuid,size=8g',
        '--tmpfs',
        f'{DOCKER_WORKDIR / ".venv"}:rw,exec,nosuid,size=2g',
        '-w',
        str(DOCKER_WORKDIR),
        '-e',
        f'{DOCKER_FLAG}=1',
        '-e',
        f'LD_LIBRARY_PATH={DOCKER_WORKDIR}/.venv/lib:$LD_LIBRARY_PATH',
    ]
    if sys.stdin.isatty():
        cmd.append('-i')
    if sys.stdout.isatty():
        cmd.append('-t')
    rsync_excludes = ' '.join(f"--exclude '{pattern}'" for pattern in ('target/', '.venv/', '__pycache__/'))
    cmd.extend(
        [
            DOCKER_IMAGE,
            'bash',
            '-lc',
            (
                'set -euo pipefail; '
                'mkdir -p /work; '
                f'rsync -a --delete {rsync_excludes} {DOCKER_SRC}/ {DOCKER_WORKDIR}/; '
                f'uv venv {DOCKER_WORKDIR}/.venv; '
                'PYTHON_LIB=$(find /root/.local/share/uv/python -maxdepth 2 -type d -name "cpython-*-linux-*-gnu" | head -n 1)/lib; '
                'export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH; '
                f'cd {DOCKER_WORKDIR}; '
                'exec bash'
            ),
        ]
    )
    _run(cmd, cwd=REPO_ROOT)


def _cmd_clippy(args: argparse.Namespace) -> None:
    cmd = _cargo_with_features(['cargo', 'clippy', '--all-targets'], use_mpi=args.mpi)
    _run(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description='Single entry point for laddu development, testing, docs, and tooling.',
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Run the command inside the laddu Docker image.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full command output (disables spinners).',
    )
    subparsers = parser.add_subparsers(dest='command')

    develop_parser = subparsers.add_parser('develop', help='Build Python packages via maturin with optional extras.')
    develop_parser.add_argument(
        '--mpi',
        action='store_true',
        help='Build the MPI variant (py-laddu-mpi).',
    )
    develop_parser.add_argument(
        '--tests',
        action='store_true',
        help='Include the tests extra when building.',
    )
    develop_parser.add_argument(
        '--docs',
        action='store_true',
        help='Include the docs extra when building.',
    )
    develop_parser.add_argument(
        '--all-extras',
        action='store_true',
        help='Include both docs and tests extras.',
    )
    develop_parser.add_argument(
        '--python-version',
        help='Python version to use for the managed virtualenv (e.g. 3.10).',
    )
    develop_parser.set_defaults(handler=_cmd_develop)

    test_parser = subparsers.add_parser('test', help='Run Rust/Python tests and doctests.')
    test_parser.add_argument(
        '--rust',
        action='store_true',
        help='Run cargo nextest tests.',
    )
    test_parser.add_argument(
        '--python',
        action='store_true',
        help='Run pytest for py-laddu.',
    )
    test_parser.add_argument(
        '--mpi',
        action='store_true',
        help='Also run the MPI Python tests and enable the mpi Cargo feature.',
    )
    test_parser.add_argument(
        '--python-version',
        help='Python version to use for Python/MPI test environments.',
    )
    test_parser.set_defaults(handler=_cmd_test)

    docs_parser = subparsers.add_parser('docs', help='Build Sphinx documentation inside py-laddu/docs.')
    docs_parser.add_argument(
        '--no-clean',
        dest='clean',
        action='store_false',
        help='Skip cleaning docs before building.',
    )
    docs_parser.add_argument(
        '--open',
        nargs='?',
        const='all',
        choices=['python', 'rust', 'all'],
        help='Open docs after build (default: all).',
    )
    docs_parser.add_argument(
        '--python-version',
        help='Python version to use for documentation builds.',
    )
    docs_parser.set_defaults(handler=_cmd_docs, clean=True)

    clean_parser = subparsers.add_parser('clean', help='Run cargo clean and optionally delete the repo venv.')
    clean_parser.add_argument(
        '--python',
        action='store_true',
        help='Delete the managed Python virtualenv as well.',
    )
    clean_parser.set_defaults(handler=_cmd_clean)

    venv_parser = subparsers.add_parser('venv', help='Ensure the repo virtualenv exists, optionally recreating it.')
    venv_parser.add_argument(
        '--recreate',
        action='store_true',
        help='Force recreation of the managed virtualenv.',
    )
    venv_parser.add_argument(
        '--python-version',
        help='Python version to create/use for the managed virtualenv.',
    )
    venv_parser.set_defaults(handler=_cmd_venv)

    docker_parser = subparsers.add_parser('docker', help='Build or run the laddu Docker container.')
    docker_parser.add_argument(
        '--build',
        action='store_true',
        help='Build the laddu:latest image instead of starting a container shell.',
    )
    docker_parser.set_defaults(handler=_cmd_docker)

    clippy_parser = subparsers.add_parser('clippy', help='Run `cargo clippy --all-targets`.')
    clippy_parser.add_argument(
        '--mpi',
        action='store_true',
        help='Enable the MPI feature when running clippy.',
    )
    clippy_parser.set_defaults(handler=_cmd_clippy)

    return parser


def main() -> int:
    if args.docker and os.environ.get(DOCKER_FLAG) != '1':
        return _run_inside_docker(DOCKER_IMAGE)

    handler = getattr(args, 'handler', None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        handler(args)
    except CLIError as exc:
        console.print(f'[red]{exc}[/]')
        return 2
    return 0


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()
    VERBOSE = args.verbose
    raise SystemExit(main())
