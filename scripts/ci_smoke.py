"""Build and exercise Linux release artifacts in isolated Python environments."""

from __future__ import annotations

# Commands below are fixed tool invocations with shell-free argument lists.
# ruff: noqa: S603
import email.parser
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

PACKAGES = {'laddu': 'laddu', 'laddu-local': '_laddu_local', 'laddu-mpi': '_laddu_mpi'}


def run(*args: str) -> None:
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def check_sdist(path: Path, package: str) -> None:
    with tarfile.open(path) as archive:
        metadata = [
            member
            for member in archive.getmembers()
            if member.name.count('/') == 1 and member.name.endswith('/PKG-INFO')
        ]
        if len(metadata) != 1:
            msg = f'{path}: expected one root PKG-INFO'
            raise ValueError(msg)
        stream = archive.extractfile(metadata[0])
        if stream is None:
            msg = f'{path}: unreadable PKG-INFO'
            raise ValueError(msg)
        parsed = email.parser.BytesParser().parsebytes(stream.read())
        if parsed['Name'] != package or not parsed['Version']:
            msg = f'{path}: incorrect package metadata'
            raise ValueError(msg)


def main() -> None:
    python = os.environ['SMOKE_PYTHON']
    # PEP 517 builds unpack the MPI sdist elsewhere. Keep dependency compilation
    # in the checkout's cached target directory instead of starting from scratch.
    os.environ.setdefault('CARGO_TARGET_DIR', str(Path('target').resolve()))
    with tempfile.TemporaryDirectory(prefix='laddu-release-smoke-') as temporary:
        root = Path(temporary)
        builder = root / 'builder'
        run('uv', 'venv', str(builder), '--python', python)
        interpreter = str(builder / 'bin/python')
        run('uv', 'pip', 'install', '--python', interpreter, 'maturin>=1.13,<2')
        for package, module in PACKAGES.items():
            # MPI is distributed as source and exercised on standard CPython.
            if package == 'laddu-mpi' and python.endswith('t'):
                continue
            dist = root / package
            dist.mkdir()
            manifest = f'python/{package}/Cargo.toml'
            if not python.endswith('t'):
                run(interpreter, '-m', 'maturin', 'sdist', '--manifest-path', manifest, '--out', str(dist))
                sdists = list(dist.glob('*.tar.gz'))
                if len(sdists) != 1:
                    msg = f'Expected one {package} sdist'
                    raise ValueError(msg)
                check_sdist(sdists[0], package)
            if package != 'laddu-mpi':
                run(
                    interpreter,
                    '-m',
                    'maturin',
                    'build',
                    '--locked',
                    '--profile',
                    'dev',
                    '--generate-stubs',
                    '--features',
                    'pyo3/abi3-py311,pyo3/abi3t-py315',
                    '--manifest-path',
                    manifest,
                    '--interpreter',
                    interpreter,
                    '--out',
                    str(dist),
                )
                artifacts = list(dist.glob('*.whl'))
                if len(artifacts) != 1:
                    msg = f'Expected one {package} wheel for {python}'
                    raise ValueError(msg)
            else:
                artifacts = sdists
            installed = root / f'{package}-installed'
            run('uv', 'venv', str(installed), '--python', python)
            installed_python = str(installed / 'bin/python')
            run(
                'uv',
                'pip',
                'install',
                '--python',
                installed_python,
                '--config-setting',
                'maturin.build-args=--profile dev --generate-stubs --locked',
                str(artifacts[0]),
            )
            run(
                installed_python,
                'python/tests/installed_artifact_contract.py',
                '--module',
                module,
                '--backend',
                'mpi' if package == 'laddu-mpi' else 'local',
                '--distribution',
                package,
            )


if __name__ == '__main__':
    main()
