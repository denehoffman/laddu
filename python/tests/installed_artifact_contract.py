"""Contract checks for an installed laddu Python distribution artifact."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

FIXTURE = Path(__file__).with_name('fixtures') / 'backend_exports.json'


def expected_exports() -> dict[str, list[str]]:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def assert_runtime_surface(module: ModuleType, expected_backend: str) -> None:
    actual_backend = module.backend()
    if actual_backend != expected_backend:
        message = f'{module.__name__}.backend() returned {actual_backend!r}, expected {expected_backend!r}'
        raise AssertionError(message)

    capabilities = module.capabilities()
    if capabilities['backend'] != expected_backend:
        message = (
            f"{module.__name__}.capabilities()['backend'] returned "
            f'{capabilities["backend"]!r}, expected {expected_backend!r}'
        )
        raise AssertionError(message)
    if capabilities['mpi'] is not (expected_backend == 'mpi'):
        message = f'{module.__name__} reported inconsistent MPI capability'
        raise AssertionError(message)

    for domain, names in expected_exports().items():
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            message = f'{module.__name__} is missing {domain} exports: {", ".join(missing)}'
            raise AssertionError(message)

    vector = module.Vec4(4.0, 1.0, 2.0, 0.5)
    invariant = vector.m2()
    if not isinstance(invariant, module.Expr) or not invariant.equation():
        message = f'{module.__name__}.Vec4 failed its installed expression smoke test'
        raise AssertionError(message)


def assert_distribution_contents(distribution: str, module: ModuleType) -> None:
    metadata = importlib.metadata.distribution(distribution)
    files = tuple(metadata.files or ())
    if not files:
        message = f'{distribution} has no installed file inventory'
        raise AssertionError(message)

    module_leaf = module.__name__.split('.')[-1]
    native_suffixes = ('.so', '.pyd', '.dylib')
    if not any(module_leaf in path.name and path.name.endswith(native_suffixes) for path in files):
        message = f'{distribution} does not contain the native module {module.__name__}'
        raise AssertionError(message)
    if not any(path.name == '__init__.pyi' and path.parts[0] == module_leaf for path in files):
        message = f'{distribution} does not contain {module_leaf}/__init__.pyi'
        raise AssertionError(message)


def check_installed_artifact(module_name: str, backend: str, distribution: str) -> None:
    module = importlib.import_module(module_name)
    assert_runtime_surface(module, backend)
    assert_distribution_contents(distribution, module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--backend', choices=('local', 'mpi'), required=True)
    parser.add_argument('--distribution', required=True)
    args = parser.parse_args()
    check_installed_artifact(args.module, args.backend, args.distribution)


if __name__ == '__main__':
    main()
