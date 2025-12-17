from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ['backend', 'backend_name', 'get_backend']

_ENV_VAR = 'LADDU_BACKEND'
_ALIAS_MAP = {
    'laddu_cpu': 'laddu_cpu',
    'laddu-cpu': 'laddu_cpu',
    'cpu': 'laddu_cpu',
    'laddu_mpi': 'laddu_mpi',
    'laddu-mpi': 'laddu_mpi',
    'mpi': 'laddu_mpi',
}
_DEFAULT_ORDER = ('laddu_mpi', 'laddu_cpu')
_backend: ModuleType | None = None
_backend_name: str | None = None


def _candidate_modules() -> tuple[str, ...]:
    forced = os.environ.get(_ENV_VAR)
    if forced:
        normalized = _ALIAS_MAP.get(forced.strip().lower())
        if normalized is None:
            msg = f"Unknown backend '{forced}'. Expected one of: {', '.join(sorted(set(_ALIAS_MAP)))}"
            raise ImportError(msg)
        return (normalized,)
    return _DEFAULT_ORDER


def get_backend() -> ModuleType:
    global _backend, _backend_name
    if _backend is not None:
        return _backend

    errors: list[str] = []
    for candidate in _candidate_modules():
        try:
            module = importlib.import_module(candidate)
        except ImportError as exc:  # pragma: no cover
            errors.append(f'{candidate}: {exc}')
            continue
        _backend = module
        _backend_name = candidate
        sys.modules.setdefault('laddu.laddu', module)
        return module

    detail = '; '.join(errors) if errors else 'unknown error'
    msg = (
        'Unable to import the laddu backend. Ensure that either the CPU or MPI '
        'extension is installed (e.g. `pip install laddu-cpu` or `laddu[mpi]`). '
        f'Details: {detail}'
    )
    raise ImportError(msg)


def backend_name() -> str:
    if _backend_name is None:
        get_backend()
    assert _backend_name is not None  # for type checkers
    return _backend_name


backend = get_backend()
