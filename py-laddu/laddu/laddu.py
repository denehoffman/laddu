from __future__ import annotations

from typing import Any

from ._backend import backend

__all__ = getattr(backend, '__all__', [])
__doc__ = getattr(backend, '__doc__', None)
__version__ = getattr(backend, '__version__', None)


def __getattr__(name: str) -> Any:
    return getattr(backend, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(backend)))
