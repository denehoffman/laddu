from __future__ import annotations as _annotations

from typing import Any as _Any

from ._backend import backend as _backend

__all__ = getattr(_backend, '__all__', [])
__doc__ = getattr(_backend, '__doc__', None)
__version__ = getattr(_backend, '__version__', None)


def __getattr__(name: str) -> _Any:
    return getattr(_backend, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_backend)))
