from __future__ import annotations

from typing import TYPE_CHECKING

from laddu.laddu import (
    finalize_mpi,
    get_rank,
    get_size,
    is_mpi_available,
    is_root,
    use_mpi,
    using_mpi,
)

if TYPE_CHECKING:
    from types import TracebackType


class MPI:
    r"""
    Context manager for using the MPI backend.

    Methods
    -------
    __init__(self, *, trigger: bool = True)
        Initialize the MPI context manager to use MPI if `trigger` is True.

    Examples
    --------
    >>> import os
    >>> from laddu.mpi import MPI
    >>>
    >>> def main():
    ...     pass
    >>>
    >>> if __name__ == '__main__':
    ...     with MPI(trigger=os.environ.get('MPI') == '1'):
    ...         main()

    """

    def __init__(self, *, trigger: bool = True) -> None:
        self._trigger = trigger

    def __enter__(self) -> None:
        use_mpi(trigger=self._trigger)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        finalize_mpi()


__all__ = [
    'MPI',
    'finalize_mpi',
    'get_rank',
    'get_size',
    'is_mpi_available',
    'is_root',
    'use_mpi',
    'using_mpi',
]
