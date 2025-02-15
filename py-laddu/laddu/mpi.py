from __future__ import annotations

from typing import TYPE_CHECKING

from laddu.laddu import (
    finalize_mpi,
    get_rank,
    get_size,
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
    >>>     ...
    >>>
    >>> if __name__ == '__main__':
    >>>     with MPI(trigger=os.environ['MPI'] == '1'):
    >>>         main()

    This code will use MPI if and only if the environment variable "MPI" is set to "1", and it will properly finalize MPI when the context manager exits.

    """

    def __init__(self, *, trigger: bool = True) -> None:
        self._trigger = trigger

    def __enter__(self) -> None:
        if self._trigger:
            use_mpi()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._trigger:
            finalize_mpi()


__all__ = [
    'MPI',
    'finalize_mpi',
    'get_rank',
    'get_size',
    'is_root',
    'use_mpi',
    'using_mpi',
]
