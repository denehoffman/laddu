from types import TracebackType

def use_mpi(*, trigger: bool) -> None: ...
def finalize_mpi() -> None: ...
def using_mpi() -> bool: ...
def is_root() -> bool: ...
def get_rank() -> int: ...
def get_size() -> int: ...
def is_mpi_available() -> bool: ...

class MPI:
    """
    A context manager for MPI.

    Example
    -------
    .. code-block:: python

        import os
        from laddu.mpi import MPI

        def main():
            ...

        if __name__ == '__main__':
            with MPI(trigger=os.environ['MPI'] == '1'):
                main()
    """

    def __init__(self, *, trigger: bool = True) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

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
