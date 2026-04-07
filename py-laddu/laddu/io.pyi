from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .data import Dataset

PandasDataFrame: TypeAlias = Any
PolarsDataFrame: TypeAlias = Any

def from_dict(
    data: dict[str, Any],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def from_numpy(
    data: dict[str, NDArray[np.floating]],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def from_pandas(
    data: PandasDataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def from_polars(
    data: PolarsDataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def from_arrow(
    data: Any,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def read_parquet(
    path: str | Path,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> Dataset: ...
def read_parquet_chunked(
    path: str | Path,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
    chunk_size: int = 10000,
) -> Iterator[Dataset]: ...
def read_root(
    path: str | Path,
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
    backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
    uproot_kwargs: dict[str, Any] | None = None,
) -> Dataset: ...
def read_root_chunked(
    path: str | Path,
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
    backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
    uproot_kwargs: dict[str, Any] | None = None,
    chunk_size: int = 10000,
) -> Iterator[Dataset]: ...
def read_amptools(
    path: str | Path,
    *,
    tree: str | None = None,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    pol_magnitude_name: str = 'pol_magnitude',
    pol_angle_name: str = 'pol_angle',
    num_entries: int | None = None,
) -> Dataset: ...
def read_amptools_chunked(
    path: str | Path,
    *,
    tree: str | None = None,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    pol_magnitude_name: str = 'pol_magnitude',
    pol_angle_name: str = 'pol_angle',
    num_entries: int | None = None,
    chunk_size: int = 10000,
) -> Iterator[Dataset]: ...
def to_numpy(
    dataset: Dataset, *, precision: Literal['f64', 'f32'] = 'f64'
) -> dict[str, NDArray[np.floating]]: ...
def to_arrow(dataset: Dataset, *, precision: Literal['f64', 'f32'] = 'f64') -> Any: ...
def write_parquet(
    dataset: Dataset,
    path: str | Path,
    *,
    chunk_size: int = 10000,
    precision: Literal['f64', 'f32'] = 'f64',
) -> None: ...
def write_root(
    dataset: Dataset,
    path: str | Path,
    *,
    tree: str | None = None,
    backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
    chunk_size: int = 10000,
    precision: Literal['f64', 'f32'] = 'f64',
    uproot_kwargs: dict[str, Any] | None = None,
) -> None: ...

__all__ = [
    'from_arrow',
    'from_dict',
    'from_numpy',
    'from_pandas',
    'from_polars',
    'read_amptools',
    'read_amptools_chunked',
    'read_parquet',
    'read_parquet_chunked',
    'read_root',
    'read_root_chunked',
    'to_arrow',
    'to_numpy',
    'write_parquet',
    'write_root',
]
