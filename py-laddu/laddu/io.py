from __future__ import annotations

from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import (
    TYPE_CHECKING as _TYPE_CHECKING,
)
from typing import (
    Literal as _Literal,
)
from typing import (
    Protocol as _Protocol,
)
from typing import (
    TypeAlias as _TypeAlias,
)
from typing import (
    cast as _cast,
)

import numpy as _np  # noqa: ICN001
from numpy.typing import NDArray as _NDArray

from ._backend import backend as _backend_module
from .data import Dataset

_backend_from_columns = _backend_module.from_columns
_backend_read_parquet = _backend_module.read_parquet
_backend_read_parquet_chunked = _backend_module.read_parquet_chunked
_backend_read_root = _backend_module.read_root
_backend_write_parquet = _backend_module.write_parquet
_backend_write_root = _backend_module.write_root

if _TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    PandasDataFrame: _TypeAlias = pd.DataFrame
    PolarsDataFrame: _TypeAlias = pl.DataFrame
else:
    PandasDataFrame: _TypeAlias = object
    PolarsDataFrame: _TypeAlias = object

FloatArray: _TypeAlias = _NDArray[_np.float32] | _NDArray[_np.float64]
ColumnInput: _TypeAlias = _Sequence[float] | FloatArray
NumpyColumns: _TypeAlias = dict[str, _np.ndarray]
UprootKwargValue: _TypeAlias = str | int | float | bool | _Sequence[str] | None
UprootKwargs: _TypeAlias = dict[str, UprootKwargValue]

_OPTIONAL_DEPENDENCY_HINTS: dict[str, str] = {
    'uproot': 'Install it with `pip install laddu[uproot]` or `pip install laddu-mpi[uproot]`.',
    'pyarrow': 'Install it with `pip install pyarrow`.',
}


class _UprootBranch(_Protocol):
    def array(
        self,
        *,
        library: _Literal['np'],
        entry_start: int | None = None,
        entry_stop: int | None = None,
    ) -> _np.ndarray | _Sequence[float]: ...


class _UprootTree(_Protocol):
    num_entries: int

    def arrays(
        self,
        *,
        library: _Literal['np'],
        **kwargs: object,
    ) -> _Mapping[str, _np.ndarray | _Sequence[float]]: ...

    def keys(self) -> _Iterable[str]: ...

    def __getitem__(self, key: str) -> _UprootBranch: ...

    def __contains__(self, key: str) -> bool: ...


class _UprootFile(_Protocol):
    def __getitem__(self, key: str) -> _UprootTree: ...

    def classnames(self) -> _Mapping[str, str]: ...


def from_dict(
    data: _Mapping[str, ColumnInput],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    return from_columns(data, p4s=p4s, aux=aux, aliases=aliases)


def from_columns(
    data: _Mapping[str, ColumnInput],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    normalized = _normalize_ingestion_columns(data)
    return _backend_from_numpy_columns(normalized, p4s=p4s, aux=aux, aliases=aliases)


def _normalize_ingestion_columns(data: _Mapping[str, ColumnInput]) -> NumpyColumns:
    normalized: NumpyColumns = {}
    for name, values in data.items():
        column = _np.asarray(values)
        if column.ndim != 1:
            msg = f"Column '{name}' must be one-dimensional"
            raise ValueError(msg)

        if column.dtype in (_np.float32, _np.float64):
            normalized[name] = _np.ascontiguousarray(column)
            continue

        if column.dtype.kind in {'b', 'i', 'u', 'f'}:
            normalized[name] = _np.ascontiguousarray(
                column.astype(_np.float64, copy=False)
            )
            continue

        try:
            normalized[name] = _np.ascontiguousarray(column.astype(_np.float64))
        except (TypeError, ValueError) as exc:
            msg = (
                f"Column '{name}' is not a numeric one-dimensional array and cannot be "
                'ingested by from_dict'
            )
            raise TypeError(msg) from exc

    return normalized


def _dataset_from_arrow_table(
    table: pa.Table,
    *,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: _Mapping[str, str | _Sequence[str]] | None,
) -> Dataset:
    converted = _arrow_table_to_numpy_columns(table)
    return _backend_from_numpy_columns(converted, p4s=p4s, aux=aux, aliases=aliases)


def _backend_from_numpy_columns(
    columns: NumpyColumns,
    *,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: _Mapping[str, str | _Sequence[str]] | None,
) -> Dataset:
    native_aliases = dict(aliases) if aliases is not None else None
    return _backend_from_columns(
        columns,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
    )


def _arrow_table_to_numpy_columns(table: pa.Table) -> NumpyColumns:
    column_names = list(table.column_names)
    converted: dict[str, _np.ndarray] = {}
    for name in column_names:
        chunked = table[name]
        converted[name] = _chunked_array_to_numpy(chunked)
    return converted


def _chunked_array_to_numpy(chunked: pa.ChunkedArray) -> _np.ndarray:
    if len(chunked.chunks) == 1:
        return _arrow_array_to_numpy(chunked.chunk(0))
    return _np.concatenate([_arrow_array_to_numpy(chunk) for chunk in chunked.chunks])


def _arrow_array_to_numpy(array: pa.Array) -> _np.ndarray:
    try:
        return _np.asarray(array.to_numpy(zero_copy_only=True))
    except (TypeError, ValueError):
        return _np.asarray(array.to_numpy(zero_copy_only=False))


def _numpy_columns_to_arrow_table(columns: NumpyColumns) -> pa.Table:
    import pyarrow as pa

    names = list(columns)
    arrays = [pa.array(columns[name], from_pandas=False) for name in names]
    return pa.Table.from_arrays(arrays, names=names)


def from_numpy(
    data: _Mapping[str, FloatArray],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    converted = {key: _np.asarray(value) for key, value in data.items()}
    normalized = _normalize_ingestion_columns(converted)
    return _backend_from_numpy_columns(normalized, p4s=p4s, aux=aux, aliases=aliases)


def from_pandas(
    data: PandasDataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    try:
        import pyarrow as pa

        table = pa.Table.from_pandas(data, preserve_index=False)
        return from_arrow(table, p4s=p4s, aux=aux, aliases=aliases)
    except ModuleNotFoundError:
        converted = {col: data[col].to_numpy() for col in data.columns}
        normalized = _normalize_ingestion_columns(converted)
        return _backend_from_numpy_columns(normalized, p4s=p4s, aux=aux, aliases=aliases)


def from_arrow(
    data: pa.Table,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    return _dataset_from_arrow_table(data, p4s=p4s, aux=aux, aliases=aliases)


def from_polars(
    data: PolarsDataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    try:
        table = data.to_arrow()
        return from_arrow(table, p4s=p4s, aux=aux, aliases=aliases)
    except ModuleNotFoundError:
        converted = {name: data.get_column(name).to_numpy() for name in data.columns}
        normalized = _normalize_ingestion_columns(converted)
        return _backend_from_numpy_columns(normalized, p4s=p4s, aux=aux, aliases=aliases)


def read_parquet(
    path: str | _Path,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
) -> Dataset:
    native_aliases = dict(aliases) if aliases is not None else None
    return _backend_read_parquet(
        path,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
    )


def read_parquet_chunked(
    path: str | _Path,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
    chunk_size: int = 10_000,
) -> _Iterator[Dataset]:
    if chunk_size < 1:
        msg = 'chunk_size must be >= 1'
        raise ValueError(msg)
    native_aliases = dict(aliases) if aliases is not None else None
    chunks = _backend_read_parquet_chunked(
        path,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
        chunk_size=chunk_size,
    )
    yield from chunks


def read_root(
    path: str | _Path,
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
    backend: _Literal['oxyroot', 'uproot'] = 'oxyroot',
    uproot_kwargs: UprootKwargs | None = None,
) -> Dataset:
    backend_name = backend.lower() if backend else 'oxyroot'
    native_aliases = dict(aliases) if aliases is not None else None

    if backend_name not in {'oxyroot', 'uproot'}:
        msg = f"Unsupported backend '{backend_name}'. Valid options are 'oxyroot' or 'uproot'."
        raise ValueError(msg)

    if backend_name == 'oxyroot':
        return _backend_read_root(
            path,
            tree=tree,
            p4s=p4s,
            aux=aux,
            aliases=native_aliases,
        )

    kwargs = dict(uproot_kwargs or {})
    backend_tree = tree or kwargs.pop('tree', None)
    return _open_with_uproot(
        _Path(path),
        tree=backend_tree,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
        uproot_kwargs=kwargs,
    )


def read_root_chunked(
    path: str | _Path,
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: _Mapping[str, str | _Sequence[str]] | None = None,
    backend: _Literal['oxyroot', 'uproot'] = 'oxyroot',
    uproot_kwargs: UprootKwargs | None = None,
    chunk_size: int = 10_000,
) -> _Iterator[Dataset]:
    if chunk_size < 1:
        msg = 'chunk_size must be >= 1'
        raise ValueError(msg)
    backend_name = backend.lower() if backend else 'oxyroot'
    native_aliases = dict(aliases) if aliases is not None else None

    if backend_name not in {'oxyroot', 'uproot'}:
        msg = f"Unsupported backend '{backend_name}'. Valid options are 'oxyroot' or 'uproot'."
        raise ValueError(msg)

    if backend_name == 'oxyroot':
        # TODO(io): Implement true oxyroot chunked reads with entry windows.
        # Current behavior is a compatibility fallback that yields one fully materialized chunk.
        yield _backend_read_root(
            path,
            tree=tree,
            p4s=p4s,
            aux=aux,
            aliases=native_aliases,
        )
        return

    kwargs = dict(uproot_kwargs or {})
    if 'entry_start' in kwargs or 'entry_stop' in kwargs:
        msg = (
            'read_root_chunked controls entry windows internally; do not pass '
            "'entry_start' or 'entry_stop' in uproot_kwargs"
        )
        raise ValueError(msg)
    backend_tree = tree or kwargs.pop('tree', None)
    yield from _open_with_uproot_chunked(
        _Path(path),
        tree=backend_tree,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
        uproot_kwargs=kwargs,
        chunk_size=chunk_size,
    )


def read_amptools(
    path: str | _Path,
    *,
    tree: str | None = None,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    pol_magnitude_name: str = 'pol_magnitude',
    pol_angle_name: str = 'pol_angle',
    num_entries: int | None = None,
) -> Dataset:
    return _open_amptools_format(
        _Path(path),
        tree=tree,
        pol_in_beam=pol_in_beam,
        pol_angle=pol_angle,
        pol_magnitude=pol_magnitude,
        pol_magnitude_name=pol_magnitude_name,
        pol_angle_name=pol_angle_name,
        entry_start=None,
        entry_stop=num_entries,
    )


def read_amptools_chunked(
    path: str | _Path,
    *,
    tree: str | None = None,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    pol_magnitude_name: str = 'pol_magnitude',
    pol_angle_name: str = 'pol_angle',
    num_entries: int | None = None,
    chunk_size: int = 10_000,
) -> _Iterator[Dataset]:
    if chunk_size < 1:
        msg = 'chunk_size must be >= 1'
        raise ValueError(msg)
    tree_name = tree or 'kin'
    total_entries = _amptools_total_entries(_Path(path), tree_name)
    limit = min(total_entries, num_entries) if num_entries is not None else total_entries
    for start in range(0, limit, chunk_size):
        stop = min(start + chunk_size, limit)
        yield _open_amptools_format(
            _Path(path),
            tree=tree,
            pol_in_beam=pol_in_beam,
            pol_angle=pol_angle,
            pol_magnitude=pol_magnitude,
            pol_magnitude_name=pol_magnitude_name,
            pol_angle_name=pol_angle_name,
            entry_start=start,
            entry_stop=stop,
        )


def to_numpy(
    dataset: Dataset,
    *,
    precision: _Literal['f64', 'f32'] = 'f64',
) -> dict[str, _np.ndarray]:
    return _coalesce_numpy_batches(
        _iter_numpy_batches(dataset, chunk_size=len(dataset), precision=precision)
    )


def to_arrow(
    dataset: Dataset,
    *,
    precision: _Literal['f64', 'f32'] = 'f64',
) -> pa.Table:
    try:
        columns = to_numpy(dataset, precision=precision)
        return _numpy_columns_to_arrow_table(columns)
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.to_arrow requires the optional dependency 'pyarrow'. "
            f'{_OPTIONAL_DEPENDENCY_HINTS["pyarrow"]}'
        )
        raise ModuleNotFoundError(msg) from exc


def write_parquet(
    dataset: Dataset,
    path: str | _Path,
    *,
    chunk_size: int = 10_000,
    precision: _Literal['f64', 'f32'] = 'f64',
) -> None:
    validated_precision = _validate_precision(precision)
    _backend_write_parquet(
        dataset,
        path,
        chunk_size=chunk_size,
        precision=validated_precision,
    )


def write_root(
    dataset: Dataset,
    path: str | _Path,
    *,
    tree: str | None = None,
    backend: _Literal['oxyroot', 'uproot'] = 'oxyroot',
    chunk_size: int = 10_000,
    precision: _Literal['f64', 'f32'] = 'f64',
    uproot_kwargs: UprootKwargs | None = None,
) -> None:
    backend_name = backend.lower() if backend else 'oxyroot'
    if backend_name not in {'oxyroot', 'uproot'}:
        msg = f"Unsupported backend '{backend_name}'. Valid options are 'oxyroot' or 'uproot'."
        raise ValueError(msg)

    validated_precision = _validate_precision(precision)
    if backend_name == 'oxyroot':
        _backend_write_root(
            dataset,
            path,
            tree=tree,
            chunk_size=chunk_size,
            precision=validated_precision,
        )
        return

    kwargs = dict(uproot_kwargs or {})
    tree_name = tree or kwargs.pop('tree', 'events')
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.write_root(... backend='uproot') requires the optional dependency "
            f"'uproot'. {_OPTIONAL_DEPENDENCY_HINTS['uproot']}"
        )
        raise ModuleNotFoundError(msg) from exc

    with uproot_module.recreate(path) as root_file:
        batches = _iter_numpy_batches(
            dataset,
            chunk_size=chunk_size,
            precision=validated_precision,
        )
        tree_obj = None
        for batch in batches:
            if tree_obj is None:
                tree_obj = root_file.mktree(tree_name, batch)
            tree_obj.extend(batch, **kwargs)

        if tree_obj is None:
            root_file.mktree(tree_name, {})


def _open_with_uproot(
    path: _Path,
    *,
    tree: str | None,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: _Mapping[str, str | _Sequence[str]] | None,
    uproot_kwargs: UprootKwargs,
) -> Dataset:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_root(... backend='uproot') requires the optional dependency "
            f"'uproot'. {_OPTIONAL_DEPENDENCY_HINTS['uproot']}"
        )
        raise ModuleNotFoundError(msg) from exc
    with uproot_module.open(path) as root_file:
        tree_obj = _select_uproot_tree(root_file, tree)
        selected_columns = _build_uproot_selected_columns(
            tree_obj,
            p4s=p4s,
            aux=aux,
            include_weight=True,
        )
        kwargs = _uproot_arrays_kwargs(uproot_kwargs, selected_columns)
        arrays = tree_obj.arrays(library='np', **kwargs)

    columns = {name: _np.asarray(values) for name, values in arrays.items()}
    if not columns:
        msg = 'ROOT tree does not contain any readable columns'
        raise ValueError(msg)
    normalized = _normalize_ingestion_columns(columns)
    return _backend_from_numpy_columns(normalized, p4s=p4s, aux=aux, aliases=aliases)


def _open_with_uproot_chunked(
    path: _Path,
    *,
    tree: str | None,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: _Mapping[str, str | _Sequence[str]] | None,
    uproot_kwargs: UprootKwargs,
    chunk_size: int,
) -> _Iterator[Dataset]:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_root_chunked(... backend='uproot') requires the optional dependency "
            f"'uproot'. {_OPTIONAL_DEPENDENCY_HINTS['uproot']}"
        )
        raise ModuleNotFoundError(msg) from exc

    with uproot_module.open(path) as root_file:
        tree_obj = _select_uproot_tree(root_file, tree)
        total_entries = int(tree_obj.num_entries)
        selected_columns = _build_uproot_selected_columns(
            tree_obj,
            p4s=p4s,
            aux=aux,
            include_weight=True,
        )
        kwargs_base = _uproot_arrays_kwargs(uproot_kwargs, selected_columns)
        if 'entry_start' in kwargs_base or 'entry_stop' in kwargs_base:
            msg = (
                'read_root_chunked controls entry windows internally; do not pass '
                "'entry_start' or 'entry_stop' in uproot_kwargs"
            )
            raise ValueError(msg)

        for start in range(0, total_entries, chunk_size):
            stop = min(start + chunk_size, total_entries)
            kwargs = dict(kwargs_base)
            kwargs['entry_start'] = start
            kwargs['entry_stop'] = stop
            arrays = tree_obj.arrays(library='np', **kwargs)
            columns = {name: _np.asarray(values) for name, values in arrays.items()}
            if not columns:
                msg = 'ROOT tree does not contain any readable columns'
                raise ValueError(msg)
            normalized = _normalize_ingestion_columns(columns)
            yield _backend_from_numpy_columns(
                normalized, p4s=p4s, aux=aux, aliases=aliases
            )


def _build_uproot_selected_columns(
    tree: _UprootTree,
    *,
    p4s: list[str] | None,
    aux: list[str] | None,
    include_weight: bool,
) -> list[str] | None:
    if p4s is None:
        return None

    available = _canonicalize_uproot_column_names(tree.keys())
    selected: list[str] = []

    for name in p4s:
        for suffix in ('_px', '_py', '_pz', '_e'):
            logical = f'{name}{suffix}'
            selected.append(_resolve_uproot_column_name(available, logical))

    selected.extend(_resolve_uproot_column_name(available, name) for name in aux or [])

    if include_weight:
        weight = _resolve_uproot_column_name_optional(available, 'weight')
        if weight is not None:
            selected.append(weight)

    # Preserve first-seen order and avoid duplicates.
    return list(dict.fromkeys(selected))


def _uproot_arrays_kwargs(
    uproot_kwargs: UprootKwargs, selected_columns: _Sequence[str] | None
) -> UprootKwargs:
    kwargs = dict(uproot_kwargs)
    if (
        selected_columns is not None
        and 'filter_name' not in kwargs
        and 'expressions' not in kwargs
    ):
        kwargs['filter_name'] = list(selected_columns)
    return kwargs


def _canonicalize_uproot_column_names(raw_keys: _Iterable[str]) -> list[str]:
    names: list[str] = []
    for key in raw_keys:
        base = key.split(';', 1)[0]
        if base:
            names.append(base)
    return list(dict.fromkeys(names))


def _resolve_uproot_column_name(available: _Sequence[str], logical_name: str) -> str:
    resolved = _resolve_uproot_column_name_optional(available, logical_name)
    if resolved is None:
        msg = f"Missing required ROOT column '{logical_name}'"
        raise KeyError(msg)
    return resolved


def _resolve_uproot_column_name_optional(
    available: _Sequence[str], logical_name: str
) -> str | None:
    for name in available:
        if name == logical_name:
            return name
    for name in available:
        if name.lower() == logical_name.lower():
            return name
    return None


def _open_amptools_format(
    path: _Path,
    *,
    tree: str | None,
    pol_in_beam: bool,
    pol_angle: float | None,
    pol_magnitude: float | None,
    pol_magnitude_name: str,
    pol_angle_name: str,
    entry_start: int | None,
    entry_stop: int | None,
) -> Dataset:
    pol_angle_rad = pol_angle * _np.pi / 180 if pol_angle is not None else None
    polarisation_requested = pol_in_beam or (
        pol_angle is not None and pol_magnitude is not None
    )
    arrays = _load_amptools_arrays(
        path,
        tree or 'kin',
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    amptools_data = _prepare_amptools_data(
        *arrays,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
    )
    columns, p4_names, aux_names = _amptools_columns(
        amptools_data,
        pol_in_beam=pol_in_beam,
        polarisation_requested=polarisation_requested,
        pol_magnitude_name=pol_magnitude_name,
        pol_angle_name=pol_angle_name,
    )
    normalized = _normalize_ingestion_columns(columns)
    return _backend_from_numpy_columns(
        normalized, p4s=p4_names, aux=aux_names, aliases=None
    )


def _select_uproot_tree(file: _UprootFile, tree_name: str | None) -> _UprootTree:
    if tree_name:
        try:
            return file[tree_name]
        except KeyError as exc:
            msg = f"Tree '{tree_name}' not found in ROOT file"
            raise KeyError(msg) from exc

    tree_candidates = [
        key.split(';')[0]
        for key, classname in file.classnames().items()
        if classname == 'TTree'
    ]
    if not tree_candidates:
        msg = 'ROOT file does not contain any TTrees'
        raise ValueError(msg)
    if len(tree_candidates) > 1:
        msg = f"Multiple TTrees found ({tree_candidates}); please specify the 'tree' argument"
        raise ValueError(msg)
    return file[tree_candidates[0]]


@_dataclass
class _AmpToolsData:
    beam_px: _np.ndarray
    beam_py: _np.ndarray
    beam_pz: _np.ndarray
    beam_e: _np.ndarray
    finals_px: _np.ndarray
    finals_py: _np.ndarray
    finals_pz: _np.ndarray
    finals_e: _np.ndarray
    weights: _np.ndarray
    pol_magnitude: _np.ndarray | None
    pol_angle: _np.ndarray | None


def _empty_numpy_buffers(
    p4_names: _Sequence[str], aux_names: _Sequence[str]
) -> dict[str, list[float]]:
    buffers: dict[str, list[float]] = (
        {f'{name}_px': [] for name in p4_names}
        | {f'{name}_py': [] for name in p4_names}
        | {f'{name}_pz': [] for name in p4_names}
        | {f'{name}_e': [] for name in p4_names}
    )
    for name in aux_names:
        buffers[name] = []
    buffers['weight'] = []
    return buffers


def _validate_precision(value: str) -> _Literal['f64', 'f32']:
    normalized = value.lower()
    if normalized == 'f64':
        return 'f64'
    if normalized == 'f32':
        return 'f32'
    msg = "precision must be 'f64' or 'f32'"
    raise ValueError(msg)


def _iter_numpy_batches(
    dataset: Dataset,
    *,
    chunk_size: int,
    precision: _Literal['f64', 'f32'],
) -> _Iterable[dict[str, _np.ndarray]]:
    validated_precision = _validate_precision(precision)
    dtype = _np.float64 if validated_precision == 'f64' else _np.float32
    p4_names = list(dataset.p4_names)
    aux_names = list(dataset.aux_names)

    buffers = _empty_numpy_buffers(p4_names, aux_names)
    count = 0

    for event in dataset.events_global:
        p4_map = event.p4s
        for name in p4_names:
            vec = p4_map[name]
            buffers[f'{name}_px'].append(float(vec.px))
            buffers[f'{name}_py'].append(float(vec.py))
            buffers[f'{name}_pz'].append(float(vec.pz))
            buffers[f'{name}_e'].append(float(vec.e))

        aux_map = event.aux
        for name in aux_names:
            buffers[name].append(float(aux_map[name]))

        buffers['weight'].append(float(event.weight))
        count += 1

        if count >= chunk_size:
            yield {
                key: _np.asarray(values, dtype=dtype) for key, values in buffers.items()
            }
            buffers = _empty_numpy_buffers(p4_names, aux_names)
            count = 0

    if count:
        yield {key: _np.asarray(values, dtype=dtype) for key, values in buffers.items()}


def _coalesce_numpy_batches(
    batches: _Iterable[dict[str, _np.ndarray]],
) -> dict[str, _np.ndarray]:
    merged: dict[str, list[_np.ndarray]] = {}
    for batch in batches:
        for key, array in batch.items():
            merged.setdefault(key, []).append(array)

    return {
        key: _np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        for key, arrays in merged.items()
    }


def _read_amptools_scalar(
    branch: _UprootBranch,
    *,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> _np.ndarray:
    array = branch.array(library='np', entry_start=entry_start, entry_stop=entry_stop)
    return _np.asarray(array, dtype=_np.float32)


def _read_amptools_matrix(
    branch: _UprootBranch,
    *,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> _np.ndarray:
    raw = branch.array(library='np', entry_start=entry_start, entry_stop=entry_stop)
    array = _np.asarray(raw)
    if array.dtype == object:
        return _amptools_object_rows_to_matrix(_cast(_Sequence[_np.ndarray], array))
    return _np.asarray(array, dtype=_np.float32)


def _amptools_object_rows_to_matrix(rows: _Sequence[_np.ndarray]) -> _np.ndarray:
    if len(rows) == 0:
        return _np.empty((0, 0), dtype=_np.float32)
    first = _np.asarray(rows[0], dtype=_np.float32)
    n_rows = len(rows)
    n_cols = int(first.shape[0])
    out = _np.empty((n_rows, n_cols), dtype=_np.float32)
    out[0, :] = first
    for row_index, row in enumerate(rows[1:], start=1):
        out[row_index, :] = _np.asarray(row, dtype=_np.float32)
    return out


def _load_amptools_arrays(
    path: _Path,
    tree_name: str,
    *,
    entry_start: int | None,
    entry_stop: int | None,
) -> tuple[_np.ndarray, ...]:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_amptools requires the optional dependency 'uproot'. "
            f'{_OPTIONAL_DEPENDENCY_HINTS["uproot"]}'
        )
        raise ModuleNotFoundError(msg) from exc
    with uproot_module.open(path) as file:
        try:
            tree = file[tree_name]
        except uproot_module.KeyInFileError as exc:
            msg = f"Input file must contain a tree named '{tree_name}'"
            raise KeyError(msg) from exc

        e_beam = _read_amptools_scalar(
            tree['E_Beam'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        px_beam = _read_amptools_scalar(
            tree['Px_Beam'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        py_beam = _read_amptools_scalar(
            tree['Py_Beam'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        pz_beam = _read_amptools_scalar(
            tree['Pz_Beam'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )

        e_final = _read_amptools_matrix(
            tree['E_FinalState'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        px_final = _read_amptools_matrix(
            tree['Px_FinalState'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        py_final = _read_amptools_matrix(
            tree['Py_FinalState'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        pz_final = _read_amptools_matrix(
            tree['Pz_FinalState'],
            entry_start=entry_start,
            entry_stop=entry_stop,
        )

        if 'Weight' in tree:
            weight = _read_amptools_scalar(
                tree['Weight'],
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
        else:
            weight = _np.ones_like(e_beam, dtype=_np.float32)

    return (
        e_beam,
        px_beam,
        py_beam,
        pz_beam,
        e_final,
        px_final,
        py_final,
        pz_final,
        weight,
    )


def _amptools_total_entries(path: _Path, tree_name: str) -> int:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_amptools_chunked requires the optional dependency 'uproot'. "
            f'{_OPTIONAL_DEPENDENCY_HINTS["uproot"]}'
        )
        raise ModuleNotFoundError(msg) from exc
    with uproot_module.open(path) as file:
        try:
            tree = file[tree_name]
        except uproot_module.KeyInFileError as exc:
            msg = f"Input file must contain a tree named '{tree_name}'"
            raise KeyError(msg) from exc
        return int(tree.num_entries)


def _derive_amptools_polarization(
    px_beam: _np.ndarray,
    py_beam: _np.ndarray,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray | None, _np.ndarray | None]:
    beam_px = px_beam
    beam_py = py_beam
    pol_magnitude_arr: _np.ndarray | None = None
    pol_angle_arr: _np.ndarray | None = None

    if pol_in_beam:
        beam_px = _np.zeros_like(px_beam, dtype=_np.float32)
        beam_py = _np.zeros_like(py_beam, dtype=_np.float32)
        pol_magnitude_arr = _np.hypot(px_beam, py_beam).astype(_np.float32, copy=False)
        pol_angle_arr = _np.arctan2(py_beam, px_beam).astype(_np.float32, copy=False)
    elif pol_angle_rad is not None and pol_magnitude is not None:
        n_events = px_beam.shape[0]
        pol_magnitude_arr = _np.full(n_events, pol_magnitude, dtype=_np.float32)
        pol_angle_arr = _np.full(n_events, pol_angle_rad, dtype=_np.float32)

    return beam_px, beam_py, pol_magnitude_arr, pol_angle_arr


def _prepare_amptools_data(
    e_beam: _np.ndarray,
    px_beam: _np.ndarray,
    py_beam: _np.ndarray,
    pz_beam: _np.ndarray,
    e_final: _np.ndarray,
    px_final: _np.ndarray,
    py_final: _np.ndarray,
    pz_final: _np.ndarray,
    weight: _np.ndarray,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
) -> _AmpToolsData:
    n_events, n_finals = e_final.shape
    if not (px_final.shape == py_final.shape == pz_final.shape == (n_events, n_finals)):
        msg = 'Final-state branches must have a consistent shape'
        raise ValueError(msg)

    beam_px, beam_py, pol_magnitude_arr, pol_angle_arr = _derive_amptools_polarization(
        px_beam,
        py_beam,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
    )

    return _AmpToolsData(
        beam_px=beam_px,
        beam_py=beam_py,
        beam_pz=pz_beam,
        beam_e=e_beam,
        finals_px=px_final,
        finals_py=py_final,
        finals_pz=pz_final,
        finals_e=e_final,
        weights=weight.astype(_np.float32, copy=False),
        pol_magnitude=pol_magnitude_arr,
        pol_angle=pol_angle_arr,
    )


def _amptools_columns(
    data: _AmpToolsData,
    *,
    pol_in_beam: bool,
    polarisation_requested: bool,
    pol_magnitude_name: str,
    pol_angle_name: str,
) -> tuple[NumpyColumns, list[str], list[str]]:
    n_events, n_finals = data.finals_e.shape
    if n_events == 0:
        msg = 'AmpTools source produced no events'
        raise ValueError(msg)
    if n_finals == 0:
        msg = 'AmpTools source produced no particles'
        raise ValueError(msg)

    p4_names = ['beam', *(f'final_state_{i}' for i in range(n_finals))]
    columns: NumpyColumns = {
        'beam_px': data.beam_px,
        'beam_py': data.beam_py,
        'beam_pz': data.beam_pz,
        'beam_e': data.beam_e,
        'weight': data.weights,
    }
    for final_idx in range(n_finals):
        name = f'final_state_{final_idx}'
        columns[f'{name}_px'] = data.finals_px[:, final_idx]
        columns[f'{name}_py'] = data.finals_py[:, final_idx]
        columns[f'{name}_pz'] = data.finals_pz[:, final_idx]
        columns[f'{name}_e'] = data.finals_e[:, final_idx]

    aux_names: list[str] = []
    if data.pol_magnitude is not None and data.pol_angle is not None:
        aux_names = [pol_magnitude_name, pol_angle_name]
        columns[pol_magnitude_name] = data.pol_magnitude
        columns[pol_angle_name] = data.pol_angle
    elif pol_in_beam or polarisation_requested:
        msg = 'Polarization inputs were requested but no polarization data was available'
        raise ValueError(msg)

    return columns, p4_names, aux_names


__all__ = [
    'from_arrow',
    'from_columns',
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
