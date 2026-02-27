from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
)

import numpy as np
from numpy.typing import NDArray

from laddu.laddu import from_columns as _backend_from_columns
from laddu.laddu import read_parquet as _backend_read_parquet
from laddu.laddu import read_root as _backend_read_root
from laddu.laddu import write_parquet as _backend_write_parquet
from laddu.laddu import write_root as _backend_write_root

from .data import Dataset

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa  # ty: ignore[unresolved-import]

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
ColumnInput: TypeAlias = Sequence[float] | FloatArray
NumpyColumns: TypeAlias = dict[str, np.ndarray]
UprootKwargValue: TypeAlias = str | int | float | bool | Sequence[str] | None
UprootKwargs: TypeAlias = dict[str, UprootKwargValue]


class _UprootBranch(Protocol):
    def array(
        self,
        *,
        library: Literal['np'],
        entry_stop: int | None = None,
    ) -> np.ndarray | Sequence[float]: ...


class _UprootTree(Protocol):
    def arrays(
        self,
        *,
        library: Literal['np'],
        **kwargs: object,
    ) -> Mapping[str, np.ndarray | Sequence[float]]: ...

    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, key: str) -> _UprootBranch: ...

    def __contains__(self, key: str) -> bool: ...


class _UprootFile(Protocol):
    def __getitem__(self, key: str) -> _UprootTree: ...

    def classnames(self) -> Mapping[str, str]: ...


def from_dict(
    data: Mapping[str, ColumnInput],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    normalized = _normalize_ingestion_columns(data)
    table = _build_arrow_table_from_columns(normalized)
    if table is not None:
        return _dataset_from_arrow_table(table, p4s=p4s, aux=aux, aliases=aliases)

    native_aliases = dict(aliases) if aliases is not None else None
    return _backend_from_columns(
        normalized,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
    )


def _normalize_ingestion_columns(data: Mapping[str, ColumnInput]) -> NumpyColumns:
    normalized: NumpyColumns = {}
    for name, values in data.items():
        column = np.asarray(values)
        if column.ndim != 1:
            msg = f"Column '{name}' must be one-dimensional"
            raise ValueError(msg)

        if column.dtype in (np.float32, np.float64):
            normalized[name] = np.ascontiguousarray(column)
            continue

        if column.dtype.kind in {'b', 'i', 'u', 'f'}:
            normalized[name] = np.ascontiguousarray(column.astype(np.float64, copy=False))
            continue

        try:
            normalized[name] = np.ascontiguousarray(column.astype(np.float64))
        except (TypeError, ValueError) as exc:
            msg = (
                f"Column '{name}' is not a numeric one-dimensional array and cannot be "
                'ingested by from_dict'
            )
            raise TypeError(msg) from exc

    return normalized


def _build_arrow_table_from_columns(columns: NumpyColumns) -> pa.Table | None:
    try:
        import pyarrow as pa  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        return None

    try:
        return pa.table(columns)
    except (TypeError, ValueError):
        return None


def _dataset_from_arrow_table(
    table: pa.Table,
    *,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: Mapping[str, str | Sequence[str]] | None,
) -> Dataset:
    converted = _arrow_table_to_numpy_columns(table)
    native_aliases = dict(aliases) if aliases is not None else None
    return _backend_from_columns(
        converted,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
    )


def _arrow_table_to_numpy_columns(table: pa.Table) -> NumpyColumns:
    column_names = list(table.column_names)
    converted: dict[str, np.ndarray] = {}
    for name in column_names:
        chunked = table[name]
        converted[name] = _chunked_array_to_numpy(chunked)
    return converted


def _chunked_array_to_numpy(chunked: pa.ChunkedArray) -> np.ndarray:
    if len(chunked.chunks) == 1:
        return _arrow_array_to_numpy(chunked.chunk(0))
    return np.concatenate([_arrow_array_to_numpy(chunk) for chunk in chunked.chunks])


def _arrow_array_to_numpy(array: pa.Array) -> np.ndarray:
    try:
        return np.asarray(array.to_numpy(zero_copy_only=True))
    except (TypeError, ValueError):
        return np.asarray(array.to_numpy(zero_copy_only=False))


def from_numpy(
    data: Mapping[str, FloatArray],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    converted = {key: np.asarray(value) for key, value in data.items()}
    return from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)


def from_pandas(
    data: pd.DataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    try:
        import pyarrow as pa  # ty: ignore[unresolved-import]

        table = pa.Table.from_pandas(data, preserve_index=False)
        return _dataset_from_arrow_table(table, p4s=p4s, aux=aux, aliases=aliases)
    except ModuleNotFoundError:
        converted = {col: data[col].to_numpy() for col in data.columns}
        return from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)


def from_polars(
    data: pl.DataFrame,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    try:
        table = data.to_arrow()
        return _dataset_from_arrow_table(table, p4s=p4s, aux=aux, aliases=aliases)
    except ModuleNotFoundError:
        converted = {name: data.get_column(name).to_numpy() for name in data.columns}
        return from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)


def read_parquet(
    path: str | Path,
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    native_aliases = dict(aliases) if aliases is not None else None
    return _backend_read_parquet(
        path,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
    )


def read_root(
    path: str | Path,
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
    backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
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
        Path(path),
        tree=backend_tree,
        p4s=p4s,
        aux=aux,
        aliases=native_aliases,
        uproot_kwargs=kwargs,
    )


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
) -> Dataset:
    return _open_amptools_format(
        Path(path),
        tree=tree,
        pol_in_beam=pol_in_beam,
        pol_angle=pol_angle,
        pol_magnitude=pol_magnitude,
        pol_magnitude_name=pol_magnitude_name,
        pol_angle_name=pol_angle_name,
        num_entries=num_entries,
    )


def to_numpy(
    dataset: Dataset,
    *,
    precision: Literal['f64', 'f32'] = 'f64',
) -> dict[str, np.ndarray]:
    return _coalesce_numpy_batches(
        _iter_numpy_batches(dataset, chunk_size=len(dataset), precision=precision)
    )


def write_parquet(
    dataset: Dataset,
    path: str | Path,
    *,
    chunk_size: int = 10_000,
    precision: Literal['f64', 'f32'] = 'f64',
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
    path: str | Path,
    *,
    tree: str | None = None,
    backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
    chunk_size: int = 10_000,
    precision: Literal['f64', 'f32'] = 'f64',
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
            "'uproot'. Install it with `pip install laddu[uproot]` or "
            '`pip install laddu-mpi[uproot]`.'
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
    path: Path,
    *,
    tree: str | None,
    p4s: list[str] | None,
    aux: list[str] | None,
    aliases: Mapping[str, str | Sequence[str]] | None,
    uproot_kwargs: UprootKwargs,
) -> Dataset:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_root(... backend='uproot') requires the optional dependency "
            "'uproot'. Install it with `pip install laddu[uproot]` or "
            '`pip install laddu-mpi[uproot]`.'
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

    columns = {name: np.asarray(values) for name, values in arrays.items()}
    if not columns:
        msg = 'ROOT tree does not contain any readable columns'
        raise ValueError(msg)
    return from_dict(columns, p4s=p4s, aux=aux, aliases=aliases)


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
    uproot_kwargs: UprootKwargs, selected_columns: Sequence[str] | None
) -> UprootKwargs:
    kwargs = dict(uproot_kwargs)
    if (
        selected_columns is not None
        and 'filter_name' not in kwargs
        and 'expressions' not in kwargs
    ):
        kwargs['filter_name'] = list(selected_columns)
    return kwargs


def _canonicalize_uproot_column_names(raw_keys: Iterable[str]) -> list[str]:
    names: list[str] = []
    for key in raw_keys:
        base = key.split(';', 1)[0]
        if base:
            names.append(base)
    return list(dict.fromkeys(names))


def _resolve_uproot_column_name(available: Sequence[str], logical_name: str) -> str:
    resolved = _resolve_uproot_column_name_optional(available, logical_name)
    if resolved is None:
        msg = f"Missing required ROOT column '{logical_name}'"
        raise KeyError(msg)
    return resolved


def _resolve_uproot_column_name_optional(
    available: Sequence[str], logical_name: str
) -> str | None:
    for name in available:
        if name == logical_name:
            return name
    for name in available:
        if name.lower() == logical_name.lower():
            return name
    return None


def _open_amptools_format(
    path: Path,
    *,
    tree: str | None,
    pol_in_beam: bool,
    pol_angle: float | None,
    pol_magnitude: float | None,
    pol_magnitude_name: str,
    pol_angle_name: str,
    num_entries: int | None,
) -> Dataset:
    pol_angle_rad = pol_angle * np.pi / 180 if pol_angle is not None else None
    polarisation_requested = pol_in_beam or (
        pol_angle is not None and pol_magnitude is not None
    )
    arrays = _load_amptools_arrays(path, tree or 'kin', entry_stop=num_entries)
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
    return from_dict(columns, p4s=p4_names, aux=aux_names)


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


@dataclass
class _AmpToolsData:
    beam_px: np.ndarray
    beam_py: np.ndarray
    beam_pz: np.ndarray
    beam_e: np.ndarray
    finals_px: np.ndarray
    finals_py: np.ndarray
    finals_pz: np.ndarray
    finals_e: np.ndarray
    weights: np.ndarray
    pol_magnitude: np.ndarray | None
    pol_angle: np.ndarray | None


def _empty_numpy_buffers(
    p4_names: Sequence[str], aux_names: Sequence[str]
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


def _validate_precision(value: str) -> Literal['f64', 'f32']:
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
    precision: Literal['f64', 'f32'],
) -> Iterable[dict[str, np.ndarray]]:
    validated_precision = _validate_precision(precision)
    dtype = np.float64 if validated_precision == 'f64' else np.float32
    p4_names = list(dataset.p4_names)
    aux_names = list(dataset.aux_names)

    buffers = _empty_numpy_buffers(p4_names, aux_names)
    count = 0

    for event in dataset:
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
                key: np.asarray(values, dtype=dtype) for key, values in buffers.items()
            }
            buffers = _empty_numpy_buffers(p4_names, aux_names)
            count = 0

    if count:
        yield {key: np.asarray(values, dtype=dtype) for key, values in buffers.items()}


def _coalesce_numpy_batches(
    batches: Iterable[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = {}
    for batch in batches:
        for key, array in batch.items():
            merged.setdefault(key, []).append(array)

    return {
        key: np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        for key, arrays in merged.items()
    }


def _read_amptools_scalar(
    branch: _UprootBranch,
    *,
    entry_stop: int | None = None,
) -> np.ndarray:
    array = branch.array(library='np', entry_stop=entry_stop)
    return np.asarray(array, dtype=np.float32)


def _read_amptools_matrix(
    branch: _UprootBranch,
    *,
    entry_stop: int | None = None,
) -> np.ndarray:
    raw = branch.array(library='np', entry_stop=entry_stop)
    return np.asarray(list(raw), dtype=np.float32)


def _load_amptools_arrays(
    path: Path,
    tree_name: str,
    *,
    entry_stop: int | None,
) -> tuple[np.ndarray, ...]:
    try:
        import uproot as uproot_module
    except ModuleNotFoundError as exc:
        msg = (
            "laddu.io.read_amptools requires the optional dependency 'uproot'. "
            'Install it with `pip install laddu[uproot]` or `pip install laddu-mpi[uproot]`.'
        )
        raise ModuleNotFoundError(msg) from exc
    with uproot_module.open(path) as file:
        try:
            tree = file[tree_name]
        except uproot_module.KeyInFileError as exc:
            msg = f"Input file must contain a tree named '{tree_name}'"
            raise KeyError(msg) from exc

        e_beam = _read_amptools_scalar(tree['E_Beam'], entry_stop=entry_stop)
        px_beam = _read_amptools_scalar(tree['Px_Beam'], entry_stop=entry_stop)
        py_beam = _read_amptools_scalar(tree['Py_Beam'], entry_stop=entry_stop)
        pz_beam = _read_amptools_scalar(tree['Pz_Beam'], entry_stop=entry_stop)

        e_final = _read_amptools_matrix(tree['E_FinalState'], entry_stop=entry_stop)
        px_final = _read_amptools_matrix(tree['Px_FinalState'], entry_stop=entry_stop)
        py_final = _read_amptools_matrix(tree['Py_FinalState'], entry_stop=entry_stop)
        pz_final = _read_amptools_matrix(tree['Pz_FinalState'], entry_stop=entry_stop)

        if 'Weight' in tree:
            weight = _read_amptools_scalar(tree['Weight'], entry_stop=entry_stop)
        else:
            weight = np.ones_like(e_beam, dtype=np.float32)

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


def _derive_amptools_polarization(
    px_beam: np.ndarray,
    py_beam: np.ndarray,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    beam_px = px_beam.copy()
    beam_py = py_beam.copy()
    pol_magnitude_arr: np.ndarray | None = None
    pol_angle_arr: np.ndarray | None = None

    if pol_in_beam:
        transverse_sq = px_beam.astype(np.float64) ** 2 + py_beam.astype(np.float64) ** 2
        pol_magnitude_arr = np.sqrt(transverse_sq).astype(np.float32)
        pol_angle_arr = np.arctan2(
            py_beam.astype(np.float64), px_beam.astype(np.float64)
        ).astype(np.float32)
        beam_px.fill(0.0)
        beam_py.fill(0.0)
    elif pol_angle_rad is not None and pol_magnitude is not None:
        n_events = px_beam.shape[0]
        pol_magnitude_arr = np.full(n_events, pol_magnitude, dtype=np.float32)
        pol_angle_arr = np.full(n_events, pol_angle_rad, dtype=np.float32)

    return beam_px, beam_py, pol_magnitude_arr, pol_angle_arr


def _prepare_amptools_data(
    e_beam: np.ndarray,
    px_beam: np.ndarray,
    py_beam: np.ndarray,
    pz_beam: np.ndarray,
    e_final: np.ndarray,
    px_final: np.ndarray,
    py_final: np.ndarray,
    pz_final: np.ndarray,
    weight: np.ndarray,
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
        weights=weight.astype(np.float32),
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
    'from_dict',
    'from_numpy',
    'from_pandas',
    'from_polars',
    'read_amptools',
    'read_parquet',
    'read_root',
    'to_numpy',
    'write_parquet',
    'write_root',
]
