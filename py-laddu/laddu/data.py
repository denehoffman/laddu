from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, cast

import numpy as np

from laddu.laddu import BinnedDataset, Event
from laddu.laddu import Dataset as _DatasetCore
from laddu.utils.vectors import Vec4

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import uproot
    from numpy.typing import NDArray


_NATIVE_DATASET_FROM_PARQUET = _DatasetCore.from_parquet
_NATIVE_DATASET_FROM_ROOT = _DatasetCore.from_root
_NATIVE_DATASET_TO_PARQUET = _DatasetCore.to_parquet
_NATIVE_DATASET_TO_ROOT = _DatasetCore.to_root


def _import_optional_dependency(
    module_name: str,
    *,
    extra: str,
    feature: str,
) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        msg = (
            f"{feature} requires the optional dependency '{module_name}'. "
            f'Install it with `pip install laddu[{extra}]` '
            f'or `pip install laddu-mpi[{extra}]`.'
        )
        raise ModuleNotFoundError(msg) from exc


class _DatasetExtensions:
    """In-memory event container backed by :class:`laddu.laddu.Event` objects.

    The helper constructors accept ``dict`` objects, pandas/Polars frames, or
    numpy arrays and ensure that the expected four-momentum columns (``*_px``,
    ``*_py``, ``*_pz``, ``*_e``) are present.

    Examples
    --------
    >>> columns = {
    ...     'beam_px': [0.0], 'beam_py': [0.0], 'beam_pz': [9.0], 'beam_e': [9.5],
    ...     'k_pi_px': [0.2], 'k_pi_py': [0.1], 'k_pi_pz': [0.3], 'k_pi_e': [0.6],
    ...     'weight': [1.0], 'pol_magnitude': [0.5], 'pol_angle': [0.0],
    ... }
    >>> dataset = Dataset.from_dict(columns)
    >>> len(dataset)
    1
    """

    @staticmethod
    def _infer_p4_names(columns: dict[str, Any]) -> list[str]:
        p4_names: list[str] = []
        for key in columns:
            if key.endswith('_px'):
                base = key[:-3]
                if base not in p4_names:
                    required = [f'{base}_{suffix}' for suffix in ('px', 'py', 'pz', 'e')]
                    missing = [name for name in required if name not in columns]
                    if missing:
                        msg = f"Missing components {missing} for four-momentum '{base}'"
                        raise KeyError(msg)
                    p4_names.append(base)
        if not p4_names:
            msg = 'No four-momentum columns found (expected *_px, *_py, *_pz, *_e)'
            raise ValueError(msg)
        return p4_names

    @staticmethod
    def _infer_aux_names(columns: dict[str, Any], used: set[str]) -> list[str]:
        aux_names: list[str] = []
        for key in columns:
            if key == 'weight' or key in used:
                continue
            aux_names.append(key)
        return aux_names

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> _DatasetCore:
        """Create a dataset from iterables keyed by column name.

        Parameters
        ----------
        data:
            Mapping whose keys are column names (e.g. ``beam_px``) and values are
            indexable sequences.
        p4s:
            Explicit four-momentum names; when omitted they are inferred from
            ``*_px`` columns.
        aux:
            Explicit auxiliary column names; inferred when omitted.
        aliases:
            Optional mapping of alias name to one or more particle identifiers.
        """
        columns = {name: np.asarray(values) for name, values in data.items()}

        if p4s is None:
            p4_names = cls._infer_p4_names(columns)
        else:
            p4_names = list(p4s)
            for name in p4_names:
                required = [f'{name}_{suffix}' for suffix in ('px', 'py', 'pz', 'e')]
                missing = [col for col in required if col not in columns]
                if missing:
                    msg = f"Missing components {missing} for four-momentum '{name}'"
                    raise KeyError(msg)

        component_names = {f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')}

        if aux is None:
            aux_names = cls._infer_aux_names(columns, component_names)
        else:
            aux_names = list(aux)
            missing_aux = [name for name in aux_names if name not in columns]
            if missing_aux:
                msg = f'Missing auxiliary columns {missing_aux}'
                raise KeyError(msg)

        n_events = len(columns[f'{p4_names[0]}_px'])
        weights = np.asarray(columns.get('weight', np.ones(n_events, dtype=float)), dtype=float)

        events: list[Event] = []
        for i in range(n_events):
            p4_vectors = [
                Vec4.from_array(
                    [
                        float(columns[f'{name}_px'][i]),
                        float(columns[f'{name}_py'][i]),
                        float(columns[f'{name}_pz'][i]),
                        float(columns[f'{name}_e'][i]),
                    ]
                )
                for name in p4_names
            ]
            aux_values = [float(columns[name][i]) for name in aux_names]
            events.append(
                Event(
                    p4_vectors,
                    aux_values,
                    float(weights[i]),
                    p4_names=p4_names,
                    aux_names=aux_names,
                )
            )

        native_aliases = dict(aliases) if aliases is not None else None
        return cls(
            events,
            p4_names=p4_names,
            aux_names=aux_names,
            aliases=native_aliases,
        )

    @classmethod
    def from_numpy(
        cls,
        data: dict[str, NDArray[np.floating]],
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> _DatasetCore:
        """Create a dataset from arrays without copying.

        Accepts any mapping of column names to ``ndarray`` objects and mirrors
        :meth:`from_dict`.
        """
        converted = {key: np.asarray(value) for key, value in data.items()}
        return cls.from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> _DatasetCore:
        """Materialise a dataset from a :class:`pandas.DataFrame`."""
        _import_optional_dependency(
            'pandas',
            extra='pandas',
            feature='Dataset.from_pandas',
        )
        converted = {col: data[col].to_list() for col in data.columns}
        return cls.from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)

    @classmethod
    def from_polars(
        cls,
        data: pl.DataFrame,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> _DatasetCore:
        """Materialise a dataset from a :class:`polars.DataFrame`."""
        _import_optional_dependency(
            'polars',
            extra='polars',
            feature='Dataset.from_polars',
        )
        converted = {col: data[col].to_list() for col in data.columns}
        return cls.from_dict(converted, p4s=p4s, aux=aux, aliases=aliases)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> _DatasetCore:
        """Read a Dataset from a Parquet file."""
        native_aliases = dict(aliases) if aliases is not None else None
        return _NATIVE_DATASET_FROM_PARQUET(
            path,
            p4s=p4s,
            aux=aux,
            aliases=native_aliases,
        )

    @classmethod
    def from_root(
        cls,
        path: str | Path,
        *,
        tree: str | None = None,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
        uproot_kwargs: dict[str, Any] | None = None,
    ) -> _DatasetCore:
        """Read a Dataset from a ROOT TTree.

        The native oxyroot backend is used by default. Pass ``backend='uproot'`` to read
        ROOT files via the Python ``uproot`` library.
        """
        backend_name = backend.lower() if backend else 'oxyroot'
        native_aliases = dict(aliases) if aliases is not None else None

        if backend_name not in {'oxyroot', 'uproot'}:
            msg = f"Unsupported backend '{backend_name}'. Valid options are 'oxyroot' or 'uproot'."
            raise ValueError(msg)

        if backend_name == 'oxyroot':
            return _NATIVE_DATASET_FROM_ROOT(
                path,
                tree=tree,
                p4s=p4s,
                aux=aux,
                aliases=native_aliases,
            )

        kwargs = dict(uproot_kwargs or {})
        backend_tree = tree or kwargs.pop('tree', None)
        return cls._open_with_uproot(
            Path(path),
            tree=backend_tree,
            p4s=p4s,
            aux=aux,
            aliases=native_aliases,
            uproot_kwargs=kwargs,
        )

    @classmethod
    def from_amptools(
        cls,
        path: str | Path,
        *,
        tree: str | None = None,
        pol_in_beam: bool = False,
        pol_angle: float | None = None,
        pol_magnitude: float | None = None,
        pol_magnitude_name: str = 'pol_magnitude',
        pol_angle_name: str = 'pol_angle',
        num_entries: int | None = None,
    ) -> _DatasetCore:
        """Read a Dataset from an AmpTools ROOT tuple."""
        return cls._open_amptools_format(
            Path(path),
            tree=tree,
            pol_in_beam=pol_in_beam,
            pol_angle=pol_angle,
            pol_magnitude=pol_magnitude,
            pol_magnitude_name=pol_magnitude_name,
            pol_angle_name=pol_angle_name,
            num_entries=num_entries,
        )

    def to_numpy(self: _DatasetCore, *, precision: str = 'f64') -> dict[str, np.ndarray]:
        """Convert the dataset to NumPy column arrays."""
        return _coalesce_numpy_batches(_iter_numpy_batches(self, chunk_size=len(self), precision=precision))

    def to_parquet(
        self: _DatasetCore,
        path: str | Path,
        *,
        chunk_size: int = 10_000,
        precision: str = 'f64',
    ) -> None:
        """Write the dataset to Parquet."""
        _NATIVE_DATASET_TO_PARQUET(
            self,
            path,
            chunk_size=chunk_size,
            precision=precision,
        )

    def to_root(
        self: _DatasetCore,
        path: str | Path,
        *,
        tree: str | None = None,
        backend: Literal['oxyroot', 'uproot'] = 'oxyroot',
        chunk_size: int = 10_000,
        precision: str = 'f64',
        uproot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Write the dataset to a ROOT TTree."""
        backend_name = backend.lower() if backend else 'oxyroot'
        if backend_name not in {'oxyroot', 'uproot'}:
            msg = f"Unsupported backend '{backend_name}'. Valid options are 'oxyroot' or 'uproot'."
            raise ValueError(msg)

        if backend_name == 'oxyroot':
            _NATIVE_DATASET_TO_ROOT(
                self,
                path,
                tree=tree,
                chunk_size=chunk_size,
                precision=precision,
            )
            return

        kwargs = dict(uproot_kwargs or {})
        tree_name = tree or kwargs.pop('tree', 'events')
        uproot_module = _import_optional_dependency(
            'uproot',
            extra='uproot',
            feature="Dataset.to_root(... backend='uproot')",
        )

        with uproot_module.recreate(path) as root_file:
            batches = _iter_numpy_batches(self, chunk_size=chunk_size, precision=precision)
            tree_obj = None
            for batch in batches:
                if tree_obj is None:
                    tree_obj = root_file.mktree(tree_name, batch)
                tree_obj.extend(batch, **kwargs)

            if tree_obj is None:
                root_file.mktree(tree_name, {})

    @classmethod
    def _open_with_uproot(
        cls,
        path: Path,
        *,
        tree: str | None,
        p4s: list[str] | None,
        aux: list[str] | None,
        aliases: Mapping[str, str | Sequence[str]] | None,
        uproot_kwargs: dict[str, Any],
    ) -> _DatasetCore:
        uproot_module = _import_optional_dependency(
            'uproot',
            extra='uproot',
            feature="Dataset.from_root(... backend='uproot')",
        )
        with uproot_module.open(path) as root_file:
            tree_obj = cls._select_uproot_tree(root_file, tree)
            arrays = tree_obj.arrays(library='np', **uproot_kwargs)

        columns = {name: np.asarray(values) for name, values in arrays.items()}
        selected = cls._prepare_uproot_columns(columns, p4s=p4s, aux=aux)
        return cls.from_numpy(selected, p4s=p4s, aux=aux, aliases=aliases)

    @classmethod
    def _open_amptools_format(
        cls,
        path: Path,
        *,
        tree: str | None,
        pol_in_beam: bool,
        pol_angle: float | None,
        pol_magnitude: float | None,
        pol_magnitude_name: str,
        pol_angle_name: str,
        num_entries: int | None,
    ) -> _DatasetCore:
        pol_angle_rad = pol_angle * np.pi / 180 if pol_angle is not None else None
        polarisation_requested = pol_in_beam or (pol_angle is not None and pol_magnitude is not None)
        p4s_list, aux_rows, weight_list = _read_amptools_events(
            path,
            tree or 'kin',
            pol_in_beam=pol_in_beam,
            pol_angle_rad=pol_angle_rad,
            pol_magnitude=pol_magnitude,
            num_entries=num_entries,
        )

        if not p4s_list:
            msg = 'AmpTools source produced no events'
            raise ValueError(msg)

        n_particles = len(p4s_list[0])
        if n_particles == 0:
            msg = 'AmpTools source produced no particles'
            raise ValueError(msg)

        p4_names = ['beam']
        if n_particles > 1:
            p4_names.extend(f'final_state_{i}' for i in range(n_particles - 1))

        aux_names: list[str] = []
        if aux_rows and aux_rows[0]:
            if polarisation_requested and len(aux_rows[0]) >= 2:
                aux_names = [pol_magnitude_name, pol_angle_name]
                extra = len(aux_rows[0]) - 2
                if extra > 0:
                    aux_names.extend(f'aux_{i}' for i in range(extra))
            else:
                aux_names = [f'aux_{i}' for i in range(len(aux_rows[0]))]

        events: list[Event] = []
        for p4s, aux, weight in zip(p4s_list, aux_rows, weight_list):
            p4_vectors = [Vec4.from_array(p4) for p4 in p4s]
            aux_values = [float(value) for value in aux]
            events.append(
                Event(
                    p4_vectors,
                    aux_values,
                    float(weight),
                    p4_names=p4_names,
                    aux_names=aux_names,
                )
            )
        return cls(events, p4_names=p4_names, aux_names=aux_names)

    @staticmethod
    def _select_uproot_tree(file: uproot.ReadOnlyDirectory, tree_name: str | None) -> uproot.TTree:
        if tree_name:
            try:
                return file[tree_name]
            except KeyError as exc:
                msg = f"Tree '{tree_name}' not found in ROOT file"
                raise KeyError(msg) from exc

        tree_candidates = [key.split(';')[0] for key, classname in file.classnames().items() if classname == 'TTree']
        if not tree_candidates:
            msg = 'ROOT file does not contain any TTrees'
            raise ValueError(msg)
        if len(tree_candidates) > 1:
            msg = f"Multiple TTrees found ({tree_candidates}); please specify the 'tree' argument"
            raise ValueError(msg)
        return file[tree_candidates[0]]

    @classmethod
    def _prepare_uproot_columns(
        cls,
        columns: dict[str, np.ndarray],
        *,
        p4s: list[str] | None,
        aux: list[str] | None,
    ) -> dict[str, np.ndarray]:
        if not columns:
            msg = 'ROOT tree does not contain any readable columns'
            raise ValueError(msg)

        data = {name: np.asarray(values) for name, values in columns.items()}
        p4_names = cls._infer_p4_names(data) if p4s is None else p4s

        component_columns = [f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')]
        missing_components = [col for col in component_columns if col not in data]
        if missing_components:
            msg = f'Missing components {missing_components} in ROOT data'
            raise KeyError(msg)

        used_components = set(component_columns)

        if aux is None:
            aux_names = cls._infer_aux_names(data, used_components)
        else:
            aux_names = aux
            missing_aux = [col for col in aux_names if col not in data]
            if missing_aux:
                msg = f'Missing auxiliary columns {missing_aux}'
                raise KeyError(msg)

        selected: dict[str, np.ndarray] = {}
        for name in component_columns:
            selected[name] = data[name]
        for name in aux_names:
            selected[name] = data[name]
        if 'weight' in data:
            selected['weight'] = data['weight']

        return selected


for _name, _attr in _DatasetExtensions.__dict__.items():
    if _name.startswith('__'):
        continue
    setattr(_DatasetCore, _name, _attr)

if TYPE_CHECKING:
    from .laddu import Dataset as Dataset
else:
    Dataset = cast('type[_DatasetExtensions]', _DatasetCore)
    Dataset.__doc__ = _DatasetExtensions.__doc__

DatasetBase = Dataset
del _name, _attr


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


def _empty_numpy_buffers(p4_names: Sequence[str], aux_names: Sequence[str]) -> dict[str, list[float]]:
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


def _iter_numpy_batches(
    dataset: _DatasetCore,
    *,
    chunk_size: int,
    precision: str,
) -> Iterable[dict[str, np.ndarray]]:
    normalized = precision.lower()
    if normalized in {'f64', 'float64', 'double'}:
        dtype = np.float64
    elif normalized in {'f32', 'float32', 'float'}:
        dtype = np.float32
    else:
        msg = "precision must be 'f64' or 'f32'"
        raise ValueError(msg)
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
            yield {key: np.asarray(values, dtype=dtype) for key, values in buffers.items()}
            buffers = _empty_numpy_buffers(p4_names, aux_names)
            count = 0

    if count:
        yield {key: np.asarray(values, dtype=dtype) for key, values in buffers.items()}


def _coalesce_numpy_batches(batches: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = {}
    for batch in batches:
        for key, array in batch.items():
            merged.setdefault(key, []).append(array)

    return {key: np.concatenate(arrays) if len(arrays) > 1 else arrays[0] for key, arrays in merged.items()}


def _read_amptools_scalar(branch: Any, *, entry_stop: int | None = None) -> np.ndarray:
    array = branch.array(library='np', entry_stop=entry_stop)
    return np.asarray(array, dtype=np.float32)


def _read_amptools_matrix(branch: Any, *, entry_stop: int | None = None) -> np.ndarray:
    raw = branch.array(library='np', entry_stop=entry_stop)
    return np.asarray(list(raw), dtype=np.float32)


def _load_amptools_arrays(
    path: Path,
    tree_name: str,
    *,
    entry_stop: int | None,
) -> tuple[np.ndarray, ...]:
    uproot_module = _import_optional_dependency(
        'uproot',
        extra='uproot',
        feature='Dataset.from_amptools',
    )
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
        pol_angle_arr = np.arctan2(py_beam.astype(np.float64), px_beam.astype(np.float64)).astype(np.float32)
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


def _read_amptools_events(
    path: Path,
    tree: str,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
    num_entries: int | None,
) -> tuple[list[list[np.ndarray]], list[list[float]], list[float]]:
    arrays = _load_amptools_arrays(path, tree, entry_stop=num_entries)
    data = _prepare_amptools_data(
        *arrays,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
    )

    n_events, n_finals = data.finals_e.shape

    p4s_list: list[list[np.ndarray]] = []
    for event_idx in range(n_events):
        event_vectors: list[np.ndarray] = [
            np.array(
                [
                    data.beam_px[event_idx],
                    data.beam_py[event_idx],
                    data.beam_pz[event_idx],
                    data.beam_e[event_idx],
                ],
                dtype=np.float32,
            )
        ]
        event_vectors.extend(
            [
                np.array(
                    [
                        data.finals_px[event_idx, final_idx],
                        data.finals_py[event_idx, final_idx],
                        data.finals_pz[event_idx, final_idx],
                        data.finals_e[event_idx, final_idx],
                    ],
                    dtype=np.float32,
                )
                for final_idx in range(n_finals)
            ]
        )
        p4s_list.append(event_vectors)

    if data.pol_magnitude is not None and data.pol_angle is not None:
        polarisation_values = np.column_stack((data.pol_magnitude, data.pol_angle))
        aux_rows = polarisation_values.astype(np.float32).tolist()
    else:
        aux_rows = [[] for _ in range(n_events)]

    weight_list = data.weights.tolist()

    return p4s_list, aux_rows, weight_list


__all__ = ['BinnedDataset', 'Dataset', 'Event']
