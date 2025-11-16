from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import numpy as np

from laddu.laddu import BinnedDataset, Event
from laddu.laddu import Dataset as _DatasetCore
from laddu.utils.vectors import Vec4

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import uproot
    from numpy.typing import NDArray


_NATIVE_DATASET_OPEN = _DatasetCore.open


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
            f"Install it with `pip install laddu[{extra}]` "
            f"or `pip install laddu-mpi[{extra}]`."
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
        if any(key.startswith('p4_') for key in columns):  # legacy format
            msg = 'Legacy column format detected (p4_N_*). Please run convert_legacy_parquet.py first.'
            raise ValueError(msg)
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
        cls, data: dict[str, Any], rest_frame_of: list[str] | None = None
    ) -> _DatasetCore:
        """Create a dataset from iterables keyed by column name.

        Parameters
        ----------
        data:
            Mapping whose keys are column names (e.g. ``beam_px``) and values are
            indexable sequences.
        rest_frame_of:
            Optional list of particle names whose combined rest frame should be
            used to boost each event (useful for quasi-two-body systems).
        """
        columns = {name: np.asarray(values) for name, values in data.items()}
        p4_names = cls._infer_p4_names(columns)
        component_names = {
            f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')
        }
        aux_names = cls._infer_aux_names(columns, component_names)

        n_events = len(columns[f'{p4_names[0]}_px'])
        weights = np.asarray(
            columns.get('weight', np.ones(n_events, dtype=float)), dtype=float
        )

        events: list[Event] = []
        for i in range(n_events):
            p4s = [
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
                    p4s,
                    aux_values,
                    float(weights[i]),
                    rest_frame_of=rest_frame_of,
                    p4_names=p4_names,
                    aux_names=aux_names,
                )
            )

        return cls(events, p4_names=p4_names, aux_names=aux_names)

    @classmethod
    def from_numpy(
        cls, data: dict[str, NDArray[np.floating]], rest_frame_of: list[str] | None = None
    ) -> _DatasetCore:
        """Create a dataset from arrays without copying.

        Accepts any mapping of column names to ``ndarray`` objects and mirrors
        :meth:`from_dict`.
        """
        converted = {key: np.asarray(value) for key, value in data.items()}
        return cls.from_dict(converted, rest_frame_of=rest_frame_of)

    @classmethod
    def from_pandas(
        cls, data: pd.DataFrame, rest_frame_of: list[str] | None = None
    ) -> _DatasetCore:
        """Materialise a dataset from a :class:`pandas.DataFrame`."""
        _import_optional_dependency(
            'pandas',
            extra='pandas',
            feature='Dataset.from_pandas',
        )
        converted = {col: data[col].to_list() for col in data.columns}
        return cls.from_dict(converted, rest_frame_of=rest_frame_of)

    @classmethod
    def from_polars(
        cls, data: pl.DataFrame, rest_frame_of: list[str] | None = None
    ) -> _DatasetCore:
        """Materialise a dataset from a :class:`polars.DataFrame`."""
        _import_optional_dependency(
            'polars',
            extra='polars',
            feature='Dataset.from_polars',
        )
        converted = {col: data[col].to_list() for col in data.columns}
        return cls.from_dict(converted, rest_frame_of=rest_frame_of)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        backend: str | None = None,
        tree: str | None = None,
        uproot_kwargs: dict[str, Any] | None = None,
        amptools_kwargs: dict[str, Any] | None = None,
    ) -> _DatasetCore:
        """Open a dataset from a file.

        Parameters
        ----------
        path:
            Parquet or ROOT file on disk.
        p4s:
            Ordered list of particle base names (e.g. ``['beam', 'kshort1']``).
        aux:
            Auxiliary scalar columns to retain (such as ``pol_magnitude``).
        aliases:
            Mapping of alternate particle identifiers to canonical ones. Values can be
            either a string (for simple renames like ``{'photon': 'beam'}``) or a
            sequence of strings when grouping multiple four-momenta together
            (e.g. ``{'ks_pair': ['kshort1', 'kshort2']}``). Aliases are only
            supported by the native (Rust) backend.
        backend:
            Backend to use for ROOT files. Supported values are ``'oxyroot'``
            (Rust loader, default for ``.root``), ``'uproot'`` (Python loader),
            ``'amptools'`` (AmpTools converter using ``uproot``), ``'native'`` /
            ``'parquet'`` (Rust loader for other formats), or ``'auto'`` to
            select based on file extension.
        tree:
            Name of the TTree to read when applicable.
        uproot_kwargs:
            Keyword arguments forwarded to :meth:`uproot.TTree.arrays`.
        amptools_kwargs:
            Keyword arguments forwarded to the AmpTools-format backend.
            Supports ``pol_in_beam``, ``pol_angle``, ``pol_magnitude``,
            ``pol_magnitude_name``, ``pol_angle_name``, and ``num_entries``.
        """
        path_obj = Path(path)
        backend_name = (
            backend.lower() if backend else cls._default_backend_for_path(path_obj)
        )
        if backend_name == 'auto':
            backend_name = cls._default_backend_for_path(path_obj)

        native_aliases = dict(aliases) if aliases is not None else None

        if backend_name in {'native', 'parquet', 'rust', 'oxyroot'}:
            return _NATIVE_DATASET_OPEN(
                path_obj,
                p4s=p4s,
                aux=aux,
                tree=tree,
                aliases=native_aliases,
            )

        if native_aliases:
            msg = "'aliases' are only supported when backend='native'/'parquet'"
            raise ValueError(msg)

        if backend_name == 'uproot':
            kwargs = dict(uproot_kwargs or {})
            backend_tree = tree or kwargs.pop('tree', None)
            return cls._open_with_uproot(
                path_obj,
                tree=backend_tree,
                p4s=p4s,
                aux=aux,
                uproot_kwargs=kwargs,
            )

        if backend_name == 'amptools':
            kwargs = dict(amptools_kwargs or {})
            backend_tree = tree or kwargs.pop('tree', None)
            dataset = cls._open_amptools_format(
                path_obj,
                tree=backend_tree,
                amptools_kwargs=kwargs,
            )
            return dataset

        msg = (
            f"Unsupported backend '{backend_name}'. "
            "Valid options are 'oxyroot', 'uproot', 'amptools', 'native', or 'auto'."
        )
        raise ValueError(msg)

    @staticmethod
    def _default_backend_for_path(path: Path) -> str:
        return 'oxyroot' if path.suffix.lower() == '.root' else 'native'

    @classmethod
    def _open_with_uproot(
        cls,
        path: Path,
        *,
        tree: str | None,
        p4s: list[str] | None,
        aux: list[str] | None,
        uproot_kwargs: dict[str, Any],
    ) -> _DatasetCore:
        uproot_module = _import_optional_dependency(
            'uproot',
            extra='uproot',
            feature="Dataset.open(... backend='uproot')",
        )
        with uproot_module.open(path) as root_file:
            tree_obj = cls._select_uproot_tree(root_file, tree)
            arrays = tree_obj.arrays(library='np', **uproot_kwargs)

        columns = {name: np.asarray(values) for name, values in arrays.items()}
        selected = cls._prepare_uproot_columns(columns, p4s=p4s, aux=aux)
        return cls.from_numpy(selected)

    @classmethod
    def _open_amptools_format(
        cls,
        path: Path,
        *,
        tree: str | None,
        amptools_kwargs: dict[str, Any],
    ) -> _DatasetCore:
        kwargs = dict(amptools_kwargs)
        pol_in_beam = kwargs.pop('pol_in_beam', False)
        pol_angle = kwargs.pop('pol_angle', None)
        pol_magnitude = kwargs.pop('pol_magnitude', None)
        pol_magnitude_name = kwargs.pop('pol_magnitude_name', 'pol_magnitude')
        pol_angle_name = kwargs.pop('pol_angle_name', 'pol_angle')
        num_entries = kwargs.pop('num_entries', None)
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            msg = f'Unsupported AmpTools options: {unknown}'
            raise TypeError(msg)

        pol_angle_rad = pol_angle * np.pi / 180 if pol_angle is not None else None
        polarisation_requested = pol_in_beam or (
            pol_angle is not None and pol_magnitude is not None
        )
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
    def _select_uproot_tree(
        file: uproot.ReadOnlyDirectory, tree_name: str | None
    ) -> uproot.TTree:
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

        component_columns = [
            f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')
        ]
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
    from laddu.laddu import Dataset as Dataset
else:
    Dataset = cast('type[_DatasetExtensions]', _DatasetCore)
    Dataset.__doc__ = _DatasetExtensions.__doc__

DatasetBase = Dataset
del _name, _attr


def open(
    path: str | Path,
    *,
    p4s: list[str],
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    alias_dict = dict(aliases) if aliases is not None else None
    return Dataset.open(
        path,
        p4s=p4s,
        aux=aux,
        aliases=alias_dict,
    )


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
        feature="Dataset.open(... backend='amptools')",
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


__all__ = ['BinnedDataset', 'Dataset', 'Event', 'open']
