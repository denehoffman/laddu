from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from laddu.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
    VariableExpression,
)
from laddu.vectors import Vec4

class Event:
    p4s: dict[str, Vec4]
    aux: dict[str, float]
    weight: float

    def __init__(
        self,
        p4s: Sequence[Vec4],
        aux: Sequence[float],
        weight: float,
        *,
        p4_names: Sequence[str] | None = None,
        aux_names: Sequence[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> None: ...
    def get_p4_sum(self, names: Sequence[str]) -> Vec4: ...
    def boost_to_rest_frame_of(self, names: Sequence[str]) -> Event: ...
    def p4(self, name: str) -> Vec4: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> float: ...

class Dataset:
    events_global: list[Event]
    events_local: list[Event]
    n_events: int
    n_events_global: int
    n_events_local: int
    n_events_weighted: float
    n_events_weighted_global: float
    n_events_weighted_local: float
    weights: NDArray[np.float64]
    weights_global: NDArray[np.float64]
    weights_local: NDArray[np.float64]
    p4_names: list[str]
    aux_names: list[str]

    def __init__(
        self,
        events: Sequence[Event],
        *,
        p4_names: Sequence[str] | None = None,
        aux_names: Sequence[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> None: ...
    @staticmethod
    def from_events_local(
        events: Sequence[Event],
        *,
        p4_names: Sequence[str] | None = None,
        aux_names: Sequence[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> Dataset: ...
    @staticmethod
    def from_events_global(
        events: Sequence[Event],
        *,
        p4_names: Sequence[str] | None = None,
        aux_names: Sequence[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> Dataset: ...
    @staticmethod
    def empty_local(
        *,
        p4_names: Sequence[str],
        aux_names: Sequence[str] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
    ) -> Dataset: ...
    def __len__(self) -> int: ...
    def __add__(self, other: Dataset | int) -> Dataset: ...
    def __radd__(self, other: Dataset | int) -> Dataset: ...
    @overload
    def __getitem__(self, index: slice) -> Dataset: ...
    @overload
    def __getitem__(self, index: int) -> Event: ...
    @overload
    def __getitem__(
        self, index: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> NDArray[np.float64]: ...
    def __getitem__(
        self,
        index: int | Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    ) -> Event | NDArray[np.float64]: ...
    def bin_by(
        self,
        variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
        bins: int,
        range: tuple[float, float],
    ) -> BinnedDataset: ...
    def filter(self, expression: VariableExpression) -> Dataset: ...
    def bootstrap(self, seed: int) -> Dataset: ...
    def event_global(self, index: int) -> Event: ...
    def push_event_local(
        self,
        *,
        p4: Mapping[str, Vec4],
        aux: Mapping[str, float] | None = None,
        weight: float = 1.0,
    ) -> None: ...
    def push_event_global(
        self,
        *,
        p4: Mapping[str, Vec4],
        aux: Mapping[str, float] | None = None,
        weight: float = 1.0,
    ) -> None: ...
    def p4_by_name(self, index: int, name: str) -> Vec4: ...
    def aux_by_name(self, index: int, name: str) -> float: ...
    def boost_to_rest_frame_of(self, names: Sequence[str]) -> Dataset: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> NDArray[np.float64]: ...
    def to_arrow(self, *, precision: Literal['f64', 'f32'] = 'f64') -> Any: ...

class BinnedDataset:
    n_bins: int
    range: tuple[float, float]
    edges: NDArray[np.float64]

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...
