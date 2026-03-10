from collections.abc import Iterator, Mapping, Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from laddu.utils.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
    VariableExpression,
)
from laddu.utils.vectors import Vec4

class Event:
    p4s: Mapping[str, Vec4]
    aux: Mapping[str, float]
    weight: float

    def __init__(
        self,
        p4s: list[Vec4],
        aux: list[float],
        weight: float,
        *,
        p4_names: list[str] | None = None,
        aux_names: list[str] | None = None,
        aliases: dict[str, str | list[str]] | None = None,
    ) -> None: ...
    def get_p4_sum(self, names: list[str]) -> Vec4: ...
    def boost_to_rest_frame_of(self, names: list[str]) -> Event: ...
    def p4(self, name: str) -> Vec4 | None: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> float: ...

class Dataset(Sequence[Event]):
    events: list[Event]
    events_local: list[Event]
    n_events: int
    n_events_local: int
    n_events_weighted: float
    n_events_weighted_local: float
    weights: NDArray[np.float64]
    weights_local: NDArray[np.float64]
    p4_names: list[str]
    aux_names: list[str]

    def __init__(
        self,
        events: list[Event],
        *,
        p4_names: list[str] | None = None,
        aux_names: list[str] | None = None,
        aliases: dict[str, str | list[str]] | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Event]: ...
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
    def iter_local(self) -> Iterator[Event]: ...
    def iter_global(self) -> Iterator[Event]: ...
    def p4_by_name(self, index: int, name: str) -> Vec4: ...
    def aux_by_name(self, index: int, name: str) -> float: ...
    def boost_to_rest_frame_of(self, names: list[str]) -> Dataset: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> NDArray[np.float64]: ...

class BinnedDataset:
    n_bins: int
    range: tuple[float, float]
    edges: NDArray[np.float64]

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...
