from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.data import Dataset, Event

class Mass:
    def __init__(self, constituents: list[int]): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...

class CosTheta:
    def __init__(
        self,
        beam: int,
        recoil: list[int],
        daughter: list[int],
        resonance: list[int],
        frame: Literal[
            'Helicity',
            'HX',
            'HEL',
            'GottfriedJackson',
            'Gottfried Jackson',
            'GJ',
            'Gottfried-Jackson',
        ] = 'Helicity',
    ): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...

class Phi:
    def __init__(
        self,
        beam: int,
        recoil: list[int],
        daughter: list[int],
        resonance: list[int],
        frame: Literal[
            'Helicity',
            'HX',
            'HEL',
            'GottfriedJackson',
            'Gottfried Jackson',
            'GJ',
            'Gottfried-Jackson',
        ] = 'Helicity',
    ): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...

class Angles:
    costheta: CosTheta
    phi: Phi
    def __init__(
        self,
        beam: int,
        recoil: list[int],
        daughter: list[int],
        resonance: list[int],
        frame: Literal[
            'Helicity',
            'HX',
            'HEL',
            'GottfriedJackson',
            'Gottfried Jackson',
            'GJ',
            'Gottfried-Jackson',
        ] = 'Helicity',
    ) -> None: ...

class PolAngle:
    def __init__(
        self,
        beam: int,
        recoil: list[int],
    ): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...

class PolMagnitude:
    def __init__(
        self,
        beam: int,
    ): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...

class Polarization:
    pol_magnitude: PolMagnitude
    pol_angle: PolAngle
    def __init__(self, beam: int, recoil: list[int]) -> None: ...

class Mandelstam:
    def __init__(
        self,
        p1: list[int],
        p2: list[int],
        p3: list[int],
        p4: list[int],
        channel: Literal['s', 't', 'u'],
    ): ...
    def value(self, event: Event) -> float: ...
    def value_on(self, dataset: Dataset) -> npt.NDArray[np.float64]: ...
