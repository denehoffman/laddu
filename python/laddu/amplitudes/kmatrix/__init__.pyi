from typing import Literal

from laddu.amplitudes import Amplitude, ParameterLike
from laddu.utils.variables import Mass

def KopfKMatrixF0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2, 3, 4],
    mass: Mass,
) -> Amplitude: ...
def KopfKMatrixF2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2, 3],
    mass: Mass,
) -> Amplitude: ...
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1],
    mass: Mass,
) -> Amplitude: ...
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
) -> Amplitude: ...
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
) -> Amplitude: ...
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: Literal[0, 1],
    mass: Mass,
) -> Amplitude: ...
