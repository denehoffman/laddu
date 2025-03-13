from typing import Literal, overload

from laddu.amplitudes import Amplitude, ParameterLike
from laddu.utils.variables import Mass

@overload
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
@overload
def KopfKMatrixF0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
@overload
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
@overload
def KopfKMatrixF2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1],
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: Literal[0, 1],
    mass: Mass,
) -> Amplitude: ...
@overload
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: int,
    mass: Mass,
) -> Amplitude: ...
