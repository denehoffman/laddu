from typing import Literal, overload

from laddu.amplitudes import Expression, ParameterLike
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
    *,
    seed: int | None = None,
) -> Expression: ...
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
    *,
    seed: int | None = None,
) -> Expression: ...
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
    *,
    seed: int | None = None,
) -> Expression: ...
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
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1],
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: Literal[0, 1, 2],
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: int,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: Literal[0, 1],
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
@overload
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: int,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
