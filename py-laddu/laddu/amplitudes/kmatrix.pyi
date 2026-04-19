from enum import Enum

from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import Mass

class KopfKMatrixF0Channel(Enum):
    PiPi: KopfKMatrixF0Channel
    FourPi: KopfKMatrixF0Channel
    KKbar: KopfKMatrixF0Channel
    EtaEta: KopfKMatrixF0Channel
    EtaEtaPrime: KopfKMatrixF0Channel

class KopfKMatrixF2Channel(Enum):
    PiPi: KopfKMatrixF2Channel
    FourPi: KopfKMatrixF2Channel
    KKbar: KopfKMatrixF2Channel
    EtaEta: KopfKMatrixF2Channel

class KopfKMatrixA0Channel(Enum):
    PiEta: KopfKMatrixA0Channel
    KKbar: KopfKMatrixA0Channel

class KopfKMatrixA2Channel(Enum):
    PiEta: KopfKMatrixA2Channel
    KKbar: KopfKMatrixA2Channel
    PiEtaPrime: KopfKMatrixA2Channel

class KopfKMatrixRhoChannel(Enum):
    PiPi: KopfKMatrixRhoChannel
    FourPi: KopfKMatrixRhoChannel
    KKbar: KopfKMatrixRhoChannel

class KopfKMatrixPi1Channel(Enum):
    PiEta: KopfKMatrixPi1Channel
    PiEtaPrime: KopfKMatrixPi1Channel

def KopfKMatrixF0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: KopfKMatrixF0Channel,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
def KopfKMatrixF2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: KopfKMatrixF2Channel,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
def KopfKMatrixA0(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: KopfKMatrixA0Channel,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
def KopfKMatrixA2(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: KopfKMatrixA2Channel,
    mass: Mass,
    *,
    seed: int | None = None,
) -> Expression: ...
def KopfKMatrixRho(
    name: str,
    couplings: tuple[
        tuple[ParameterLike, ParameterLike],
        tuple[ParameterLike, ParameterLike],
    ],
    channel: KopfKMatrixRhoChannel,
    mass: Mass,
) -> Expression: ...
def KopfKMatrixPi1(
    name: str,
    couplings: tuple[tuple[ParameterLike, ParameterLike],],
    channel: KopfKMatrixPi1Channel,
    mass: Mass,
) -> Expression: ...
