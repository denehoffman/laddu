from typing import Literal, overload

from laddu.amplitudes import Amplitude
from laddu.utils.variables import Angles

@overload
def Ylm(name: str, l: Literal[0], m: Literal[0], angles: Angles) -> Amplitude: ...
@overload
def Ylm(name: str, l: Literal[1], m: Literal[-1, 0, 1], angles: Angles) -> Amplitude: ...
@overload
def Ylm(
    name: str, l: Literal[2], m: Literal[-2, -1, 0, 1, 2], angles: Angles
) -> Amplitude: ...
@overload
def Ylm(
    name: str,
    l: Literal[3],
    m: Literal[-3, -2, -1, 0, 1, 2, 3],
    angles: Angles,
) -> Amplitude: ...
@overload
def Ylm(
    name: str,
    l: Literal[4],
    m: Literal[-4, -3, -2, -1, 0, 1, 2, 3, 4],
    angles: Angles,
) -> Amplitude: ...
@overload
def Ylm(name: str, l: int, m: int, angles: Angles) -> Amplitude: ...
