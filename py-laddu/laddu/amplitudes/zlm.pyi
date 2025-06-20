from typing import Literal, overload

from laddu.amplitudes import Amplitude
from laddu.utils.variables import Angles, Polarization

@overload
def Zlm(
    name: str,
    l: Literal[0],
    m: Literal[0],
    r: Literal['+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[1],
    m: Literal[-1, 0, 1],
    r: Literal['+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[2],
    m: Literal[-2, -1, 0, 1, 2],
    r: Literal['+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[3],
    m: Literal[-3, -2, -1, 0, 1, 2, 3],
    r: Literal['+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[4],
    m: Literal[-4, -3, -2, -1, 0, 1, 2, 3, 4],
    r: Literal['+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: int,
    m: int,
    r: str,
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
def PolPhase(
    name: str,
    polarization: Polarization,
) -> Amplitude: ...
