from typing import Literal, overload

from laddu.amplitudes import Amplitude
from laddu.utils.variables import Angles, CosTheta, Phi, Polarization

@overload
def Zlm(
    name: str,
    l: Literal[0],  # noqa: E741
    m: Literal[0],
    r: Literal["+", "plus", "pos", "positive", "-", "minus", "neg", "negative"],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[1],  # noqa: E741
    m: Literal[-1, 0, 1],
    r: Literal["+", "plus", "pos", "positive", "-", "minus", "neg", "negative"],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[2],  # noqa: E741
    m: Literal[-2, -1, 0, 1, 2],
    r: Literal["+", "plus", "pos", "positive", "-", "minus", "neg", "negative"],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[3],  # noqa: E741
    m: Literal[-3, -2, -1, 0, 1, 2, 3],
    r: Literal["+", "plus", "pos", "positive", "-", "minus", "neg", "negative"],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
@overload
def Zlm(
    name: str,
    l: Literal[4],  # noqa: E741
    m: Literal[-4, -3, -2, -1, 0, 1, 2, 3, 4],
    r: Literal["+", "plus", "pos", "positive", "-", "minus", "neg", "negative"],
    angles: Angles,
    polarization: Polarization,
) -> Amplitude: ...
