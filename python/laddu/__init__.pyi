from abc import ABCMeta, abstractmethod
from pathlib import Path

from laddu.amplitudes import Expression, Manager, constant, parameter
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import Zlm
from laddu.convert import convert_from_amptools
from laddu.data import BinnedDataset, Dataset, Event, open
from laddu.likelihoods import NLL, LikelihoodManager, Status
from laddu.utils.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
)
from laddu.utils.vectors import Vector3, Vector4

from . import amplitudes, convert, data, utils

__version__: str

class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status) -> tuple[Status, bool]: ...

def open_amptools(
    path: str | Path,
    tree: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> Dataset: ...

__all__ = [
    'NLL',
    'Angles',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'CosTheta',
    'Dataset',
    'Event',
    'Expression',
    'LikelihoodManager',
    'Manager',
    'Mandelstam',
    'Mass',
    'Observer',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'PolarComplexScalar',
    'Polarization',
    'Scalar',
    'Status',
    'Vector3',
    'Vector4',
    'Ylm',
    'Zlm',
    '__version__',
    'amplitudes',
    'constant',
    'convert',
    'convert_from_amptools',
    'data',
    'open',
    'open_amptools',
    'parameter',
    'utils',
]
