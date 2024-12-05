from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from laddu.amplitudes import Manager, constant, parameter
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import Zlm
from laddu.convert import convert_from_amptools, read_root_file
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

from . import amplitudes, convert, data, likelihoods, utils
from .laddu import version

if TYPE_CHECKING:
    from pathlib import Path

__version__ = version()


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status) -> tuple[Status, bool]:
        pass


def open_amptools(
    path: str | Path,
    tree: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> Dataset:
    pol_angle_rad = pol_angle * np.pi / 180 if pol_angle else None
    p4s_list, eps_list, weight_list = read_root_file(
        path, tree, pol_in_beam, pol_angle_rad, pol_magnitude, num_entries
    )
    return Dataset(
        [
            Event(
                [Vector4.from_array(p4) for p4 in p4s],
                [Vector3.from_array(eps_vec) for eps_vec in eps],
                weight,
            )
            for p4s, eps, weight in zip(p4s_list, eps_list, weight_list)
        ]
    )


__all__ = [
    'NLL',
    'Angles',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'CosTheta',
    'Dataset',
    'Event',
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
    'likelihoods',
    'open',
    'open_amptools',
    'parameter',
    'utils',
]
