from abc import ABCMeta, abstractmethod

from laddu.amplitudes import Expression, Manager, constant, parameter
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import Zlm
from laddu.data import BinnedDataset, Dataset, open, open_binned
from laddu.likelihoods import NLL, LikelihoodManager, Status
from laddu.utils.variables import Angles, CosTheta, Mass, Phi, PolAngle, Polarization, PolMagnitude
from laddu.utils.vectors import Vector3, Vector4

from . import amplitudes, data, utils

class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status, expression: Expression) -> tuple[Status, Expression, bool]: ...

__version__: str

__all__ = [
    "__version__",
    "Dataset",
    "open",
    "BinnedDataset",
    "open_binned",
    "utils",
    "data",
    "amplitudes",
    "Vector3",
    "Vector4",
    "CosTheta",
    "Phi",
    "Angles",
    "PolMagnitude",
    "PolAngle",
    "Polarization",
    "Mass",
    "Manager",
    "LikelihoodManager",
    "NLL",
    "Expression",
    "Status",
    "Observer",
    "parameter",
    "constant",
    "Scalar",
    "ComplexScalar",
    "PolarComplexScalar",
    "Ylm",
    "Zlm",
    "BreitWigner",
]
