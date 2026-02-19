import os
from typing import Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from laddu.amplitudes import (
    Evaluator,
    Expression,
    One,
    ParameterLike,
    TestAmplitude,
    Zero,
    constant,
    expr_product,
    expr_sum,
    parameter,
)
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.kmatrix import (
    KopfKMatrixA0,
    KopfKMatrixA2,
    KopfKMatrixF0,
    KopfKMatrixF2,
    KopfKMatrixPi1,
    KopfKMatrixRho,
)
from laddu.amplitudes.phase_space import PhaseSpaceFactor
from laddu.amplitudes.piecewise import (
    PiecewiseComplexScalar,
    PiecewisePolarComplexScalar,
    PiecewiseScalar,
)
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import PolPhase, Zlm
from laddu.data import BinnedDataset, Dataset, Event
from laddu.experimental import BinnedGuideTerm, Regularizer
from laddu.extensions import (
    NLL,
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    StochasticNLL,
    Swarm,
    SwarmParticle,
    Walker,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)
from laddu.utils.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    Topology,
    VariableExpression,
)
from laddu.utils.vectors import Vec3, Vec4

__all__ = [
    'NLL',
    'Angles',
    'AutocorrelationTerminator',
    'BinnedDataset',
    'BinnedGuideTerm',
    'BreitWigner',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
    'KopfKMatrixA0',
    'KopfKMatrixA2',
    'KopfKMatrixF0',
    'KopfKMatrixF2',
    'KopfKMatrixPi1',
    'KopfKMatrixRho',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'MCMCSummary',
    'Mandelstam',
    'Mass',
    'MinimizationStatus',
    'MinimizationSummary',
    'One',
    'ParameterLike',
    'PhaseSpaceFactor',
    'Phi',
    'PiecewiseComplexScalar',
    'PiecewisePolarComplexScalar',
    'PiecewiseScalar',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Regularizer',
    'Scalar',
    'StochasticNLL',
    'Swarm',
    'SwarmParticle',
    'TestAmplitude',
    'Topology',
    'VariableExpression',
    'Vec3',
    'Vec4',
    'Walker',
    'Ylm',
    'Zero',
    'Zlm',
    'available_parallelism',
    'constant',
    'expr_product',
    'expr_sum',
    'finalize_mpi',
    'from_columns',
    'get_rank',
    'get_size',
    'integrated_autocorrelation_times',
    'is_mpi_available',
    'is_root',
    'likelihood_product',
    'likelihood_sum',
    'parameter',
    'read_parquet',
    'read_root',
    'use_mpi',
    'using_mpi',
    'version',
    'write_parquet',
    'write_root',
]

def version() -> str:
    """Return the version string of the loaded laddu backend."""

def available_parallelism() -> int:
    """Return the number of logical CPU cores available to laddu."""

def use_mpi(*, trigger: bool = True) -> None:
    """Enable the MPI backend if the extension was compiled with MPI support."""

def finalize_mpi() -> None:
    """Finalize and tear down the MPI runtime."""

def using_mpi() -> bool:
    """Return ``True`` if the MPI backend is currently active."""

def is_mpi_available() -> bool:
    """Return ``True`` when the extension was built with MPI support."""

def is_root() -> bool:
    """Return ``True`` when the current MPI rank is the root process."""

def get_rank() -> int:
    """Return the MPI rank of the current process (``0`` when MPI is disabled)."""

def get_size() -> int:
    """Return the total number of MPI processes (``1`` when MPI is disabled)."""

def read_parquet(
    path: str | os.PathLike[str],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    """Load a dataset from a Parquet file using the loaded backend."""

def from_columns(
    columns: Mapping[
        str, Sequence[float] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
    ],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    """Build a dataset from in-memory columnar arrays."""

def read_root(
    path: str | os.PathLike[str],
    *,
    tree: str | None = None,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
) -> Dataset:
    """Load a dataset from a ROOT file using the loaded backend."""

def write_parquet(
    dataset: Dataset,
    path: str | os.PathLike[str],
    *,
    chunk_size: int | None = None,
    precision: Literal['f64', 'f32'] = 'f64',
) -> None:
    """Write a dataset to a Parquet file using the loaded backend."""

def write_root(
    dataset: Dataset,
    path: str | os.PathLike[str],
    *,
    tree: str | None = None,
    chunk_size: int | None = None,
    precision: Literal['f64', 'f32'] = 'f64',
) -> None:
    """Write a dataset to a ROOT file using the loaded backend."""
