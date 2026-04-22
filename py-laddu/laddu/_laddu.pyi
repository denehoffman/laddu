import os
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitudes import (
    CompiledExpression,
    Evaluator,
    Expression,
    One,
    Parameter,
    TestAmplitude,
    Zero,
    expr_product,
    expr_sum,
    parameter,
)
from laddu.amplitudes.breit_wigner import BreitWigner, BreitWignerNonRelativistic
from laddu.amplitudes.common import (
    ComplexScalar,
    PolarComplexScalar,
    Scalar,
    VariableScalar,
)
from laddu.amplitudes.flatte import Flatte
from laddu.amplitudes.kmatrix import (
    KopfKMatrixA0,
    KopfKMatrixA0Channel,
    KopfKMatrixA2,
    KopfKMatrixA2Channel,
    KopfKMatrixF0,
    KopfKMatrixF0Channel,
    KopfKMatrixF2,
    KopfKMatrixF2Channel,
    KopfKMatrixPi1,
    KopfKMatrixPi1Channel,
    KopfKMatrixRho,
    KopfKMatrixRhoChannel,
)
from laddu.amplitudes.lookup_table import (
    LookupTable,
    LookupTableComplex,
    LookupTablePolar,
    LookupTableScalar,
)
from laddu.amplitudes.phase_space import PhaseSpaceFactor
from laddu.amplitudes.spin_factors import (
    BlattWeisskopf,
    ClebschGordan,
    PhotonSDME,
    Wigner3j,
    WignerD,
)
from laddu.amplitudes.voigt import Voigt
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import PolPhase, Zlm
from laddu.data import BinnedDataset, Dataset, Event
from laddu.experimental import BinnedGuideTerm, Regularizer
from laddu.extensions import (
    NLL,
    ControlFlow,
    EnsembleStatus,
    GradientFreeStatus,
    GradientStatus,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    StochasticNLL,
    SwarmStatus,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)
from laddu.utils.angular_momentum import allowed_projections, helicity_combinations
from laddu.utils.variables import (
    Angles,
    CosTheta,
    Decay,
    Mandelstam,
    Mass,
    Particle,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    Reaction,
    VariableExpression,
)
from laddu.utils.vectors import Vec3, Vec4

__all__ = [
    'NLL',
    'Angles',
    'BinnedDataset',
    'BinnedGuideTerm',
    'BlattWeisskopf',
    'BreitWigner',
    'BreitWignerNonRelativistic',
    'ClebschGordan',
    'CompiledExpression',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'Decay',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
    'Flatte',
    'GradientFreeStatus',
    'GradientStatus',
    'KopfKMatrixA0',
    'KopfKMatrixA0Channel',
    'KopfKMatrixA2',
    'KopfKMatrixA2Channel',
    'KopfKMatrixF0',
    'KopfKMatrixF0Channel',
    'KopfKMatrixF2',
    'KopfKMatrixF2Channel',
    'KopfKMatrixPi1',
    'KopfKMatrixPi1Channel',
    'KopfKMatrixRho',
    'KopfKMatrixRhoChannel',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'LookupTable',
    'LookupTableComplex',
    'LookupTablePolar',
    'LookupTableScalar',
    'MCMCSummary',
    'Mandelstam',
    'Mass',
    'MinimizationStatus',
    'MinimizationSummary',
    'One',
    'Parameter',
    'ParquetChunkIter',
    'Particle',
    'PhaseSpaceFactor',
    'Phi',
    'PhotonSDME',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Reaction',
    'Regularizer',
    'Scalar',
    'StochasticNLL',
    'SwarmStatus',
    'TestAmplitude',
    'VariableExpression',
    'VariableScalar',
    'Vec3',
    'Vec4',
    'Voigt',
    'Wigner3j',
    'WignerD',
    'Ylm',
    'Zero',
    'Zlm',
    'allowed_projections',
    'available_parallelism',
    'expr_product',
    'expr_sum',
    'finalize_mpi',
    'from_columns',
    'get_rank',
    'get_size',
    'get_threads',
    'helicity_combinations',
    'integrated_autocorrelation_times',
    'is_mpi_available',
    'is_root',
    'likelihood_product',
    'likelihood_sum',
    'parameter',
    'read_parquet',
    'read_parquet_chunked',
    'read_root',
    'set_threads',
    'use_mpi',
    'using_mpi',
    'version',
    'write_parquet',
    'write_root',
]

class ParquetChunkIter:
    def __iter__(self) -> ParquetChunkIter: ...
    def __next__(self) -> Dataset: ...

def version() -> str:
    """Return the version string of the loaded laddu backend."""

def available_parallelism() -> int:
    """Return the number of logical CPU cores available to laddu."""

def get_threads() -> int:
    """Return the global default thread count, or ``0`` for the ambient default."""

def set_threads(n_threads: int | None) -> None:
    """Set the global default thread count for omitted or zero-valued thread arguments."""

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

def read_parquet_chunked(
    path: str | os.PathLike[str],
    *,
    p4s: list[str] | None = None,
    aux: list[str] | None = None,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
    chunk_size: int | None = None,
) -> ParquetChunkIter:
    """Load a dataset from a Parquet file in chunks using the loaded backend."""

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
