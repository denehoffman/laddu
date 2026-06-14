import os
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitude import (
    CompiledExpression,
    Evaluator,
    Expression,
    InitialValue,
    One,
    Parameter,
    ParameterMap,
    TestAmplitude,
    Zero,
    expr_product,
    expr_sum,
    parameter,
)
from laddu.amplitudes.angular import (
    BlattWeisskopf,
    PhotonSDME,
    PolPhase,
    WignerD,
    Ylm,
    Zlm,
)
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
from laddu.amplitudes.lookup import (
    LookupTable,
    LookupTableComplex,
    LookupTablePolar,
    LookupTableScalar,
)
from laddu.amplitudes.resonance import (
    BreitWigner,
    BreitWignerNonRelativistic,
    Flatte,
    PhaseSpaceFactor,
    Voigt,
)
from laddu.amplitudes.scalar import (
    ComplexScalar,
    PolarComplexScalar,
    Scalar,
    VariableScalar,
)
from laddu.data import BinnedDataset, Dataset, Event
from laddu.experimental import BinnedGuideTerm, Regularizer
from laddu.generation import (
    CallbackSink,
    DatasetSink,
    DecayParticlePlan,
    DecayPlan,
    Envelope,
    EnvelopeStats,
    EnvelopeViolationPolicy,
    EventGenerator,
    GeneratedEvent,
    GeneratedSink,
    GenerationMode,
    GenerationOptions,
    GenerationOutput,
    GenerationPlan,
    GenerationResult,
    GenerationStats,
    InitialParticlePlan,
    MassSampler,
    MomentumSource,
    ParquetSink,
    PlannedMass,
    ProductionPlan,
    RootSink,
    VertexGenerator,
)
from laddu.likelihood import (
    NLL,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    StochasticNLL,
    likelihood_product,
    likelihood_sum,
)
from laddu.math import Histogram, clebsch_gordan
from laddu.optimize import (
    ControlFlow,
    EnsembleStatus,
    GradientFreeStatus,
    GradientStatus,
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    SwarmStatus,
    integrated_autocorrelation_times,
)
from laddu.quantum import (
    AllowedPartialWave,
    Charge,
    ExternalId,
    Isospin,
    J,
    L,
    M,
    Parity,
    PartialWave,
    ParticleProperties,
    Reflectivity,
    RuleSet,
    S,
    SelectionRules,
    Statistics,
    allowed_partial_waves,
    allowed_projections,
    coupled_spins,
)
from laddu.reaction import (
    Axes,
    Axis,
    Channel,
    Frame,
    Particle,
    ParticleSource,
    TwoBodyCoupling,
    Vertex,
)
from laddu.samplers import (
    energy,
    histogram_energy,
    histogram_mass,
    mass_from_properties,
    rest,
    t_exponential,
    t_histogram,
    uniform_energy,
    uniform_mass,
)
from laddu.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    VariableExpression,
)
from laddu.vectors import Vec3, Vec4

__all__ = [
    'NLL',
    'AllowedPartialWave',
    'Angles',
    'Axes',
    'Axis',
    'BinnedDataset',
    'BinnedGuideTerm',
    'BlattWeisskopf',
    'BreitWigner',
    'BreitWignerNonRelativistic',
    'CallbackSink',
    'Channel',
    'Charge',
    'CompiledExpression',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'DatasetSink',
    'DecayParticlePlan',
    'DecayPlan',
    'EnsembleStatus',
    'Envelope',
    'EnvelopeStats',
    'EnvelopeViolationPolicy',
    'Evaluator',
    'Event',
    'EventGenerator',
    'Expression',
    'ExternalId',
    'Flatte',
    'Frame',
    'GeneratedEvent',
    'GeneratedSink',
    'GenerationMode',
    'GenerationOptions',
    'GenerationOutput',
    'GenerationPlan',
    'GenerationResult',
    'GenerationStats',
    'GradientFreeStatus',
    'GradientStatus',
    'Histogram',
    'InitialParticlePlan',
    'InitialValue',
    'Isospin',
    'J',
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
    'L',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'LookupTable',
    'LookupTableComplex',
    'LookupTablePolar',
    'LookupTableScalar',
    'M',
    'MCMCSummary',
    'Mandelstam',
    'Mass',
    'MassSampler',
    'MinimizationStatus',
    'MinimizationSummary',
    'MomentumSource',
    'One',
    'Parameter',
    'ParameterMap',
    'Parity',
    'ParquetBatchWriter',
    'ParquetChunkIter',
    'ParquetSink',
    'PartialWave',
    'Particle',
    'ParticleProperties',
    'ParticleSource',
    'PhaseSpaceFactor',
    'Phi',
    'PhotonSDME',
    'PlannedMass',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'ProductionPlan',
    'Reflectivity',
    'Regularizer',
    'RootSink',
    'RuleSet',
    'S',
    'Scalar',
    'SelectionRules',
    'Statistics',
    'StochasticNLL',
    'SwarmStatus',
    'TestAmplitude',
    'TwoBodyCoupling',
    'VariableExpression',
    'VariableScalar',
    'Vec3',
    'Vec4',
    'Vertex',
    'VertexGenerator',
    'Voigt',
    'WignerD',
    'Ylm',
    'Zero',
    'Zlm',
    'allowed_partial_waves',
    'allowed_projections',
    'available_parallelism',
    'clebsch_gordan',
    'coupled_spins',
    'energy',
    'expr_product',
    'expr_sum',
    'finalize_mpi',
    'from_columns',
    'get_rank',
    'get_size',
    'get_threads',
    'histogram_energy',
    'histogram_mass',
    'integrated_autocorrelation_times',
    'is_mpi_available',
    'is_root',
    'likelihood_product',
    'likelihood_sum',
    'mass_from_properties',
    'open_parquet_writer',
    'parameter',
    'read_parquet',
    'read_parquet_chunked',
    'read_root',
    'rest',
    'set_threads',
    't_exponential',
    't_histogram',
    'uniform_energy',
    'uniform_mass',
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

class ParquetBatchWriter:
    def write(self, dataset: Dataset) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> ParquetBatchWriter: ...  # noqa: PYI034
    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> bool: ...

def open_parquet_writer(
    path: str | os.PathLike[str],
    *,
    chunk_size: int | None = None,
    precision: Literal['f64', 'f32'] = 'f64',
) -> ParquetBatchWriter:
    """Open a streaming Parquet writer for compatible Dataset batches."""

def write_root(
    dataset: Dataset,
    path: str | os.PathLike[str],
    *,
    tree: str | None = None,
    chunk_size: int | None = None,
    precision: Literal['f64', 'f32'] = 'f64',
) -> None:
    """Write a dataset to a ROOT file using the loaded backend."""
