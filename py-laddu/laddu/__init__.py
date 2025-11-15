from __future__ import annotations

import laddu.laddu as _laddu
from laddu import amplitudes, data, experimental, extensions, mpi, utils
from laddu.amplitudes import (
    AmplitudeOne,
    AmplitudeZero,
    Manager,
    Model,
    amplitude_product,
    amplitude_sum,
    constant,
    parameter,
)
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.phase_space import PhaseSpaceFactor
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import PolPhase, Zlm
from laddu.data import BinnedDataset, Dataset, Event
from laddu.extensions import (
    NLL,
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodID,
    LikelihoodManager,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodTerm,
    LikelihoodZero,
    MCMCObserver,
    MCMCSummary,
    MCMCTerminator,
    MinimizationObserver,
    MinimizationStatus,
    MinimizationSummary,
    MinimizationTerminator,
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
)
from laddu.utils.vectors import Vec3, Vec4

DatasetBase = Dataset

__doc__: str | None = _laddu.__doc__
__version__: str = _laddu.version()
available_parallelism = _laddu.available_parallelism


__all__ = [
    'NLL',
    'AmplitudeOne',
    'AmplitudeZero',
    'Angles',
    'AutocorrelationTerminator',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'DatasetBase',
    'EnsembleStatus',
    'Event',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodID',
    'LikelihoodManager',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodTerm',
    'LikelihoodZero',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'Manager',
    'Mandelstam',
    'Mass',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'Model',
    'PhaseSpaceFactor',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Scalar',
    'StochasticNLL',
    'Swarm',
    'SwarmParticle',
    'Topology',
    'Vec3',
    'Vec4',
    'Walker',
    'Ylm',
    'Zlm',
    '__version__',
    'amplitude_product',
    'amplitude_sum',
    'amplitudes',
    'constant',
    'data',
    'experimental',
    'extensions',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
    'mpi',
    'parameter',
    'utils',
]
