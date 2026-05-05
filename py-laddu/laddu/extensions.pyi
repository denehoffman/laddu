from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal, TypeAlias, overload

import ganesh
import numpy as np
import numpy.typing as npt

from laddu.amplitude import CompiledExpression, Evaluator, Expression, ParameterMap
from laddu.data import Dataset

def likelihood_sum(
    likelihoods: Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def likelihood_product(
    likelihoods: Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def LikelihoodOne() -> LikelihoodExpression: ...
def LikelihoodZero() -> LikelihoodExpression: ...
def integrated_autocorrelation_times(
    samples: Sequence[Sequence[Sequence[float]]] | npt.ArrayLike,
    *,
    c: float | None = None,
) -> npt.NDArray[np.float64]: ...

class LikelihoodExpression:
    parameters: ParameterMap
    n_free: int
    n_fixed: int
    n_parameters: int

    def fix_parameter(self, name: str, value: float) -> None: ...
    def free_parameter(self, name: str) -> None: ...
    def rename_parameter(self, old: str, new: str) -> None: ...
    def rename_parameters(self, mapping: Mapping[str, str]) -> None: ...
    def __add__(self, other: LikelihoodExpression | int) -> LikelihoodExpression: ...
    def __radd__(self, other: LikelihoodExpression | int) -> LikelihoodExpression: ...
    def __mul__(self, other: LikelihoodExpression) -> LikelihoodExpression: ...
    def __rmul__(self, other: LikelihoodExpression) -> LikelihoodExpression: ...
    def evaluate(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: MinimizerInit,
        *,
        method: Literal[
            'lbfgsb',
            'adam',
            'conjugate-gradient',
            'trust-region',
            'nelder-mead',
            'cma-es',
            'differential-evolution',
            'pso',
        ] = 'lbfgsb',
        config: MinimizerConfig | None = None,
        options: MinimizerOptions | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        threads: int = 0,
    ) -> ganesh.MinimizationSummary: ...
    def mcmc(
        self,
        p0: SamplerInit,
        *,
        method: Literal['aies', 'ess'] = 'aies',
        config: SamplerConfig | None = None,
        options: SamplerOptions | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator | Sequence[MCMCTerminator] | None = None,
        threads: int = 0,
    ) -> ganesh.MCMCSummary: ...

class ControlFlow(Enum):
    Continue = 0
    Break = 1

GradientStatus: TypeAlias = ganesh.GradientStatus
GradientFreeStatus: TypeAlias = ganesh.GradientFreeStatus
SwarmStatus: TypeAlias = ganesh.SwarmStatus
EnsembleStatus: TypeAlias = ganesh.EnsembleStatus

class MinimizationObserver(metaclass=ABCMeta):
    @abstractmethod
    def observe(self, step: int, status: MinimizationStatus) -> None: ...

class MinimizationTerminator(metaclass=ABCMeta):
    @abstractmethod
    def check_for_termination(
        self, step: int, status: MinimizationStatus
    ) -> ControlFlow: ...

class MCMCObserver(metaclass=ABCMeta):
    @abstractmethod
    def observe(self, step: int, status: EnsembleStatus) -> None: ...

class MCMCTerminator(metaclass=ABCMeta):
    @abstractmethod
    def check_for_termination(self, step: int, status: EnsembleStatus) -> ControlFlow: ...

MinimizationStatus: TypeAlias = GradientStatus | GradientFreeStatus | SwarmStatus
MinimizationSummary: TypeAlias = ganesh.MinimizationSummary
MCMCSummary: TypeAlias = ganesh.MCMCSummary

MinimizerConfig: TypeAlias = (
    ganesh.LBFGSBConfig
    | ganesh.AdamConfig
    | ganesh.ConjugateGradientConfig
    | ganesh.TrustRegionConfig
    | ganesh.NelderMeadConfig
    | ganesh.CMAESConfig
    | ganesh.DifferentialEvolutionConfig
    | ganesh.PSOConfig
)
MinimizerOptions: TypeAlias = (
    ganesh.LBFGSBOptions
    | ganesh.AdamOptions
    | ganesh.ConjugateGradientOptions
    | ganesh.TrustRegionOptions
    | ganesh.NelderMeadOptions
    | ganesh.CMAESOptions
    | ganesh.DifferentialEvolutionOptions
    | ganesh.PSOOptions
)
SamplerConfig: TypeAlias = ganesh.AIESConfig | ganesh.ESSConfig
SamplerOptions: TypeAlias = ganesh.AIESOptions | ganesh.ESSOptions
MinimizerInit: TypeAlias = (
    Sequence[float]
    | npt.ArrayLike
    | ganesh.NelderMeadInit
    | ganesh.CMAESInit
    | ganesh.DifferentialEvolutionInit
    | ganesh.PSOInit
)
SamplerInit: TypeAlias = (
    Sequence[Sequence[float]] | npt.ArrayLike | ganesh.AIESInit | ganesh.ESSInit
)

class StochasticNLL:
    nll: NLL
    expression: Expression
    compiled_expression: CompiledExpression

    def minimize(
        self,
        p0: MinimizerInit,
        *,
        method: Literal[
            'lbfgsb',
            'adam',
            'conjugate-gradient',
            'trust-region',
            'nelder-mead',
            'cma-es',
            'differential-evolution',
            'pso',
        ] = 'lbfgsb',
        config: MinimizerConfig | None = None,
        options: MinimizerOptions | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        threads: int = 0,
    ) -> ganesh.MinimizationSummary: ...
    def mcmc(
        self,
        p0: SamplerInit,
        *,
        method: Literal['aies', 'ess'] = 'aies',
        config: SamplerConfig | None = None,
        options: SamplerOptions | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator | Sequence[MCMCTerminator] | None = None,
        threads: int = 0,
    ) -> ganesh.MCMCSummary: ...

class NLL:
    parameters: ParameterMap
    n_free: int
    n_fixed: int
    n_parameters: int
    data: Dataset
    accmc: Dataset
    data_evaluator: Evaluator
    accmc_evaluator: Evaluator
    expression: Expression
    compiled_expression: CompiledExpression

    def __init__(
        self,
        expression: Expression,
        ds_data: Dataset,
        ds_accmc: Dataset,
        *,
        n_mc: float | None = None,
    ) -> None: ...
    def to_expression(self) -> LikelihoodExpression: ...
    def to_stochastic(
        self, batch_size: int, *, seed: int | None = None
    ) -> StochasticNLL: ...
    def fix_parameter(self, name: str, value: float) -> None: ...
    def free_parameter(self, name: str) -> None: ...
    def rename_parameter(self, old: str, new: str) -> None: ...
    def rename_parameters(self, mapping: Mapping[str, str]) -> None: ...
    def activate(self, name: str | Sequence[str], *, strict: bool = True) -> None: ...
    def activate_all(self) -> None: ...
    def deactivate(self, name: str | Sequence[str], *, strict: bool = True) -> None: ...
    def deactivate_all(self) -> None: ...
    def isolate(self, name: str | Sequence[str], *, strict: bool = True) -> None: ...
    def evaluate(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def project_weights(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: None = None,
        subsets: None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def project_weights(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: str | Sequence[str],
        subsets: None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def project_weights(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: None = None,
        subsets: Sequence[Sequence[str] | None],
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_weights(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: str | Sequence[str] | None = None,
        subsets: Sequence[Sequence[str] | None] | None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def project_weights_and_gradients(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: None = None,
        subsets: None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def project_weights_and_gradients(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: str | Sequence[str],
        subsets: None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def project_weights_and_gradients(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: None = None,
        subsets: Sequence[Sequence[str] | None],
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def project_weights_and_gradients(
        self,
        parameters: Sequence[float] | npt.ArrayLike,
        *,
        subset: str | Sequence[str] | None = None,
        subsets: Sequence[Sequence[str] | None] | None = None,
        strict: bool = False,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def minimize(
        self,
        p0: MinimizerInit,
        *,
        method: Literal[
            'lbfgsb',
            'adam',
            'conjugate-gradient',
            'trust-region',
            'nelder-mead',
            'cma-es',
            'differential-evolution',
            'pso',
        ] = 'lbfgsb',
        config: MinimizerConfig | None = None,
        options: MinimizerOptions | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        threads: int = 0,
    ) -> ganesh.MinimizationSummary: ...
    def mcmc(
        self,
        p0: SamplerInit,
        *,
        method: Literal['aies', 'ess'] = 'aies',
        config: SamplerConfig | None = None,
        options: SamplerOptions | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator | Sequence[MCMCTerminator] | None = None,
        threads: int = 0,
    ) -> ganesh.MCMCSummary: ...

def LikelihoodScalar(name: str) -> LikelihoodExpression: ...

__all__ = [
    'NLL',
    'ControlFlow',
    'EnsembleStatus',
    'GradientFreeStatus',
    'GradientStatus',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'StochasticNLL',
    'SwarmStatus',
    'likelihood_product',
    'likelihood_sum',
]
