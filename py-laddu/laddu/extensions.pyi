from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitudes import Evaluator, Expression
from laddu.data import Dataset

def likelihood_sum(
    likelihoods: Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def likelihood_product(
    likelihoods: Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def LikelihoodOne() -> LikelihoodExpression: ...
def LikelihoodZero() -> LikelihoodExpression: ...

class LikelihoodExpression:
    parameters: list[str]

    def load(self) -> LikelihoodEvaluator: ...
    def __add__(self, other: LikelihoodExpression | int) -> LikelihoodExpression: ...
    def __radd__(self, other: LikelihoodExpression | int) -> LikelihoodExpression: ...
    def __mul__(self, other: LikelihoodExpression) -> LikelihoodExpression: ...
    def __rmul__(self, other: LikelihoodExpression) -> LikelihoodExpression: ...

class MinimizationStatus:
    x: npt.NDArray[np.float64]
    fx: float
    message: str
    err: npt.NDArray[np.float64] | None
    n_f_evals: int
    n_g_evals: int
    cov: npt.NDArray[np.float64] | None
    hess: npt.NDArray[np.float64] | None
    converged: bool
    swarm: Swarm | None

class MinimizationSummary:
    bounds: list[tuple[float, float]] | None
    parameter_names: list[str] | None
    message: str
    x0: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    fx: float
    cost_evals: int
    gradient_evals: int
    converged: bool
    covariance: npt.NDArray[np.float64]

    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...

class MCMCSummary:
    bounds: list[tuple[float, float]] | None
    parameter_names: list[str] | None
    message: str
    cost_evals: int
    gradient_evals: int
    converged: bool
    dimension: tuple[int, int, int]

    def get_chain(
        self, *, burn: int | None = None, thin: int | None = None
    ) -> npt.NDArray[np.float64]: ...
    def get_flat_chain(
        self, *, burn: int | None = None, thin: int | None = None
    ) -> npt.NDArray[np.float64]: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...

class EnsembleStatus:
    message: str
    n_f_evals: int
    n_g_evals: int
    walkers: list[Walker]
    dimension: tuple[int, int, int]

    def get_chain(
        self, *, burn: int | None = None, thin: int | None = None
    ) -> npt.NDArray[np.float64]: ...
    def get_flat_chain(
        self, *, burn: int | None = None, thin: int | None = None
    ) -> npt.NDArray[np.float64]: ...

class Swarm:
    particles: list[SwarmParticle]

class SwarmParticle:
    x: npt.NDArray[np.float64]
    fx: float
    x_best: npt.NDArray[np.float64]
    fx_best: float
    velocity: npt.NDArray[np.float64]

class Walker:
    dimension: tuple[int, int]

    def get_latest(self) -> tuple[npt.NDArray[np.float64], float]: ...

class ControlFlow(Enum):
    Continue = 0
    Break = 1

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

class LikelihoodEvaluator:
    parameters: list[str]

    def evaluate(
        self,
        parameters: list[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: list[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: list[float] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder-mead', 'adam', 'pso'] = 'lbfgsb',
        settings: dict | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MinimizationSummary: ...
    def mcmc(
        self,
        p0: list[list[float]] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['aies', 'ess'] = 'aies',
        settings: dict | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator
        | AutocorrelationTerminator
        | Sequence[MCMCTerminator | AutocorrelationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MCMCSummary: ...

class StochasticNLL:
    nll: NLL

    def minimize(
        self,
        p0: list[float] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder-mead', 'adam', 'pso'] = 'lbfgsb',
        settings: dict | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MinimizationSummary: ...
    def mcmc(
        self,
        p0: list[list[float]] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['aies', 'ess'] = 'aies',
        settings: dict | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator
        | AutocorrelationTerminator
        | Sequence[MCMCTerminator | AutocorrelationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MCMCSummary: ...

class NLL:
    parameters: list[str]
    free_parameters: list[str]
    fixed_parameters: list[str]
    n_free: int
    n_fixed: int
    n_parameters: int
    data: Dataset
    accmc: Dataset

    def __init__(
        self,
        expression: Expression,
        ds_data: Dataset,
        ds_accmc: Dataset,
    ) -> None: ...
    def to_expression(self) -> LikelihoodExpression: ...
    def to_stochastic(
        self, batch_size: int, *, seed: int | None = None
    ) -> StochasticNLL: ...
    def fix(self, name: str, value: float) -> NLL: ...
    def free(self, name: str) -> NLL: ...
    def rename_parameter(self, old: str, new: str) -> NLL: ...
    def rename_parameters(self, mapping: dict[str, str]) -> NLL: ...
    def activate(self, name: str | list[str], *, strict: bool = True) -> None: ...
    def activate_all(self) -> None: ...
    def deactivate(self, name: str | list[str], *, strict: bool = True) -> None: ...
    def deactivate_all(self) -> None: ...
    def isolate(self, name: str | list[str], *, strict: bool = True) -> None: ...
    def evaluate(
        self,
        parameters: list[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: list[float] | npt.ArrayLike,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project(
        self,
        parameters: list[float] | npt.ArrayLike,
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_with(
        self,
        parameters: list[float] | npt.ArrayLike,
        name: str | list[str],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: list[float] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder-mead', 'adam', 'pso'] = 'lbfgsb',
        settings: dict | None = None,
        observers: MinimizationObserver | Sequence[MinimizationObserver] | None = None,
        terminators: MinimizationTerminator
        | Sequence[MinimizationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MinimizationSummary: ...
    def mcmc(
        self,
        p0: list[list[float]] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['aies', 'ess'] = 'aies',
        settings: dict | None = None,
        observers: MCMCObserver | Sequence[MCMCObserver] | None = None,
        terminators: MCMCTerminator
        | AutocorrelationTerminator
        | Sequence[MCMCTerminator | AutocorrelationTerminator]
        | None = None,
        max_steps: int | None = None,
        debug: bool = False,
        threads: int = 0,
    ) -> MCMCSummary: ...

def LikelihoodScalar(name: str) -> LikelihoodExpression: ...

class AutocorrelationTerminator:
    taus: npt.NDArray[np.float64]

    def __init__(
        self,
        *,
        n_check: int = 50,
        n_taus_threshold: int = 50,
        dtau_threshold: float = 0.01,
        discard: float = 0.5,
        terminate: bool = True,
        c: float = 7.0,
        verbose: bool = False,
    ) -> None: ...

def integrated_autocorrelation_times(
    samples: npt.ArrayLike, *, c: float | None = None
) -> npt.NDArray[np.float64]: ...

__all__ = [
    'NLL',
    'AutocorrelationTerminator',
    'ControlFlow',
    'EnsembleStatus',
    'LikelihoodEvaluator',
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
    'Swarm',
    'SwarmParticle',
    'Walker',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
