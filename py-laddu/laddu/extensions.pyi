from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

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
    free_parameters: list[str]
    fixed_parameters: list[str]

    def fix(self, name: str, value: float) -> LikelihoodExpression: ...
    def free(self, name: str) -> LikelihoodExpression: ...
    def rename_parameter(self, old: str, new: str) -> LikelihoodExpression: ...
    def rename_parameters(self, mapping: dict[str, str]) -> LikelihoodExpression: ...
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

class LineSearchConfig:
    @staticmethod
    def morethuente(
        *,
        max_iterations: int | None = None,
        max_zoom: int | None = None,
        c1: float | None = None,
        c2: float | None = None,
    ) -> LineSearchConfig: ...
    @staticmethod
    def hagerzhang(
        *,
        max_iterations: int | None = None,
        max_bisects: int | None = None,
        delta: float | None = None,
        sigma: float | None = None,
        epsilon: float | None = None,
        theta: float | None = None,
        gamma: float | None = None,
    ) -> LineSearchConfig: ...

class SimplexConfig:
    @staticmethod
    def scaled_orthogonal(
        *, orthogonal_multiplier: float = 1.05, orthogonal_zero_step: float = 0.00025
    ) -> SimplexConfig: ...
    @staticmethod
    def orthogonal(*, simplex_size: float = 1.0) -> SimplexConfig: ...
    @staticmethod
    def custom(simplex: list[list[float]] | npt.ArrayLike) -> SimplexConfig: ...

class SwarmInitializerConfig:
    @staticmethod
    def random_in_limits(
        bounds: Sequence[tuple[float, float]], n_particles: int
    ) -> SwarmInitializerConfig: ...
    @staticmethod
    def latin_hypercube(
        bounds: Sequence[tuple[float, float]], n_particles: int
    ) -> SwarmInitializerConfig: ...
    @staticmethod
    def custom(swarm: list[list[float]] | npt.ArrayLike) -> SwarmInitializerConfig: ...

class AIESMoveConfig:
    @staticmethod
    def stretch(weight: float, *, a: float = 2.0) -> AIESMoveConfig: ...
    @staticmethod
    def walk(weight: float) -> AIESMoveConfig: ...

class ESSMoveConfig:
    @staticmethod
    def differential(weight: float) -> ESSMoveConfig: ...
    @staticmethod
    def gaussian(weight: float) -> ESSMoveConfig: ...
    @staticmethod
    def global_(
        weight: float,
        *,
        scale: float | None = None,
        rescale_cov: float | None = None,
        n_components: int | None = None,
    ) -> ESSMoveConfig: ...

class LBFGSBSettings:
    def __init__(
        self,
        *,
        m: int | None = None,
        skip_hessian: bool = False,
        line_search: LineSearchConfig | None = None,
        eps_f: float | None = None,
        eps_g: float | None = None,
        eps_norm_g: float | None = None,
    ) -> None: ...

class AdamSettings:
    def __init__(
        self,
        *,
        alpha: float | None = None,
        beta_1: float | None = None,
        beta_2: float | None = None,
        epsilon: float | None = None,
        beta_c: float | None = None,
        eps_loss: float | None = None,
        patience: int | None = None,
    ) -> None: ...

class NelderMeadSettings:
    def __init__(
        self,
        *,
        simplex: SimplexConfig | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        adaptive: bool = False,
        expansion_method: Literal['greedyminimization', 'greedyexpansion'] | None = None,
        eps_f: float | None = None,
        f_terminator: Literal['amoeba', 'absolute', 'stddev'] | None = None,
        eps_x: float | None = None,
        x_terminator: Literal['diameter', 'higham', 'rowan', 'singer'] | None = None,
    ) -> None: ...

class PSOSettings:
    def __init__(
        self,
        initializer: SwarmInitializerConfig,
        *,
        swarm_topology: Literal['global', 'ring'] = 'global',
        swarm_update_method: Literal[
            'sync', 'synchronous', 'async', 'asynchronous'
        ] = 'sync',
        swarm_boundary_method: Literal['inf', 'shr'] = 'inf',
        use_transform: bool = False,
        swarm_velocity_bounds: Sequence[tuple[float, float]] | None = None,
        omega: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
    ) -> None: ...

class AIESSettings:
    def __init__(self, *, moves: Sequence[AIESMoveConfig] | None = None) -> None: ...

class ESSSettings:
    def __init__(
        self,
        *,
        moves: Sequence[ESSMoveConfig] | None = None,
        n_adaptive: int | None = None,
        mu: float | None = None,
        max_steps: int | None = None,
    ) -> None: ...

MinimizerSettings: TypeAlias = (
    LBFGSBSettings | AdamSettings | NelderMeadSettings | PSOSettings
)
SamplerSettings: TypeAlias = AIESSettings | ESSSettings

class LikelihoodEvaluator:
    parameters: list[str]
    free_parameters: list[str]
    fixed_parameters: list[str]
    n_free: int
    n_fixed: int
    n_parameters: int

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
        settings: MinimizerSettings | None = None,
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
        settings: SamplerSettings | None = None,
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
        settings: MinimizerSettings | None = None,
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
        settings: SamplerSettings | None = None,
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
        *,
        n_mc: float | None = None,
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
    def project_weights(
        self,
        parameters: list[float] | npt.ArrayLike,
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_weights_subset(
        self,
        parameters: list[float] | npt.ArrayLike,
        name: str | list[str],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_weights_subsets(
        self,
        parameters: list[float] | npt.ArrayLike,
        subsets: list[list[str]],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_weights_and_gradients_subset(
        self,
        parameters: list[float] | npt.ArrayLike,
        name: str | list[str],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def minimize(
        self,
        p0: list[float] | npt.ArrayLike,
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder-mead', 'adam', 'pso'] = 'lbfgsb',
        settings: MinimizerSettings | None = None,
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
        settings: SamplerSettings | None = None,
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
    'AIESMoveConfig',
    'AIESSettings',
    'AdamSettings',
    'AutocorrelationTerminator',
    'ControlFlow',
    'ESSMoveConfig',
    'ESSSettings',
    'EnsembleStatus',
    'LBFGSBSettings',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'LineSearchConfig',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'NelderMeadSettings',
    'PSOSettings',
    'SimplexConfig',
    'StochasticNLL',
    'Swarm',
    'SwarmInitializerConfig',
    'SwarmParticle',
    'Walker',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
