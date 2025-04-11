from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitudes import Evaluator, Model
from laddu.data import Dataset

class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status) -> tuple[Status, bool]: ...

class MCMCObserver(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, ensemble: Ensemble) -> tuple[Ensemble, bool]: ...

class SwarmObserver(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, swarm: Swarm) -> tuple[Swarm, bool]: ...

def likelihood_sum(
    likelihoods: Sequence[LikelihoodID | LikelihoodExpression]
    | Sequence[LikelihoodID]
    | Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def likelihood_product(
    likelihoods: Sequence[LikelihoodID | LikelihoodExpression]
    | Sequence[LikelihoodID]
    | Sequence[LikelihoodExpression],
) -> LikelihoodExpression: ...
def LikelihoodOne() -> LikelihoodExpression: ...
def LikelihoodZero() -> LikelihoodExpression: ...

class LikelihoodID:
    def __add__(
        self, other: LikelihoodID | LikelihoodExpression | int
    ) -> LikelihoodExpression: ...
    def __radd__(
        self, other: LikelihoodID | LikelihoodExpression | int
    ) -> LikelihoodExpression: ...
    def __mul__(
        self, other: LikelihoodID | LikelihoodExpression
    ) -> LikelihoodExpression: ...
    def __rmul__(
        self, other: LikelihoodID | LikelihoodExpression
    ) -> LikelihoodExpression: ...

class LikelihoodExpression:
    def __add__(
        self, other: LikelihoodID | LikelihoodExpression | int
    ) -> LikelihoodExpression: ...
    def __radd__(
        self, other: LikelihoodID | LikelihoodExpression | int
    ) -> LikelihoodExpression: ...
    def __mul__(
        self, other: LikelihoodID | LikelihoodExpression
    ) -> LikelihoodExpression: ...
    def __rmul__(
        self, other: LikelihoodID | LikelihoodExpression
    ) -> LikelihoodExpression: ...

class LikelihoodTerm: ...

class LikelihoodManager:
    def __init__(self) -> None: ...
    def register(
        self, likelihood_term: LikelihoodTerm, *, name: str | None = None
    ) -> LikelihoodID: ...
    def load(
        self, likelihood_expression: LikelihoodExpression
    ) -> LikelihoodEvaluator: ...

class LikelihoodEvaluator:
    parameters: list[str]
    def evaluate(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: list[float] | npt.NDArray[np.float64],
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder_mead'] = 'lbfgsb',
        max_steps: int = 4000,
        debug: bool = False,
        verbose: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> Status: ...
    def mcmc(
        self,
        p0: list[list[float]] | npt.NDArray[np.float64],
        n_steps: int,
        *,
        method: Literal['ESS', 'AIES'] = 'ESS',
        debug: bool = False,
        verbose: bool = False,
        seed: int = 0,
        **kwargs,  # noqa: ANN003
    ) -> Ensemble: ...

class NLL:
    parameters: list[str]
    data: Dataset
    accmc: Dataset
    def __init__(
        self,
        model: Model,
        ds_data: Dataset,
        ds_accmc: Dataset,
    ) -> None: ...
    def as_term(self) -> LikelihoodTerm: ...
    def activate(self, name: str | list[str]) -> None: ...
    def activate_all(self) -> None: ...
    def deactivate(self, name: str | list[str]) -> None: ...
    def deactivate_all(self) -> None: ...
    def isolate(self, name: str | list[str]) -> None: ...
    def evaluate(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        threads: int | None = None,
    ) -> float: ...
    def evaluate_gradient(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def project_with(
        self,
        parameters: list[float] | npt.NDArray[np.float64],
        name: str | list[str],
        *,
        mc_evaluator: Evaluator | None = None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: list[float] | npt.NDArray[np.float64],
        *,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal['lbfgsb', 'nelder_mead'] = 'lbfgsb',
        max_steps: int = 4000,
        debug: bool = False,
        verbose: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> Status: ...
    def mcmc(
        self,
        p0: list[list[float]] | npt.NDArray[np.float64],
        n_steps: int,
        *,
        method: Literal['ESS', 'AIES'] = 'ESS',
        debug: bool = False,
        verbose: bool = False,
        seed: int = 0,
        **kwargs,  # noqa: ANN003
    ) -> Ensemble: ...

def LikelihoodScalar(name: str) -> LikelihoodTerm: ...

class LBFGSB:
    def __init__(
        self,
        eps_f_abs: float | None = None,
        eps_g_abs: float | None = None,
        tol_g_abs: float | None = None,
    ) -> None: ...

class SimplexConstructionMethod:
    @staticmethod
    def orthogonal(simplex_size: int) -> SimplexConstructionMethod: ...
    @staticmethod
    def custom(
        simplex: list[list[float]] | list[npt.NDArray[np.float64]],
    ) -> SimplexConstructionMethod: ...

class NelderMead:
    def __init__(
        self,
        eps_x_rel: float | None = None,
        eps_x_abs: float | None = None,
        eps_f_rel: float | None = None,
        eps_f_abs: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        adaptive: int | None = None,
        construction_method: SimplexConstructionMethod | None = None,
        simplex_expansion_method: str | None = None,
        terminator_f: str | None = None,
        terminator_x: str | None = None,
    ) -> None: ...

class Status:
    x: npt.NDArray[np.float64]
    err: npt.NDArray[np.float64] | None
    x0: npt.NDArray[np.float64]
    fx: float
    cov: npt.NDArray[np.float64] | None
    hess: npt.NDArray[np.float64] | None
    message: str
    converged: bool
    bounds: list[Bound] | None
    n_f_evals: int
    n_g_evals: int

    def __init__(self) -> None: ...
    def save_as(self, path: str) -> None: ...
    @staticmethod
    def load_from(path: str) -> Status: ...
    def as_dict(self) -> dict[str, Any]: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...

class ESSMove:
    @staticmethod
    def differential(weight: float = 1.0) -> ESSMove: ...
    @staticmethod
    def gaussian(weight: float = 1.0) -> ESSMove: ...
    @staticmethod
    def global_move(
        weight: float = 1.0,
        *,
        scale: float | None = None,
        rescale_cov: float | None = None,
        n_components: int | None = None,
    ) -> ESSMove: ...

class ESS:
    def __init__(
        self,
        moves: list[ESSMove],
        n_adaptive: int | None = None,
        max_steps: int | None = None,
        mu: float | None = None,
    ) -> None: ...

class AIESMove:
    @staticmethod
    def stretch(weight: float = 1.0, a: float | None = None) -> AIESMove: ...
    @staticmethod
    def walk(weight: float = 1.0) -> AIESMove: ...

class AIES:
    def __init__(
        self,
        moves: list[AIESMove],
    ) -> None: ...

class Ensemble:
    dimension: tuple[int, int, int]

    def __init__(self) -> None: ...
    def save_as(self, path: str) -> None: ...
    @staticmethod
    def load_from(path: str) -> Status: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...
    def get_chain(self, *, burn: int = 0, thin: int = 1) -> npt.NDArray[np.float64]: ...
    def get_flat_chain(
        self, *, burn: int = 0, thin: int = 1
    ) -> npt.NDArray[np.float64]: ...
    def get_integrated_autocorrelation_times(
        self, *, c: float = 7.0, burn: int = 0, thin: int = 1
    ) -> npt.NDArray[np.float64]: ...

class Point:
    x: npt.NDArray[np.float64]
    fx: float

    def __init__(self) -> None: ...
    def save_as(self, path: str) -> None: ...
    @staticmethod
    def load_from(path: str) -> Status: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...

class Particle:
    position: Point
    velocity: npt.NDArray[np.float64]
    best: Point

    def __init__(self) -> None: ...
    def save_as(self, path: str) -> None: ...
    @staticmethod
    def load_from(path: str) -> Status: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...

class SwarmPositionInitializer:
    @staticmethod
    def zero(n_particles: int, n_dimensions: int) -> SwarmPositionInitializer: ...
    @staticmethod
    def random_in_limits(
        n_particles: int, limits: list[tuple[float, float]]
    ) -> SwarmPositionInitializer: ...
    @staticmethod
    def custom(
        positions: list[list[float]] | list[npt.NDArray[np.float64]],
    ) -> SwarmPositionInitializer: ...
    @staticmethod
    def latin_hypercube(
        n_particles: int, limits: list[tuple[float, float]]
    ) -> SwarmPositionInitializer: ...

class SwarmVelocityInitializer:
    @staticmethod
    def zero() -> SwarmVelocityInitializer: ...
    @staticmethod
    def random_in_limits(
        limits: list[tuple[float, float]],
    ) -> SwarmVelocityInitializer: ...

class Swarm:
    dimension: int
    particles: list[Particle]
    gbest: Point
    message: str
    converged: bool
    bounds: list[Bound] | None

    def __init__(
        self,
        position_initializer: SwarmPositionInitializer,
        *,
        velocity_initializer: SwarmVelocityInitializer | None = None,
        boundary_method: str | None = None,
    ) -> None: ...
    def save_as(self, path: str) -> None: ...
    @staticmethod
    def load_from(path: str) -> Swarm: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...
    def as_dict(self) -> dict[str, Any]: ...

class PSO:
    def __init__(
        self,
        omega: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
        topology: str | None = None,
        update_method: str | None = None,
    ) -> None: ...

class Bound:
    lower: float
    upper: float

def integrated_autocorrelation_times(
    x: npt.NDArray[np.float64], *, c: float = 7.0
) -> npt.NDArray[np.float64]: ...

class AutocorrelationObserver:
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

__all__ = [
    'NLL',
    'AutocorrelationObserver',
    'Bound',
    'Ensemble',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodID',
    'LikelihoodManager',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodTerm',
    'LikelihoodZero',
    'Status',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
