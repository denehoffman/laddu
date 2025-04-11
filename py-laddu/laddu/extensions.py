from __future__ import annotations

from abc import ABCMeta, abstractmethod

from laddu.laddu import (
    AIES,
    ESS,
    LBFGSB,
    NLL,
    PSO,
    AIESMove,
    AutocorrelationObserver,
    Bound,
    Ensemble,
    ESSMove,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodID,
    LikelihoodManager,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodTerm,
    LikelihoodZero,
    NelderMead,
    Particle,
    Point,
    SimplexConstructionMethod,
    Status,
    Swarm,
    SwarmPositionInitializer,
    SwarmVelocityInitializer,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status) -> tuple[Status, bool]:
        pass


class MCMCObserver(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, ensemble: Ensemble) -> tuple[Ensemble, bool]:
        pass


class SwarmObserver(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, swarm: Swarm) -> tuple[Swarm, bool]:
        pass


__all__ = [
    'AIES',
    'ESS',
    'LBFGSB',
    'NLL',
    'PSO',
    'AIESMove',
    'AutocorrelationObserver',
    'Bound',
    'ESSMove',
    'Ensemble',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodID',
    'LikelihoodManager',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodTerm',
    'LikelihoodZero',
    'MCMCObserver',
    'NelderMead',
    'Observer',
    'Particle',
    'Point',
    'SimplexConstructionMethod',
    'Status',
    'Swarm',
    'SwarmObserver',
    'SwarmPositionInitializer',
    'SwarmVelocityInitializer',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
