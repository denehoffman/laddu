from __future__ import annotations

from abc import ABCMeta, abstractmethod

from laddu.laddu import (
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
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    StochasticNLL,
    Swarm,
    SwarmParticle,
    Walker,
    likelihood_product,
    likelihood_sum,
)


class MinimizationObserver(metaclass=ABCMeta):
    @abstractmethod
    def observe(self, step: int, status: MinimizationStatus) -> None:
        pass


class MinimizationTerminator(metaclass=ABCMeta):
    @abstractmethod
    def check_for_termination(self, step: int, status: MinimizationStatus) -> ControlFlow:
        pass


class MCMCObserver(metaclass=ABCMeta):
    @abstractmethod
    def observe(self, step: int, status: EnsembleStatus) -> None:
        pass


class MCMCTerminator(metaclass=ABCMeta):
    @abstractmethod
    def check_for_termination(self, step: int, status: EnsembleStatus) -> ControlFlow:
        pass


__all__ = [
    'NLL',
    'AutocorrelationTerminator',
    'ControlFlow',
    'EnsembleStatus',
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
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'StochasticNLL',
    'Swarm',
    'SwarmParticle',
    'Walker',
    'likelihood_product',
    'likelihood_sum',
]
