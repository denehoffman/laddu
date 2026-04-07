from __future__ import annotations

from abc import ABCMeta, abstractmethod

import ganesh

from laddu.laddu import (
    NLL,
    ControlFlow,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    StochasticNLL,
    Swarm,
    SwarmParticle,
    Walker,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)

GradientStatus = ganesh.GradientStatus
GradientFreeStatus = ganesh.GradientFreeStatus
SwarmStatus = ganesh.SwarmStatus
EnsembleStatus = ganesh.EnsembleStatus
MinimizationStatus = GradientStatus | GradientFreeStatus | SwarmStatus
MinimizationSummary = ganesh.MinimizationSummary
MCMCSummary = ganesh.MCMCSummary


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
    'ControlFlow',
    'EnsembleStatus',
    'GradientFreeStatus',
    'GradientStatus',
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
    'SwarmStatus',
    'Walker',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
