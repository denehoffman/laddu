from __future__ import annotations as _annotations

from abc import ABCMeta as _ABCMeta
from abc import abstractmethod as _abstractmethod

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


class MinimizationObserver(metaclass=_ABCMeta):
    @_abstractmethod
    def observe(self, step: int, status: MinimizationStatus) -> None:
        pass


class MinimizationTerminator(metaclass=_ABCMeta):
    @_abstractmethod
    def check_for_termination(self, step: int, status: MinimizationStatus) -> ControlFlow:
        pass


class MCMCObserver(metaclass=_ABCMeta):
    @_abstractmethod
    def observe(self, step: int, status: EnsembleStatus) -> None:
        pass


class MCMCTerminator(metaclass=_ABCMeta):
    @_abstractmethod
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
    'SwarmStatus',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
