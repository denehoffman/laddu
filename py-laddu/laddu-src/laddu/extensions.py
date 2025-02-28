from __future__ import annotations

from abc import ABCMeta, abstractmethod

from laddu.laddu import (
    NLL,
    AutocorrelationObserver,
    Bound,
    Ensemble,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodID,
    LikelihoodManager,
    LikelihoodScalar,
    LikelihoodTerm,
    Status,
    integrated_autocorrelation_times,
)


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, status: Status) -> tuple[Status, bool]:
        pass


class MCMCObserver(metaclass=ABCMeta):
    @abstractmethod
    def callback(self, step: int, ensemble: Ensemble) -> tuple[Ensemble, bool]:
        pass


__all__ = [
    'NLL',
    'AutocorrelationObserver',
    'Bound',
    'Ensemble',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodID',
    'LikelihoodManager',
    'LikelihoodScalar',
    'LikelihoodTerm',
    'MCMCObserver',
    'Observer',
    'Status',
    'integrated_autocorrelation_times',
]
