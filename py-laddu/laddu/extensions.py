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
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodTerm,
    LikelihoodZero,
    Status,
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
    'MCMCObserver',
    'Observer',
    'Status',
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
]
