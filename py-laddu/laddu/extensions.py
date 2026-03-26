from __future__ import annotations

from abc import ABCMeta, abstractmethod

from laddu.laddu import (
    NLL,
    AdamSettings,
    AIESMoveConfig,
    AIESSettings,
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
    ESSMoveConfig,
    ESSSettings,
    LBFGSBSettings,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    LineSearchConfig,
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    NelderMeadSettings,
    PSOSettings,
    SimplexConfig,
    StochasticNLL,
    Swarm,
    SwarmInitializerConfig,
    SwarmParticle,
    Walker,
    integrated_autocorrelation_times,
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
