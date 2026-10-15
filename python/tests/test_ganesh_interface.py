# ruff: noqa: S101, PLR2004 -- explicit numerical test expectations
"""Exercise ganesh configuration and callback dispatch through the public API."""

import laddu as ld
import numpy as np
import pytest


def likelihood(*, bounded=False):
    x = ld.scalar('x')
    a = ld.parameter('a', initial=0.5, bounds=(-1.0, 2.0) if bounded else None)
    model = ld.Model((a * x + 1.0).norm_sqr())

    def dataset(values):
        return ld.Dataset.from_arrays(p4s={}, scalars={'x': values}, weights=np.ones(len(values)))

    return ld.Likelihood(
        [ld.NLL(model, data=dataset([0.2, 0.4, 0.8]), accepted_mc=dataset([0.1, 0.3, 0.5, 0.9]), name='signal')]
    )


MINIMIZERS = [
    'Adam',
    'ConjugateGradient',
    'LBFGSB',
    'TrustRegion',
    'NelderMead',
    'CMAES',
    'DifferentialEvolution',
    'SimulatedAnnealing',
    'PSO',
]


@pytest.mark.parametrize('algorithm', MINIMIZERS)
@pytest.mark.parametrize('bounded', [False, True])
def test_every_minimizer_accepts_parameter_metadata(algorithm, bounded):
    target = likelihood(bounded=bounded)
    result = target.fit(
        initial=ld.ganesh.VectorInit([0.5]),
        config=getattr(ld.ganesh, algorithm + 'Config')(),
        terminators=[ld.ganesh.MaxSteps(3)],
        seed=12,
    )
    assert isinstance(result, ld.FitResult)
    assert result.parameter_names == ['a']
    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.values))
    if bounded:
        assert -1.0 <= result.values[0] <= 2.0


TERMINATORS = [
    ('Adam', 'AdamEMATerminator'),
    ('ConjugateGradient', 'ConjugateGradientGTerminator'),
    ('LBFGSB', 'LBFGSBFTerminator'),
    ('LBFGSB', 'LBFGSBGTerminator'),
    ('LBFGSB', 'LBFGSBInfNormGTerminator'),
    ('TrustRegion', 'TrustRegionGTerminator'),
    ('NelderMead', 'NelderMeadFTerminator'),
    ('NelderMead', 'NelderMeadXTerminator'),
    ('SimulatedAnnealing', 'SimulatedAnnealingTerminator'),
    *[
        ('CMAES', name)
        for name in (
            'CMAESSigmaTerminator',
            'CMAESNoEffectAxisTerminator',
            'CMAESNoEffectCoordTerminator',
            'CMAESConditionCovTerminator',
            'CMAESEqualFunValuesTerminator',
            'CMAESStagnationTerminator',
            'CMAESTolXUpTerminator',
            'CMAESTolFunTerminator',
            'CMAESTolXTerminator',
        )
    ],
]


@pytest.mark.parametrize(('algorithm', 'terminator'), TERMINATORS)
def test_every_minimizer_terminator_is_accepted(algorithm, terminator):
    fit = likelihood().fit(
        config=getattr(ld.ganesh, algorithm + 'Config')(),
        default_callbacks=False,
        terminators=[getattr(ld.ganesh, terminator)(), ld.ganesh.MaxSteps(3)],
        observers=[ld.ganesh.DebugObserver(), ld.ganesh.ProgressObserver(interval=100)],
    )
    assert np.isfinite(fit.objective)


def test_custom_terminator_replaces_defaults():
    # A huge tolerance would otherwise stop Nelder-Mead on its first step.
    target = likelihood()
    config = ld.ganesh.NelderMeadConfig(initial_simplex=[[0.5], [0.5]])
    steps = []
    target.fit(
        config=config, default_callbacks=False, terminators=[lambda step, _status: steps.append(step) or step >= 4]
    )
    assert steps == list(range(5))


def test_callbacks_preserve_order_and_exceptions():
    calls = []

    def first(step, _status):
        calls.append(step)
        return False

    likelihood().fit(config=ld.ganesh.AdamConfig(), default_callbacks=False, terminators=[first, ld.ganesh.MaxSteps(1)])
    assert calls == [0]
    calls.clear()
    likelihood().fit(
        config=ld.ganesh.NelderMeadConfig(),
        default_callbacks=False,
        terminators=[first, ld.ganesh.NelderMeadFTerminator(tolerance=1e6), ld.ganesh.MaxSteps(10)],
    )
    assert calls == [0]

    def fail(_step, _status):
        message = 'callback failed'
        raise RuntimeError(message)

    for name in ('observers', 'terminators'):
        with pytest.raises(RuntimeError, match='callback failed'):
            likelihood().fit(config=ld.ganesh.AdamConfig(), **{name: [fail]})


def test_incompatible_callbacks_fail_before_evaluating():
    calls = []
    with pytest.raises(TypeError, match='selected ganesh algorithm'):
        likelihood().fit(
            config=ld.ganesh.AdamConfig(),
            terminators=[ld.ganesh.NelderMeadFTerminator()],
            observers=[lambda step, _status: calls.append(step)],
        )
    assert calls == []
    with pytest.raises(TypeError, match='minimizer configuration'):
        likelihood().fit(config=ld.ganesh.AIESConfig())


@pytest.mark.parametrize('algorithm', ['ConjugateGradient', 'LBFGSB'])
@pytest.mark.parametrize('search', ['MoreThuenteLineSearch', 'HagerZhangLineSearch'])
def test_custom_line_searches(algorithm, search):
    fit = likelihood().fit(
        config=getattr(ld.ganesh, algorithm + 'Config')(line_search=getattr(ld.ganesh, search)()),
        terminators=[ld.ganesh.MaxSteps(3)],
    )
    assert np.isfinite(fit.objective)


def test_explicit_bounds_names_and_transforms_survive_expert_path():
    config = ld.ganesh.LBFGSBConfig(
        bounds=[(0.4, 0.6)],
        parameter_names=['custom'],
        error_mode='skip',
        transform=ld.ganesh.ScaleTransform([2.0]),
    )
    fit = likelihood().fit(config=config, use_parameter_metadata=False, terminators=[ld.ganesh.MaxSteps(3)])
    assert fit.parameter_names == ['custom']
    assert 0.8 <= fit.values[0] <= 1.2
    assert fit.covariance is None


@pytest.mark.parametrize('algorithm', ['CMAES', 'DifferentialEvolution', 'SimulatedAnnealing', 'PSO'])
def test_random_minimizers_are_seeded(algorithm):
    target = likelihood()
    config = getattr(ld.ganesh, algorithm + 'Config')()

    def fit(seed):
        return target.fit(config=config, seed=seed, terminators=[ld.ganesh.MaxSteps(4)])

    np.testing.assert_array_equal(fit(7).values, fit(7).values)
    assert not np.array_equal(fit(7).values, fit(8).values)


@pytest.mark.parametrize('algorithm', ['AIES', 'ESS'])
def test_samplers_accept_all_callback_and_config_controls(algorithm):
    target = likelihood(bounded=True)
    init = getattr(ld.ganesh, algorithm + 'Init')([[0.3], [0.4], [0.6], [0.7]])
    config = getattr(ld.ganesh, algorithm + 'Config')(
        parameter_names=['custom'],
        transform=ld.ganesh.ScaleTransform([2.0]),
        chain_storage=ld.ganesh.ChainStorage.sampled(2),
    )

    def sample():
        return target.sample(
            init,
            config=config,
            seed=9,
            default_callbacks=False,
            use_parameter_metadata=False,
            terminators=[ld.ganesh.AutocorrelationTerminator(n_check=100), ld.ganesh.MaxSteps(4)],
        )

    summary = sample()
    assert summary.parameter_names == ['custom']
    assert summary.dimension[0] == 4
    assert summary.dimension[2] == 1
    assert np.isfinite(summary.chain).all()
    np.testing.assert_array_equal(summary.chain, sample().chain)


@pytest.mark.parametrize('fraction', [0.5, 1.0])
def test_stochastic_adam_is_reproducible_and_uses_custom_callbacks(fraction):
    target = likelihood()

    def fit(seed):
        return target.fit_stochastic(
            config=ld.ganesh.AdamConfig(alpha=0.05),
            fraction=fraction,
            seed=seed,
            default_callbacks=False,
            terminators=[ld.ganesh.AdamEMATerminator(patience=100), ld.ganesh.MaxSteps(8)],
        )

    a, b = fit(123), fit(123)
    np.testing.assert_array_equal(a.x, b.x)
    assert np.isfinite(a.fx)
    assert a.x[0] != 0.5
    if fraction < 1:
        assert not np.array_equal(a.x, fit(124).x)
    else:
        # With full batches, stochastic Adam follows deterministic Adam exactly.
        deterministic = target.fit(
            config=ld.ganesh.AdamConfig(alpha=0.05),
            default_callbacks=False,
            terminators=[ld.ganesh.MaxSteps(8)],
        )
        np.testing.assert_allclose(a.x, deterministic.values, rtol=1e-12)


@pytest.mark.parametrize('method', ['fit_restarts', 'bootstrap_fit'])
def test_ensemble_helpers_forward_expert_options(method):
    calls = []
    result = getattr(likelihood(), method)(
        2,
        config=ld.ganesh.AdamConfig(parameter_names=['a']),
        default_callbacks=False,
        use_parameter_metadata=False,
        observers=[lambda step, _status: calls.append(step)],
        terminators=[ld.ganesh.MaxSteps(2)],
    )
    assert result is not None
    assert calls
