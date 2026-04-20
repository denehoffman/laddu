from __future__ import annotations

import math
from typing import Any, cast

import ganesh
import numpy as np
import pytest
from laddu import (
    NLL,
    Dataset,
    Event,
    LikelihoodScalar,
    Scalar,
    Vec3,
    constant,
    get_threads,
    likelihood_sum,
    parameter,
    set_threads,
    threads,
)
from laddu.experimental import Regularizer

_ERROR_EXPECTATIONS: dict[str, tuple[type[Exception], str]] = {
    'evaluate short': (ValueError, 'length mismatch'),
    'evaluate_gradient long': (ValueError, 'length mismatch'),
    'project_weights strict subset unknown amplitude': (
        ValueError,
        'No registered amplitude',
    ),
}


def _dataset_from_weights(weights: list[float]) -> Dataset:
    events = [
        Event(
            [Vec3(0.0, 0.0, 0.0).with_mass(0.0)],
            [],
            weight,
            p4_names=['beam'],
            aux_names=[],
        )
        for weight in weights
    ]
    return Dataset(events, p4_names=['beam'], aux_names=[])


def _simple_scalar_nll() -> NLL:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    return NLL(expr, data, mc)


def _two_parameter_nll() -> NLL:
    amp_a = Scalar('alpha_amp', parameter('alpha'))
    amp_b = Scalar('beta_amp', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    return NLL(expr, data, mc)


def test_nll_exposes_evaluators_expression_and_compiled_expression() -> None:
    nll = _two_parameter_nll()

    assert str(nll.data_evaluator.compiled_expression) == str(nll.compiled_expression)
    assert 'alpha_amp(id=0)' in str(nll.expression.compiled_expression)
    assert 'beta_amp(id=1)' in str(nll.accmc_evaluator.expression.compiled_expression)

    nll.deactivate('beta_amp')
    compiled = str(nll.compiled_expression)
    assert 'alpha_amp(id=0)' in compiled
    assert 'beta_amp(id=1)' not in compiled
    assert 'const 0' in compiled


def test_stochastic_nll_exposes_expression_and_compiled_expression() -> None:
    stochastic = _two_parameter_nll().to_stochastic(1, seed=0)

    assert 'alpha_amp(id=0)' in str(stochastic.expression.compiled_expression)
    assert 'beta_amp(id=1)' in str(stochastic.compiled_expression)


def _assert_ganesh_summary_class(summary: object, expected_name: str) -> None:
    assert summary.__class__.__module__ == 'ganesh'
    assert summary.__class__.__name__ == expected_name


def test_integrated_autocorrelation_times_shape() -> None:
    import laddu as ld

    samples = np.random.default_rng(0).normal(size=(4, 16, 3))
    taus = ld.integrated_autocorrelation_times(samples)

    assert taus.shape == (3,)
    assert np.all(np.isfinite(taus))


def test_python_regression_table_driven_error_paths() -> None:
    nll = _simple_scalar_nll()
    cases = [
        ('evaluate short', lambda: nll.evaluate([])),
        ('evaluate_gradient long', lambda: nll.evaluate_gradient([1.0, 2.0])),
        (
            'project_weights strict subset unknown amplitude',
            lambda: nll.project_weights([2.0], subset='missing_amplitude', strict=True),
        ),
    ]

    for label, fn in cases:
        exc_type, match = _ERROR_EXPECTATIONS[label]
        with pytest.raises(exc_type, match=match):
            fn()


def test_minimize_typed_config_requires_matching_method() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        TypeError,
        match=r'structural `LBFGSBConfig` extraction requires',
    ):
        nll.minimize(
            [2.0],
            method='lbfgsb',
            config=ganesh.AdamConfig(alpha=0.1),
            options=ganesh.LBFGSBOptions(max_steps=1),
        )


def test_mcmc_typed_config_requires_matching_method() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        TypeError,
        match=r'structural `ESSConfig` extraction requires',
    ):
        nll.mcmc(
            [[2.0], [2.1]],
            method='ess',
            config=ganesh.AIESConfig(moves=[ganesh.AIESStretchMove(1.0)]),
            options=ganesh.ESSOptions(max_steps=1),
        )


def test_minimize_wrong_type_config_fails_fast() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        TypeError,
        match=r'structural `LBFGSBConfig` extraction requires',
    ):
        nll.minimize(
            [2.0],
            config=cast(Any, []),
            options=ganesh.LBFGSBOptions(max_steps=1),
        )


def test_mcmc_wrong_type_config_fails_fast() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        TypeError,
        match=r'structural `AIESConfig` extraction requires',
    ):
        nll.mcmc(
            [[2.0], [2.1]],
            config=cast(Any, []),
            options=ganesh.AIESOptions(max_steps=1),
        )


def test_minimize_accepts_structural_ganesh_config() -> None:
    class ConfigLike:
        def __init__(self) -> None:
            self.memory_limit = 4
            self.bounds = None
            self.parameter_names = None
            self.bounds_handling = None
            self.line_search = None
            self.error_mode = None

    nll = _simple_scalar_nll()
    summary = nll.minimize(
        [2.0],
        method='lbfgsb',
        config=cast(Any, ConfigLike()),
        options=ganesh.LBFGSBOptions(max_steps=2),
    )

    _assert_ganesh_summary_class(summary, 'MinimizationSummary')
    assert summary.parameter_names == ['scale']


def test_minimize_rejects_duck_typed_wrong_ganesh_config() -> None:
    class ConfigLike:
        def __ganesh_config__(self) -> ganesh.AdamConfig:
            return ganesh.AdamConfig(alpha=0.1)

    nll = _simple_scalar_nll()

    with pytest.raises(TypeError, match=r'structural `LBFGSBConfig` extraction requires'):
        nll.minimize(
            [2.0],
            method='lbfgsb',
            config=cast(Any, ConfigLike()),
            options=ganesh.LBFGSBOptions(max_steps=1),
        )


def test_mcmc_accepts_structural_ganesh_init() -> None:
    class InitLike:
        def __init__(self) -> None:
            self.walkers = [[2.0], [2.1]]

    nll = _simple_scalar_nll()
    summary = nll.mcmc(
        cast(Any, InitLike()),
        method='aies',
        options=ganesh.AIESOptions(max_steps=2),
    )

    _assert_ganesh_summary_class(summary, 'MCMCSummary')
    assert summary.parameter_names == ['scale']


def test_ganesh_minimize_config_options_smoke() -> None:
    nll = _two_parameter_nll()

    lbfgsb = nll.minimize(
        [2.0, -0.5],
        method='lbfgsb',
        config=ganesh.LBFGSBConfig(
            memory_limit=4,
            line_search=ganesh.MoreThuenteLineSearch(max_iterations=8),
        ),
        options=ganesh.LBFGSBOptions(max_steps=4),
    )
    adam = nll.minimize(
        [2.0, -0.5],
        method='adam',
        config=ganesh.AdamConfig(alpha=0.05),
        options=ganesh.AdamOptions(
            max_steps=4,
            ema=ganesh.AdamEMATerminator(patience=2),
        ),
    )
    conjugate_gradient = nll.minimize(
        [2.0, -0.5],
        method='conjugate-gradient',
        config=ganesh.ConjugateGradientConfig(),
        options=ganesh.ConjugateGradientOptions(max_steps=4),
    )
    trust_region = nll.minimize(
        [2.0, -0.5],
        method='trust-region',
        config=ganesh.TrustRegionConfig(),
        options=ganesh.TrustRegionOptions(max_steps=4),
    )
    nelder_mead = nll.minimize(
        ganesh.NelderMeadInit(
            construction_method=ganesh.OrthogonalSimplex([2.0, -0.5], simplex_size=0.25),
        ),
        method='nelder-mead',
        config=ganesh.NelderMeadConfig(expansion_method='greedy_expansion'),
        options=ganesh.NelderMeadOptions(
            max_steps=4,
            f_terminators=ganesh.NelderMeadAbsoluteFTerminator(),
            x_terminators=ganesh.NelderMeadDiameterXTerminator(),
        ),
    )
    pso = nll.minimize(
        ganesh.PSOInit(
            np.array(
                [
                    [2.25, -0.5],
                    [1.75, -0.75],
                    [2.5, 0.0],
                    [1.5, -0.25],
                ],
                dtype=np.float64,
            )
        ),
        method='pso',
        config=ganesh.PSOConfig(
            bounds=[(0.0, 3.0), (-2.0, 2.0)], omega=0.7, c1=0.2, c2=0.2
        ),
        options=ganesh.PSOOptions(max_steps=2),
    )

    for summary in (
        lbfgsb,
        adam,
        conjugate_gradient,
        trust_region,
        nelder_mead,
        pso,
    ):
        _assert_ganesh_summary_class(summary, 'MinimizationSummary')
        assert summary.parameter_names == ['alpha', 'beta']
        assert summary.x.shape == (2,)


def test_nelder_mead_custom_simplex_accepts_numpy_array() -> None:
    nll = _two_parameter_nll()

    summary = nll.minimize(
        ganesh.NelderMeadInit(
            construction_method=ganesh.CustomSimplex(
                np.array(
                    [
                        [2.0, -0.5],
                        [2.25, -0.5],
                        [2.0, -0.25],
                    ],
                    dtype=np.float64,
                )
            )
        ),
        method='nelder-mead',
        options=ganesh.NelderMeadOptions(max_steps=4),
    )

    assert summary.parameter_names == ['alpha', 'beta']
    assert summary.x.shape == (2,)


def test_pso_init_accepts_numpy_array() -> None:
    nll = _two_parameter_nll()

    summary = nll.minimize(
        ganesh.PSOInit(
            np.array(
                [
                    [2.25, -0.5],
                    [1.75, -0.75],
                    [2.5, 0.0],
                ],
                dtype=np.float64,
            )
        ),
        method='pso',
        config=ganesh.PSOConfig(omega=0.7, c1=0.2, c2=0.2),
        options=ganesh.PSOOptions(max_steps=2),
    )

    assert summary.parameter_names == ['alpha', 'beta']
    assert summary.x.shape == (2,)


def test_minimize_method_and_line_search_aliases_match_canonical() -> None:
    nll = _simple_scalar_nll()

    canonical = nll.minimize(
        [2.0],
        method='lbfgsb',
        config=ganesh.LBFGSBConfig(
            line_search=ganesh.MoreThuenteLineSearch(),
        ),
        options=ganesh.LBFGSBOptions(max_steps=4),
    )
    alias = nll.minimize(
        [2.0],
        method=cast(Any, 'L-BFGS-B'),
        config=ganesh.LBFGSBConfig(
            line_search=ganesh.MoreThuenteLineSearch(),
        ),
        options=ganesh.LBFGSBOptions(max_steps=4),
    )

    np.testing.assert_allclose([alias.fx], [canonical.fx], equal_nan=True)
    np.testing.assert_allclose(alias.x, canonical.x, equal_nan=True)
    assert alias.parameter_names == canonical.parameter_names


def test_mcmc_method_alias_matches_canonical_shape() -> None:
    nll = _simple_scalar_nll()

    canonical = nll.mcmc(
        [[2.0], [2.1]],
        method='aies',
        config=ganesh.AIESConfig(moves=[ganesh.AIESStretchMove(1.0)]),
        options=ganesh.AIESOptions(max_steps=2),
    )
    alias = nll.mcmc(
        [[2.0], [2.1]],
        method=cast(Any, 'A I E S'),
        config=ganesh.AIESConfig(moves=[ganesh.AIESStretchMove(1.0)]),
        options=ganesh.AIESOptions(max_steps=2),
    )

    assert alias.dimension == canonical.dimension
    assert alias.parameter_names == canonical.parameter_names


def test_minimize_method_typo_still_raises_value_error() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(ValueError, match=r'Invalid minimizer: lbgfsb'):
        nll.minimize([2.0], method=cast(Any, 'lbgfsb'))


def test_ganesh_mcmc_config_options_smoke() -> None:
    nll = _simple_scalar_nll()

    aies = nll.mcmc(
        [[2.0], [2.1]],
        method='aies',
        config=ganesh.AIESConfig(moves=[ganesh.AIESStretchMove(1.0, a=3.0)]),
        options=ganesh.AIESOptions(max_steps=2),
    )
    ess = nll.mcmc(
        [[2.0], [2.1], [1.9]],
        method='ess',
        config=ganesh.ESSConfig(
            moves=[ganesh.ESSGlobalMove(1.0, scale=1.5, n_components=2)],
            n_adaptive=1,
            mu=1.1,
        ),
        options=ganesh.ESSOptions(max_steps=2),
    )

    assert aies.parameter_names == ['scale']
    assert ess.parameter_names == ['scale']
    _assert_ganesh_summary_class(aies, 'MCMCSummary')
    _assert_ganesh_summary_class(ess, 'MCMCSummary')


def test_ganesh_sampler_defaults_construct_and_run() -> None:
    nll = _simple_scalar_nll()

    aies = nll.mcmc(
        [[2.0], [2.1]],
        method='aies',
        config=ganesh.AIESConfig(),
        options=ganesh.AIESOptions(max_steps=2),
    )
    ess = nll.mcmc(
        [[2.0], [2.1], [1.9], [2.2]],
        method='ess',
        config=ganesh.ESSConfig(),
        options=ganesh.ESSOptions(max_steps=2),
    )

    assert aies.dimension[2] == 1
    assert ess.dimension[2] == 1


def test_regularizer_l1_matches_rust_implementation() -> None:
    expr = Regularizer(['alpha', 'beta'], 2.0, weights=[1.0, 0.5])
    evaluator = likelihood_sum([expr]).load()
    assert evaluator.parameters == ('alpha', 'beta')
    params = [1.5, -2.0]
    assert evaluator.evaluate(params) == pytest.approx(7.0)
    grad = evaluator.evaluate_gradient(params).tolist()
    assert grad == pytest.approx([2.0, -1.0])


def test_regularizer_l2_gradient_matches_rust() -> None:
    expr = Regularizer(['x', 'y'], 3.0, p=2, weights=[1.0, 2.0])
    evaluator = likelihood_sum([expr]).load()
    params = [3.0, 4.0]
    assert evaluator.evaluate(params) == pytest.approx(15.0)
    grad = evaluator.evaluate_gradient(params).tolist()
    denom = math.sqrt(1.0 * params[0] ** 2 + 2.0 * params[1] ** 2)
    assert grad == pytest.approx([3.0 * params[0] / denom, 3.0 * params[1] / denom])


def test_regularizer_weight_mismatch_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        Regularizer(['alpha', 'beta'], 1.0, weights=[1.0])


def test_regularizer_invalid_norm_raises() -> None:
    with pytest.raises(ValueError):
        Regularizer(['alpha'], 1.0, p=3)


def test_nll_matches_constant_model() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [2.0]
    intensity = params[0] ** 2
    expected = -2.0 * (3.0 * math.log(intensity) - intensity)
    assert nll.evaluate(params) == pytest.approx(expected)
    grad = nll.evaluate_gradient(params)
    assert grad.tolist() == pytest.approx([-4.0 * (3.0 / params[0] - params[0])])


def test_nll_project_returns_expected_weights() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    projection = nll.project_weights([2.0])
    assert projection.tolist() == pytest.approx([1.0, 3.0])


def test_nll_project_weights_subsets_matches_repeated_subset_projection() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [['amp_a'], ['amp_b'], ['amp_a', 'amp_b']]

    batched = nll.project_weights(params, subsets=subsets)
    repeated = [nll.project_weights(params, subset=subset).tolist() for subset in subsets]
    for batched_row, repeated_row in zip(batched.tolist(), repeated, strict=False):
        assert batched_row == pytest.approx(repeated_row)


def test_nll_project_weights_subsets_handles_empty_and_duplicate_subsets() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]

    empty = nll.project_weights(params, subsets=[])
    assert empty.shape == (0, 0)

    subsets = [
        ['amp_b'],
        ['amp_a'],
        ['amp_a', 'amp_b'],
        ['amp_a'],
        ['amp_b'],
    ]
    batched = nll.project_weights(params, subsets=subsets)
    repeated = [nll.project_weights(params, subset=subset).tolist() for subset in subsets]
    for batched_row, repeated_row in zip(batched.tolist(), repeated, strict=False):
        assert batched_row == pytest.approx(repeated_row)


def test_nll_project_weights_subsets_allow_none_for_total_projection() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [None, ['amp_a'], ['amp_b'], ['amp_a', 'amp_b']]

    batched = nll.project_weights(params, subsets=subsets)
    repeated = [
        nll.project_weights(params).tolist(),
        nll.project_weights(params, subset=['amp_a']).tolist(),
        nll.project_weights(params, subset=['amp_b']).tolist(),
        nll.project_weights(params, subset=['amp_a', 'amp_b']).tolist(),
    ]
    assert batched.shape == (len(subsets), 2)
    for batched_row, repeated_row in zip(batched.tolist(), repeated, strict=False):
        assert batched_row == pytest.approx(repeated_row)


def test_nll_project_weights_default_skips_missing_subset_names() -> None:
    nll = _two_parameter_nll()
    params = [1.25, -0.5]

    mixed = nll.project_weights(params, subset=['alpha_amp', 'missing_amplitude'])
    valid_only = nll.project_weights(params, subset=['alpha_amp'])
    empty = nll.project_weights(params, subset=['missing_amplitude'])

    assert mixed.tolist() == pytest.approx(valid_only.tolist())
    assert empty.tolist() == pytest.approx([0.0, 0.0])


def test_nll_project_weights_subsets_default_skips_missing_names() -> None:
    nll = _two_parameter_nll()
    params = [1.25, -0.5]
    subsets = [None, ['alpha_amp', 'missing_amplitude'], ['missing_amplitude']]

    batched = nll.project_weights(params, subsets=subsets)
    repeated = [
        nll.project_weights(params).tolist(),
        nll.project_weights(params, subset=['alpha_amp']).tolist(),
        [0.0, 0.0],
    ]

    assert batched.shape == (len(subsets), 2)
    for batched_row, repeated_row in zip(batched.tolist(), repeated, strict=False):
        assert batched_row == pytest.approx(repeated_row)


def test_nll_project_weights_and_gradients_subset_matches_expected_weights() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [['amp_b'], ['amp_a'], ['amp_a', 'amp_b'], ['amp_a']]

    for subset in subsets:
        weights, gradients = nll.project_weights_and_gradients(params, subset=subset)
        expected_weights = nll.project_weights(params, subset=subset)
        assert weights.tolist() == pytest.approx(expected_weights.tolist())
        assert gradients.shape[0] == weights.shape[0]
        assert gradients.shape[1] == len(params)


def test_nll_project_weights_and_gradients_subsets_matches_repeated_subset_calls() -> (
    None
):
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [['amp_b'], ['amp_a'], ['amp_a', 'amp_b'], ['amp_a']]

    batched_weights, batched_gradients = nll.project_weights_and_gradients(
        params, subsets=subsets
    )
    assert batched_weights.shape == (len(subsets), 2)
    assert batched_gradients.shape == (len(subsets), 2, len(params))

    for index, subset in enumerate(subsets):
        expected_weights, expected_gradients = nll.project_weights_and_gradients(
            params, subset=subset
        )
        assert batched_weights[index].tolist() == pytest.approx(expected_weights.tolist())
        np.testing.assert_allclose(batched_gradients[index], expected_gradients)


def test_nll_project_weights_and_gradients_subsets_allow_none_for_total_projection() -> (
    None
):
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [None, ['amp_a'], ['amp_b'], ['amp_a', 'amp_b']]

    batched_weights, batched_gradients = nll.project_weights_and_gradients(
        params, subsets=subsets
    )
    assert batched_weights.shape == (len(subsets), 2)
    assert batched_gradients.shape == (len(subsets), 2, len(params))

    repeated = [
        nll.project_weights_and_gradients(params),
        nll.project_weights_and_gradients(params, subset=['amp_a']),
        nll.project_weights_and_gradients(params, subset=['amp_b']),
        nll.project_weights_and_gradients(params, subset=['amp_a', 'amp_b']),
    ]
    for index, (expected_weights, expected_gradients) in enumerate(repeated):
        assert batched_weights[index].tolist() == pytest.approx(expected_weights.tolist())
        np.testing.assert_allclose(batched_gradients[index], expected_gradients)


def test_nll_project_weights_and_gradients_default_skips_missing_subset_names() -> None:
    nll = _two_parameter_nll()
    params = [1.25, -0.5]

    mixed_weights, mixed_gradients = nll.project_weights_and_gradients(
        params, subset=['alpha_amp', 'missing_amplitude']
    )
    valid_weights, valid_gradients = nll.project_weights_and_gradients(
        params, subset=['alpha_amp']
    )
    empty_weights, empty_gradients = nll.project_weights_and_gradients(
        params, subset=['missing_amplitude']
    )

    assert mixed_weights.tolist() == pytest.approx(valid_weights.tolist())
    np.testing.assert_allclose(mixed_gradients, valid_gradients)
    assert empty_weights.tolist() == pytest.approx([0.0, 0.0])
    np.testing.assert_allclose(empty_gradients, np.zeros((2, len(params))))


def test_nll_project_weights_and_gradients_subsets_default_skips_missing_names() -> None:
    nll = _two_parameter_nll()
    params = [1.25, -0.5]
    subsets = [None, ['alpha_amp', 'missing_amplitude'], ['missing_amplitude']]

    batched_weights, batched_gradients = nll.project_weights_and_gradients(
        params, subsets=subsets
    )
    repeated = [
        nll.project_weights_and_gradients(params),
        nll.project_weights_and_gradients(params, subset=['alpha_amp']),
        (np.array([0.0, 0.0]), np.zeros((2, len(params)))),
    ]

    assert batched_weights.shape == (len(subsets), 2)
    assert batched_gradients.shape == (len(subsets), 2, len(params))
    for index, (expected_weights, expected_gradients) in enumerate(repeated):
        assert batched_weights[index].tolist() == pytest.approx(expected_weights.tolist())
        np.testing.assert_allclose(batched_gradients[index], expected_gradients)


def test_nll_project_weights_rejects_subset_and_subsets_together() -> None:
    nll = _simple_scalar_nll()
    dynamic_nll = cast(Any, nll)

    with pytest.raises(ValueError, match='mutually exclusive'):
        dynamic_nll.project_weights([2.0], subset='scale', subsets=[['scale']])

    with pytest.raises(ValueError, match='mutually exclusive'):
        dynamic_nll.project_weights_and_gradients(
            [2.0], subset='scale', subsets=[['scale']]
        )


def test_repeated_short_calls_with_threads_remain_stable() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    dataset = _dataset_from_weights([1.0, 2.0, 3.0, 4.0])
    evaluator = expr.load(dataset)
    nll = NLL(expr, dataset, _dataset_from_weights([0.5, 1.5, 2.5, 0.5]))
    params = [1.25]

    expected_eval = evaluator.evaluate(params, threads=2)
    expected_batch = evaluator.evaluate_batch(params, [0, 2], threads=2)
    expected_gradient = evaluator.evaluate_gradient(params, threads=2)
    expected_gradient_batch = evaluator.evaluate_gradient_batch(params, [0, 2], threads=2)
    expected_nll_value = nll.evaluate(params, threads=2)
    expected_nll_gradient = nll.evaluate_gradient(params, threads=2)
    expected_projection = nll.project_weights(params, threads=2)
    expected_subset_weights, expected_subset_gradients = (
        nll.project_weights_and_gradients(params, subset=['scale'], threads=2)
    )

    for _ in range(32):
        assert evaluator.evaluate(params, threads=2) == pytest.approx(expected_eval)
        assert evaluator.evaluate_batch(
            params, [0, 2], threads=2
        ).tolist() == pytest.approx(expected_batch.tolist())
        np.testing.assert_allclose(
            evaluator.evaluate_gradient(params, threads=2), expected_gradient
        )
        np.testing.assert_allclose(
            evaluator.evaluate_gradient_batch(params, [0, 2], threads=2),
            expected_gradient_batch,
        )

        assert nll.evaluate(params, threads=2) == pytest.approx(expected_nll_value)
        assert nll.evaluate_gradient(params, threads=2).tolist() == pytest.approx(
            expected_nll_gradient.tolist()
        )
        assert nll.project_weights(params, threads=2).tolist() == pytest.approx(
            expected_projection.tolist()
        )
        subset_weights, subset_gradients = nll.project_weights_and_gradients(
            params, subset=['scale'], threads=2
        )
        assert subset_weights.tolist() == pytest.approx(expected_subset_weights.tolist())
        np.testing.assert_allclose(subset_gradients, expected_subset_gradients)


def test_set_threads_aligns_omitted_and_zero_thread_requests() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    dataset = _dataset_from_weights([1.0, 2.0, 3.0, 4.0])
    evaluator = expr.load(dataset)
    nll = NLL(expr, dataset, _dataset_from_weights([0.5, 1.5, 2.5, 0.5]))
    params = [1.25]

    try:
        set_threads(2)

        assert evaluator.evaluate(params) == pytest.approx(
            evaluator.evaluate(params, threads=0)
        )
        assert evaluator.evaluate(params, threads=None) == pytest.approx(
            evaluator.evaluate(params, threads=0)
        )
        np.testing.assert_allclose(
            evaluator.evaluate_gradient(params, threads=None),
            evaluator.evaluate_gradient(params, threads=0),
        )
        assert nll.evaluate(params, threads=None) == pytest.approx(
            nll.evaluate(params, threads=0)
        )
        assert nll.project_weights(params).tolist() == pytest.approx(
            nll.project_weights(params, threads=0).tolist()
        )
        assert nll.project_weights(params, threads=None).tolist() == pytest.approx(
            nll.project_weights(params, threads=0).tolist()
        )

        set_threads(1)
        assert evaluator.evaluate(params) == pytest.approx(
            evaluator.evaluate(params, threads=0)
        )
        assert nll.evaluate(params) == pytest.approx(nll.evaluate(params, threads=0))
        assert evaluator.evaluate(params, threads=2) == pytest.approx(
            evaluator.evaluate(params, threads=0)
        )
    finally:
        set_threads(0)


def test_threads_context_manager_restores_previous_default() -> None:
    try:
        set_threads(0)
        assert get_threads() == 0

        with threads(2):
            assert get_threads() == 2
        assert get_threads() == 0
    finally:
        set_threads(0)


def test_threads_context_manager_nests() -> None:
    try:
        set_threads(1)
        with threads(2):
            assert get_threads() == 2
            with threads(3):
                assert get_threads() == 3
            assert get_threads() == 2
        assert get_threads() == 1
    finally:
        set_threads(0)


def test_threads_context_manager_restores_after_exception() -> None:
    try:
        set_threads(1)
        with threads(2):
            message = 'boom'
            with pytest.raises(RuntimeError, match=message):
                raise RuntimeError(message)
        assert get_threads() == 1
    finally:
        set_threads(0)


def test_threads_context_aligns_none_and_zero_requests() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    dataset = _dataset_from_weights([1.0, 2.0, 3.0, 4.0])
    evaluator = expr.load(dataset)
    params = [1.25]

    try:
        set_threads(0)
        with threads(2):
            assert get_threads() == 2
            assert evaluator.evaluate(params, threads=None) == pytest.approx(
                evaluator.evaluate(params, threads=0)
            )
            np.testing.assert_allclose(
                evaluator.evaluate_gradient(params, threads=None),
                evaluator.evaluate_gradient(params, threads=0),
            )
        assert get_threads() == 0
    finally:
        set_threads(0)


def test_explicit_thread_argument_overrides_context_default_for_a_single_call() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    dataset = _dataset_from_weights([1.0, 2.0, 3.0, 4.0])
    evaluator = expr.load(dataset)
    nll = NLL(expr, dataset, _dataset_from_weights([0.5, 1.5, 2.5, 0.5]))
    params = [1.25]

    try:
        set_threads(1)
        with threads(2):
            assert get_threads() == 2
            assert evaluator.evaluate(params, threads=3) == pytest.approx(
                evaluator.evaluate(params, threads=0)
            )
            assert nll.evaluate(params, threads=3) == pytest.approx(
                nll.evaluate(params, threads=None)
            )
            assert get_threads() == 2
        assert get_threads() == 1
    finally:
        set_threads(0)


def test_nll_parameter_fix_free_and_rename() -> None:
    amp = Scalar('scale', parameter('scale'))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)

    reference = nll.evaluate([1.7])
    fixed = nll.fix('scale', 1.7)
    assert fixed.parameters == ('scale',)
    assert fixed.free_parameters == ()
    assert fixed.fixed_parameters == ('scale',)
    assert fixed.n_free == 0
    assert pytest.approx(fixed.evaluate([])) == reference

    renamed = nll.rename_parameter('scale', 'beta')
    assert renamed.parameters == ('beta',)
    assert pytest.approx(renamed.evaluate([1.7])) == reference

    renamed_multi = nll.rename_parameters({'scale': 'gamma'})
    assert renamed_multi.parameters == ('gamma',)


def test_nll_free_from_fixed_parameter() -> None:
    amp = Scalar('scale', constant('scale', 2.0))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0])
    mc = _dataset_from_weights([1.0])
    nll = NLL(expr, data, mc)
    assert nll.n_free == 0
    assert nll.fixed_parameters == ('scale',)

    freed = nll.free('scale')
    assert freed.n_free == 1
    assert freed.free_parameters == ('scale',)
    assert freed.fixed_parameters == ()

    reference = NLL(Scalar('scale', parameter('scale')).norm_sqr(), data, mc)
    assert pytest.approx(freed.evaluate([2.5])) == reference.evaluate([2.5])


def test_evaluator_parameters_include_fixed_entries() -> None:
    expr = likelihood_sum([LikelihoodScalar('alpha'), LikelihoodScalar('beta')])
    expr = expr.fix('alpha', 1.5)
    evaluator = expr.load()
    assert evaluator.parameters == ('alpha', 'beta')
    assert evaluator.free_parameters == ('beta',)
    assert evaluator.fixed_parameters == ('alpha',)
    assert evaluator.evaluate([2.0]) == pytest.approx(3.5)
    # NOTE: Passing the wrong number of parameters currently panics within Rust,
    # so we skip calling evaluator.evaluate([10.0, 2.0]) here until the API
    # surfaces a safe error.
    grad_free = evaluator.evaluate_gradient([2.0]).tolist()
    assert grad_free == pytest.approx([1.0])
    # NOTE: evaluator.evaluate_gradient([10.0, 2.0]) would also panic for the
    # same reason as above.
