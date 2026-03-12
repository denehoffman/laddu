from __future__ import annotations

import math
from typing import Any, cast

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
    'project_weights_subset unknown amplitude': (ValueError, 'No registered amplitude'),
    'minimize settings wrong type': (TypeError, 'dict'),
    'mcmc settings wrong type': (TypeError, 'dict'),
    'minimize malformed line search method': (
        TypeError,
        r'Invalid line search method|not-a-valid-method',
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


def test_python_regression_table_driven_error_paths() -> None:
    nll = _simple_scalar_nll()
    cases = [
        ('evaluate short', lambda: nll.evaluate([])),
        ('evaluate_gradient long', lambda: nll.evaluate_gradient([1.0, 2.0])),
        (
            'project_weights_subset unknown amplitude',
            lambda: nll.project_weights_subset([2.0], 'missing_amplitude'),
        ),
        (
            'minimize settings wrong type',
            lambda: nll.minimize([2.0], settings=cast(Any, [])),
        ),
        ('mcmc settings wrong type', lambda: nll.mcmc([[2.0]], settings=cast(Any, []))),
        (
            'minimize malformed line search method',
            lambda: nll.minimize(
                [2.0],
                settings={
                    'line_search': {
                        'method': 'not-a-valid-method',
                    },
                },
            ),
        ),
    ]

    for label, fn in cases:
        exc_type, match = _ERROR_EXPECTATIONS[label]
        with pytest.raises(exc_type, match=match):
            fn()


def test_minimize_unknown_algorithm_setting_suggests_close_match() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        ValueError,
        match=r"Unknown key 'alpah' in adam settings.*Did you mean 'alpha'\?",
    ):
        nll.minimize(
            [2.0],
            method='adam',
            settings={
                'alpah': 0.1,
            },
        )


def test_minimize_unknown_line_search_key_suggests_close_match() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        ValueError,
        match=(
            r"Unknown key 'max_iteratons' in morethuente line_search settings"
            r".*Did you mean 'max_iterations'\?"
        ),
    ):
        nll.minimize(
            [2.0],
            method='lbfgsb',
            settings={
                'line_search': {
                    'method': 'morethuente',
                    'max_iteratons': 10,
                },
            },
        )


def test_mcmc_unknown_move_setting_suggests_close_match() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        ValueError,
        match=r"Unknown key 'aa' in stretch AIES move settings.*Did you mean 'a'\?",
    ):
        nll.mcmc(
            [[2.0], [2.1]],
            method='aies',
            settings={
                'moves': [('stretch', {'aa': 3.0}, 1.0)],
            },
        )


def test_unknown_algorithm_setting_without_close_match_lists_allowed_keys() -> None:
    nll = _simple_scalar_nll()

    with pytest.raises(
        ValueError,
        match=r"Unknown key 'potato' in adam settings\..*Allowed keys: .*'alpha'",
    ) as error:
        nll.minimize(
            [2.0],
            method='adam',
            settings={
                'potato': 0.1,
            },
        )

    assert 'Did you mean' not in str(error.value)


def test_regularizer_l1_matches_rust_implementation() -> None:
    expr = Regularizer(['alpha', 'beta'], 2.0, weights=[1.0, 0.5])
    evaluator = likelihood_sum([expr]).load()
    assert evaluator.parameters == ['alpha', 'beta']
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


def test_nll_project_weights_subsets_matches_repeated_project_weights_subset() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]
    subsets = [['amp_a'], ['amp_b'], ['amp_a', 'amp_b']]

    batched = nll.project_weights_subsets(params, subsets)
    repeated = [nll.project_weights_subset(params, subset).tolist() for subset in subsets]
    for batched_row, repeated_row in zip(batched.tolist(), repeated):
        assert batched_row == pytest.approx(repeated_row)


def test_nll_project_weights_subsets_handles_empty_and_duplicate_subsets() -> None:
    amp_a = Scalar('amp_a', parameter('alpha'))
    amp_b = Scalar('amp_b', parameter('beta'))
    expr = (amp_a + amp_b).norm_sqr()
    data = _dataset_from_weights([1.0, 2.0])
    mc = _dataset_from_weights([0.5, 1.5])
    nll = NLL(expr, data, mc)
    params = [1.25, -0.5]

    empty = nll.project_weights_subsets(params, [])
    assert empty.shape == (0, 0)

    subsets = [
        ['amp_b'],
        ['amp_a'],
        ['amp_a', 'amp_b'],
        ['amp_a'],
        ['amp_b'],
    ]
    batched = nll.project_weights_subsets(params, subsets)
    repeated = [nll.project_weights_subset(params, subset).tolist() for subset in subsets]
    for batched_row, repeated_row in zip(batched.tolist(), repeated):
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
        weights, gradients = nll.project_weights_and_gradients_subset(params, subset)
        expected_weights = nll.project_weights_subset(params, subset)
        assert weights.tolist() == pytest.approx(expected_weights.tolist())
        assert gradients.shape[0] == weights.shape[0]
        assert gradients.shape[1] == len(params)


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
        nll.project_weights_and_gradients_subset(params, ['scale'], threads=2)
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
        subset_weights, subset_gradients = nll.project_weights_and_gradients_subset(
            params, ['scale'], threads=2
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
    assert fixed.parameters == ['scale']
    assert fixed.free_parameters == []
    assert fixed.fixed_parameters == ['scale']
    assert fixed.n_free == 0
    assert pytest.approx(fixed.evaluate([])) == reference

    renamed = nll.rename_parameter('scale', 'beta')
    assert renamed.parameters == ['beta']
    assert pytest.approx(renamed.evaluate([1.7])) == reference

    renamed_multi = nll.rename_parameters({'scale': 'gamma'})
    assert renamed_multi.parameters == ['gamma']


def test_nll_free_from_fixed_parameter() -> None:
    amp = Scalar('scale', constant('scale', 2.0))
    expr = amp.norm_sqr()
    data = _dataset_from_weights([1.0])
    mc = _dataset_from_weights([1.0])
    nll = NLL(expr, data, mc)
    assert nll.n_free == 0
    assert nll.fixed_parameters == ['scale']

    freed = nll.free('scale')
    assert freed.n_free == 1
    assert freed.free_parameters == ['scale']
    assert freed.fixed_parameters == []

    reference = NLL(Scalar('scale', parameter('scale')).norm_sqr(), data, mc)
    assert pytest.approx(freed.evaluate([2.5])) == reference.evaluate([2.5])


def test_evaluator_parameters_include_fixed_entries() -> None:
    expr = likelihood_sum([LikelihoodScalar('alpha'), LikelihoodScalar('beta')])
    expr = expr.fix('alpha', 1.5)
    evaluator = expr.load()
    assert evaluator.parameters == ['alpha', 'beta']
    assert evaluator.free_parameters == ['beta']
    assert evaluator.fixed_parameters == ['alpha']
    assert evaluator.evaluate([2.0]) == pytest.approx(3.5)
    # NOTE: Passing the wrong number of parameters currently panics within Rust,
    # so we skip calling evaluator.evaluate([10.0, 2.0]) here until the API
    # surfaces a safe error.
    grad_free = evaluator.evaluate_gradient([2.0]).tolist()
    assert grad_free == pytest.approx([1.0])
    # NOTE: evaluator.evaluate_gradient([10.0, 2.0]) would also panic for the
    # same reason as above.
