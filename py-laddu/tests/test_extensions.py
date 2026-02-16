from __future__ import annotations

import math
from typing import Any, cast

import pytest
from laddu import (
    NLL,
    Dataset,
    Event,
    LikelihoodScalar,
    Scalar,
    Vec3,
    constant,
    likelihood_sum,
    parameter,
)
from laddu.experimental import Regularizer


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
        (
            'evaluate short',
            ValueError,
            'length mismatch',
            lambda: nll.evaluate([]),
        ),
        (
            'evaluate_gradient long',
            ValueError,
            'length mismatch',
            lambda: nll.evaluate_gradient([1.0, 2.0]),
        ),
        (
            'project_with unknown amplitude',
            ValueError,
            'No registered amplitude',
            lambda: nll.project_with([2.0], 'missing_amplitude'),
        ),
        (
            'minimize settings wrong type',
            TypeError,
            'dict',
            lambda: nll.minimize([2.0], settings=cast(Any, [])),
        ),
        (
            'mcmc settings wrong type',
            TypeError,
            'dict',
            lambda: nll.mcmc([[2.0]], settings=cast(Any, [])),
        ),
        (
            'minimize malformed line search method',
            TypeError,
            r'Invalid line search method|not-a-valid-method',
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

    for _label, exc_type, match, fn in cases:
        with pytest.raises(exc_type, match=match):
            fn()


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
    projection = nll.project([2.0])
    assert projection.tolist() == pytest.approx([1.0, 3.0])


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
