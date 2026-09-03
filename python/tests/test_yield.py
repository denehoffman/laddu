# ruff: noqa: S101

import laddu as ld
import pytest

EXPECTED_SELECTED_YIELD = 3.0
EXPECTED_GENERATED_YIELD = 1.5
EXPECTED_CORRECTED_YIELD = 4.5
EXPECTED_ACCEPTED_RESIDUAL = -2.0
EXPECTED_ACCEPTED_ABSOLUTE_RESIDUAL = 2.0
ENSEMBLE_SOURCE_ID = 88


def dataset(values: list[float], weights: list[float]) -> ld.Dataset:
    return ld.Dataset.from_arrays(p4s={}, scalars={'x': values}, weights=weights)


def test_yield_exposes_scalar_spaces_and_failed_closure_without_rescaling() -> None:
    model = ld.Model(ld.scalar('x') * ld.parameter('scale', initial=0.25))
    likelihood = ld.Likelihood(
        [
            ld.ExtendedNLL(
                model,
                data=dataset([2.0, 3.0], [1.0, 2.0]),
                accepted_mc=dataset([4.0], [1.0]),
                name='signal',
            )
        ]
    )
    result = likelihood.yield_context(
        'signal',
        generated_mc=dataset([6.0], [1.0]),
        parameters=[0.25],
    )

    assert result.selected_yield().central == EXPECTED_SELECTED_YIELD
    assert result.accepted_fitted_yield().central == 1.0
    assert result.generated_fitted_yield().central == EXPECTED_GENERATED_YIELD
    assert result.fitted_acceptance().central == pytest.approx(2.0 / 3.0)
    assert result.corrected_observed_yield().central == EXPECTED_CORRECTED_YIELD

    closure = result.rate_closure()
    assert closure.status == 'failed'
    assert closure.accepted_residual == EXPECTED_ACCEPTED_RESIDUAL
    assert closure.accepted_absolute_residual == EXPECTED_ACCEPTED_ABSOLUTE_RESIDUAL
    assert closure.accepted_relative_residual == pytest.approx(2.0 / 3.0)
    assert closure.selected_yield.central == EXPECTED_SELECTED_YIELD
    assert not closure.is_closed()


def test_yield_reports_closed_and_shape_only_contexts_explicitly() -> None:
    extended_model = ld.Model(ld.scalar('x') * ld.parameter('scale', initial=0.75))
    data = dataset([2.0, 3.0], [1.0, 2.0])
    accepted = dataset([4.0], [1.0])
    generated = dataset([6.0], [1.0])
    extended = ld.Likelihood(
        [ld.ExtendedNLL(extended_model, data=data, accepted_mc=accepted, name='extended')]
    ).yield_context('extended', generated_mc=generated, parameters=[0.75])
    assert extended.rate_closure().status == 'closed'
    assert extended.rate_closure().is_closed()

    shape_model = ld.Model(ld.scalar('x'))
    shape = ld.Likelihood([ld.NLL(shape_model, data=data, accepted_mc=accepted, name='shape')]).yield_context(
        'shape', generated_mc=generated, parameters=[]
    )
    assert not shape.has_absolute_rate
    assert shape.fitted_acceptance().central == pytest.approx(2.0 / 3.0)
    assert shape.corrected_observed_yield().central == EXPECTED_CORRECTED_YIELD
    with pytest.raises(ld.LadduError, match='does not determine an absolute rate'):
        shape.accepted_fitted_yield()
    closure = shape.rate_closure()
    assert closure.status == 'not_applicable'
    assert closure.reason is not None


def test_yield_preserves_ensemble_draw_order_and_source() -> None:
    model = ld.Model(ld.scalar('x') * ld.parameter('scale', initial=0.25))
    likelihood = ld.Likelihood(
        [
            ld.ExtendedNLL(
                model,
                data=dataset([2.0, 3.0], [1.0, 2.0]),
                accepted_mc=dataset([4.0], [1.0]),
                name='signal',
            )
        ]
    )
    ensemble = ld.Ensemble.from_arrays([[0.5], [0.75]], parameter_names=['scale'], source_id=ENSEMBLE_SOURCE_ID)
    result = likelihood.yield_context(
        'signal',
        generated_mc=dataset([6.0], [1.0]),
        parameters=[0.25],
        ensemble=ensemble,
    )

    accepted = result.accepted_fitted_yield()
    assert accepted.source_id == ENSEMBLE_SOURCE_ID
    assert accepted.draws.tolist() == [2.0, 3.0]
    assert result.selected_yield().draws.tolist() == [3.0, 3.0]
    assert result.corrected_observed_yield().draws.tolist() == [4.5, 4.5]
    closure = result.rate_closure()
    assert closure.accepted_residual_draws == [-1.0, 0.0]
    assert closure.corrected_residual_draws == [1.5, 0.0]
