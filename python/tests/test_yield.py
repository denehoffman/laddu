# ruff: noqa: S101

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import laddu as ld
import pytest

EXPECTED_SELECTED_YIELD = 3.0
EXPECTED_GENERATED_YIELD = 1.5
EXPECTED_CORRECTED_YIELD = 4.5
EXPECTED_REFERENCE_CORRECTED_YIELD = 6.0
EXPECTED_REFERENCE_ACCEPTANCE = 0.5
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


def test_yield_projection_keeps_rate_spaces_and_invalid_bins_visible() -> None:
    model = ld.Model(ld.scalar('x') * ld.parameter('scale', initial=0.25))
    likelihood = ld.Likelihood(
        [ld.ExtendedNLL(model, data=dataset([2.0, 3.0], [1.0, 2.0]), accepted_mc=dataset([4.0], [1.0]), name='signal')]
    )
    result = likelihood.yield_context('signal', generated_mc=dataset([6.0], [1.0]), parameters=[0.25])
    axis = ld.Axis(ld.scalar('x'), edges=[0.0, 5.0, 10.0])
    projection = result.projection(axis)
    assert projection.shape == [2]
    assert projection.selected == [3.0, 0.0]
    assert projection.accepted[0] == pytest.approx(1.0)
    assert projection.generated[1] == pytest.approx(1.5)
    assert projection.validity == ['missing_generated_support', 'missing_accepted_support']
    assert all(math.isnan(value) for value in projection.corrected)
    assert projection.selected_histogram().values == projection.selected

    full = result.projection(ld.Axis(ld.scalar('x'), edges=[0.0, 10.0]))
    assert full.validity == ['valid']
    assert full.acceptance == pytest.approx([2.0 / 3.0])
    assert full.corrected == pytest.approx([4.5])
    projections = result.projection_set({'joint': [axis, axis], 'single': axis})
    assert list(projections) == ['joint', 'single']
    assert projections['joint'].shape == [2, 2]
    assert projections['joint'].selected == [3.0, 0.0, 0.0, 0.0]
    aliases = result.projection_set({'single': axis, 'same': [axis]})
    assert list(aliases) == ['single', 'same']
    assert aliases['single'].selected == aliases['same'].selected

    invalid = ld.Axis(ld.scalar('x'), edges=[-sys.float_info.max, sys.float_info.max])
    with pytest.raises(ld.LadduError, match='invalid bin volume'):
        result.projection_set({'valid': axis, 'invalid': invalid})


def test_yield_projection_rejects_workspace_over_budget_and_stays_usable() -> None:
    x = ld.scalar('x')
    data = dataset([0.25, 1.25], [1.0, 1.0])
    likelihood = ld.Likelihood(
        [ld.NLL(ld.Model(x * 0.0 + 1.0), data=data, accepted_mc=data, name='signal')],
        execution=ld.Execution('cpu', threads=1, memory=ld.MemoryPlan(host='128 KiB')),
    )
    result = likelihood.yield_context('signal', generated_mc=data, parameters=[])
    requests: dict[str, ld.Axis | Sequence[ld.Axis]] = {
        f'projection-{index}': ld.Axis(x, edges=[0.0, 1.0 + index * 1.0e-6, 2.0]) for index in range(1000)
    }
    with pytest.raises(ld.LadduError, match=r'memory|budget'):
        result.projection_set(requests)
    assert result.projection(ld.Axis(x, edges=[0.0, 2.0])).selected == [2.0]


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


def test_reference_correction_is_explicit_distinct_and_inspectable() -> None:
    data = dataset([2.0, 3.0], [1.0, 2.0])
    accepted = dataset([4.0], [1.0])
    fitted = ld.Likelihood(
        [
            ld.ExtendedNLL(
                ld.Model(ld.scalar('x') * ld.parameter('scale', initial=0.25)),
                data=data,
                accepted_mc=accepted,
                name='fitted',
            )
        ]
    ).yield_context(
        'fitted',
        generated_mc=dataset([6.0], [1.0]),
        parameters=[0.25],
    )
    reference = ld.Likelihood(
        [
            ld.ExtendedNLL(
                ld.Model(ld.scalar('x') * ld.parameter('reference_scale', initial=0.25)),
                data=data,
                accepted_mc=accepted,
                name='reference',
            )
        ]
    )

    corrected = fitted.reference_corrected(
        reference,
        'reference',
        generated_mc=dataset([8.0], [1.0]),
        parameters=[0.25],
    )

    assert isinstance(corrected, ld.ReferenceCorrectedYield)
    assert fitted.corrected_observed_yield().central == EXPECTED_CORRECTED_YIELD
    assert corrected.value.central == EXPECTED_REFERENCE_CORRECTED_YIELD
    assert corrected.acceptance.central == EXPECTED_REFERENCE_ACCEPTANCE
    assert corrected.reference_term_name == 'reference'
    assert corrected.provenance.reference_parameters == [0.25]
    assert corrected.rate_closure_status == 'not_applicable'


def test_reference_correction_reports_invalid_contexts() -> None:
    data = dataset([2.0], [1.0])
    fitted_likelihood = ld.Likelihood(
        [
            ld.ExtendedNLL(
                ld.Model(ld.scalar('x')),
                data=data,
                accepted_mc=dataset([4.0], [1.0]),
                name='fitted',
            )
        ]
    )
    fitted = fitted_likelihood.yield_context('fitted', generated_mc=dataset([6.0], [1.0]), parameters=[])
    empty_support = ld.Likelihood(
        [
            ld.ExtendedNLL(
                ld.Model(ld.scalar('x')),
                data=data,
                accepted_mc=dataset([], []),
                name='reference',
            )
        ]
    )

    with pytest.raises(ld.LadduError, match='accepted MC sample has zero fitted support'):
        fitted.reference_corrected(
            empty_support,
            'reference',
            generated_mc=dataset([8.0], [1.0]),
            parameters=[],
        )

    parameterized = ld.Likelihood(
        [
            ld.ExtendedNLL(
                ld.Model(ld.scalar('x') + ld.parameter('offset')),
                data=data,
                accepted_mc=dataset([4.0], [1.0]),
                name='reference',
            )
        ]
    )
    with pytest.raises(ld.LadduError, match='got NaN'):
        fitted.reference_corrected(
            parameterized,
            'reference',
            generated_mc=dataset([8.0], [1.0]),
            parameters=[float('nan')],
        )

    with pytest.raises(TypeError):
        cast('Any', fitted).reference_corrected()
