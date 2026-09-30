"""Python checks for raw yield and fitted cross-section projections."""

# ruff: noqa: S101

import unittest
from typing import TYPE_CHECKING

import laddu as ld
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


def dataset(values: list[float], weights: list[float]) -> ld.Dataset:
    return ld.Dataset.from_arrays(p4s={}, scalars={'x': np.asarray(values)}, weights=np.asarray(weights))


def fixture() -> tuple[ld.Likelihood, ld.Dataset]:
    x = ld.scalar('x')
    signal = (ld.parameter('signal', initial=1.0) * x).tagged('signal')
    background = ld.parameter('background', initial=1.0).tagged('background')
    model = ld.Model((signal + background).norm_sqr())
    likelihood = ld.Likelihood(
        [
            ld.ExtendedNLL(
                model,
                data=dataset([0.0, 1.0], [1.0, 4.0]),
                accepted_mc=dataset([0.0, 1.0], [1.0, 1.0]),
                name='signal',
            )
        ]
    )
    generated = dataset([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    return likelihood, generated


class ProjectionSetTests(unittest.TestCase):
    def test_fitted_projection_uses_generated_intensity_and_bin_width(self) -> None:
        likelihood, generated = fixture()
        section = likelihood.cross_section(
            'signal',
            generated,
            ld.Luminosity(2.0, ld.AreaUnit.NANOBARN),
            [1.0, 1.0],
        )
        axis = ld.Axis(ld.scalar('x'), edges=[-0.5, 0.5, 2.5])
        result = section.project(
            [axis],
            components={
                'signal': ['signal'],
                'background': ['background'],
                'coherent': ['signal', 'background'],
            },
        )
        np.testing.assert_allclose(result.total.central, [0.5, 3.25])
        np.testing.assert_allclose(result.components['coherent'].central, result.total.central)
        single_axis = section.project(axes=axis)
        np.testing.assert_allclose(single_axis.total.central, result.total.central)
        named = section.project({'fine': axis, 'joint': [axis, axis]})
        assert list(named) == ['fine', 'joint']
        assert named['fine'].shape == [2]
        assert named['joint'].shape == [2, 2]
        np.testing.assert_allclose(named['fine'].total.central, result.total.central)
        with pytest.raises(ld.LadduError, match='at least one projection'):
            section.project({})
        with pytest.raises(TypeError):
            section.project({'invalid': 1.0})  # ty: ignore[invalid-argument-type]
        assert (
            result.components['signal'].central[1] + result.components['background'].central[1]
            != result.total.central[1]
        )

    def test_raw_projection_set_preserves_mapping_order_and_shapes(self) -> None:
        likelihood, generated = fixture()
        yields = likelihood.yield_context('signal', generated_mc=generated, parameters=[1.0, 1.0])
        fine = ld.Axis(ld.scalar('x'), edges=[-0.5, 0.5, 2.5])
        wide = ld.Axis(ld.scalar('x'), edges=[-0.5, 2.5])
        projections = yields.projection_set({'fine': fine, 'joint': [fine, wide]})
        assert list(projections) == ['fine', 'joint']
        assert projections['fine'].shape == [2]
        assert projections['joint'].shape == [2, 1]

    def test_combined_tagged_draws_keep_luminosity_and_branching_normalization(self) -> None:
        x = ld.scalar('x')
        terms = []
        generated = {}
        luminosities = {'p1': 2.0, 'p2': 4.0}
        factors = {'p1': 0.5, 'p2': 0.25}
        for period in luminosities:
            a = (ld.parameter(f'{period}_a', initial=1.0) * x).tagged('a')
            b = ld.parameter(f'{period}_b', initial=1.0).tagged('b')
            terms.append(
                ld.ExtendedNLL(
                    ld.Model((a + b).norm_sqr()),
                    data=dataset([0.25, 0.75], [3.0, -0.2]),
                    accepted_mc=dataset([0.25, 0.75], [1.0, 1.0]),
                    name=period,
                )
            )
            generated[period] = dataset([0.25, 0.75], [2.0, 3.0])
        likelihood = ld.Likelihood(terms, execution=ld.Execution('cpu', normalization='general', mpi=False))
        draws = np.array([[0.5, 1.5, 1.5, 0.5], [1.25, 0.75, 0.75, 1.25]])
        ensemble = ld.Ensemble.from_arrays(draws, parameter_names=likelihood.parameter_names)
        sections = [
            (
                likelihood.cross_section(
                    period,
                    generated[period],
                    ld.Luminosity(luminosity, ld.AreaUnit.NANOBARN),
                    likelihood.default_parameters,
                    ensemble=ensemble,
                ),
                factors[period],
            )
            for period, luminosity in luminosities.items()
        ]
        components: dict[str, Sequence[str]] = {'a': ['a'], 'a_alias': ['a', 'a'], 'coherent': ['b', 'a']}
        axes: dict[str, ld.Axis | Sequence[ld.Axis]] = {
            'fine': ld.Axis(x, edges=[0.0, 0.5, 1.0]),
            'wide': ld.Axis(x, edges=[0.0, 1.0]),
        }
        combined = ld.CrossSection.combine(sections)
        actual = combined.project(axes, components=components)
        assert list(actual) == list(axes)
        rows = np.vstack([likelihood.default_parameters, draws])
        numerator = np.zeros((len(rows), 2))
        a_numerator = np.zeros_like(numerator)
        for period in luminosities:
            a = rows[:, likelihood.parameter_names.index(f'{period}_a'), None]
            b = rows[:, likelihood.parameter_names.index(f'{period}_b'), None]
            numerator += (a * np.array([0.25, 0.75]) + b) ** 2 * [2.0, 3.0]
            a_numerator += (a * np.array([0.25, 0.75])) ** 2 * [2.0, 3.0]
        exposure = sum(luminosities[period] * factors[period] for period in luminosities)
        for name, result in actual.items():
            expected = numerator / exposure
            expected_a = a_numerator / exposure
            if name == 'fine':
                expected /= 0.5
                expected_a /= 0.5
            else:
                expected = expected.sum(axis=1, keepdims=True)
                expected_a = expected_a.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(result.total.central, expected[0])
            np.testing.assert_allclose(result.total.draws, expected[1:])
            np.testing.assert_allclose(result.components['coherent'].central, expected[0])
            np.testing.assert_allclose(result.components['coherent'].draws, expected[1:])
            np.testing.assert_allclose(result.components['a'].central, expected_a[0])
            np.testing.assert_allclose(result.components['a'].draws, expected_a[1:])
            np.testing.assert_allclose(result.components['a_alias'].draws, expected_a[1:])


if __name__ == '__main__':
    unittest.main()
