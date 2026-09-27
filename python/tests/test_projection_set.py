"""Python checks for raw yield and fitted cross-section projections."""

# ruff: noqa: S101

import unittest

import laddu as ld
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
