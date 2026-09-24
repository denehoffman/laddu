# ruff: noqa: PT027, S101

import unittest
from typing import TYPE_CHECKING, Any, cast

import laddu as ld
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class ProjectionSetTests(unittest.TestCase):
    @staticmethod
    def yield_context() -> ld.Yield:
        x = ld.scalar('x')
        signal = (ld.parameter('signal', initial=1.5) * x).tagged('signal')
        background = ld.parameter('background', initial=0.75).tagged('background')
        model = ld.Model((signal + background).norm_sqr())

        def dataset(values: list[float], weights: list[float]) -> ld.Dataset:
            return ld.Dataset.from_arrays(p4s={}, scalars={'x': np.asarray(values)}, weights=np.asarray(weights))

        likelihood = ld.Likelihood(
            [
                ld.ExtendedNLL(
                    model,
                    data=dataset([0.25, 0.75, 1.25], [1.0, 2.0, 1.0]),
                    accepted_mc=dataset([0.25, 0.75, 1.25], [1.0, 1.0, 1.0]),
                    name='signal',
                )
            ]
        )
        return likelihood.yield_context(
            'signal',
            generated_mc=dataset([0.25, 0.75, 1.25, 1.75], [1.0, 1.0, 1.0, 1.0]),
            parameters=[1.5, 0.75],
        )

    def test_mapping_order_axis_shapes_and_global_components_are_preserved(self) -> None:
        result = self.yield_context()
        fine = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])
        wide = ld.Axis(ld.scalar('x'), edges=[0.0, 2.0])
        components: dict[str, Sequence[str]] = {
            'signal': ['signal'],
            'signal_alias': ['signal', 'signal'],
        }

        projections = result.cross_section_projection_set(
            {'fine': fine, 'joint': [fine, wide]},
            ld.Luminosity(2.0, ld.AreaUnit.NANOBARN),
            components=components,
        )

        assert type(projections) is dict
        assert list(projections) == ['fine', 'joint']
        assert projections['fine'].shape == [2]
        assert projections['joint'].shape == [2, 1]
        assert set(projections['fine'].components) == set(projections['joint'].components)

    def test_projection_bins_match_yield_conversion(self) -> None:
        result = self.yield_context()
        fine = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])
        luminosity = ld.Luminosity(2.0, ld.AreaUnit.NANOBARN)

        yields = result.projection_set({'fine': fine, 'joint': [fine, fine]})
        sections = result.cross_section_projection_set({'fine': fine, 'joint': [fine, fine]}, luminosity)

        np.testing.assert_allclose(
            sections['fine'].observed.central,
            yields['fine'].to_cross_section(luminosity).observed.central,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            sections['joint'].observed.central,
            yields['joint'].to_cross_section(luminosity).observed.central,
            equal_nan=True,
        )

    def test_invalid_projection_mappings_fail_atomically(self) -> None:
        result = self.yield_context()
        axis = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])

        with self.assertRaisesRegex(ld.LadduError, 'at least one projection'):
            result.projection_set({})
        with self.assertRaisesRegex(ld.LadduError, 'names must not be empty'):
            result.projection_set({'': axis})
        with self.assertRaisesRegex(ld.LadduError, 'at least one axis'):
            result.projection_set({'empty': []})
        with self.assertRaisesRegex(TypeError, 'mapping'):
            result.projection_set(cast('Any', [axis]))
        with self.assertRaisesRegex(TypeError, 'Axis or a sequence'):
            result.projection_set({'bad': cast('Any', object())})


if __name__ == '__main__':
    unittest.main()
