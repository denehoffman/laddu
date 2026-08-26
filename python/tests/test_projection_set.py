# ruff: noqa: PT027, S101

import unittest
from typing import TYPE_CHECKING, Any, cast

import laddu as ld
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class ProjectionSetTests(unittest.TestCase):
    @staticmethod
    def cross_section() -> ld.CrossSection:
        x = ld.scalar('x')
        signal = (ld.parameter('signal', initial=1.5) * x).tagged('signal')
        background = ld.parameter('background', initial=0.75).tagged('background')
        model = ld.Model((signal + background).norm_sqr())

        def dataset(values: list[float], weights: list[float]) -> ld.Dataset:
            return ld.Dataset.from_arrays(
                p4s={},
                scalars={'x': np.asarray(values)},
                weights=np.asarray(weights),
            )

        data = dataset([0.25, 0.75, 1.25], [1.0, 2.0, 1.0])
        accepted = dataset([0.25, 0.75, 1.25], [1.0, 1.0, 1.0])
        generated = dataset([0.25, 0.75, 1.25, 1.75], [1.0, 1.0, 1.0, 1.0])
        likelihood = ld.Likelihood([ld.NLL(model, data=data, accepted_mc=accepted, name='signal')])
        return likelihood.cross_section(
            'signal',
            generated_mc=generated,
            luminosity=2.0,
            parameters=[1.5, 0.75],
        )

    def test_mapping_order_axis_shapes_and_global_components_are_preserved(self) -> None:
        cross_section = self.cross_section()
        fine = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])
        wide = ld.Axis(ld.scalar('x'), edges=[0.0, 2.0])
        components: dict[str, Sequence[str]] = {
            'signal': ['signal'],
            'signal_alias': ['signal', 'signal'],
        }

        results = cross_section.projection_set(
            {'fine': fine, 'joint': [fine, wide]},
            components=components,
        )

        assert type(results) is dict
        assert list(results) == ['fine', 'joint']
        assert results['fine'].shape == [2]
        assert results['joint'].shape == [2, 1]
        assert set(results['fine'].components) == set(results['joint'].components)

    def test_invalid_projection_mappings_fail_without_partial_results(self) -> None:
        cross_section = self.cross_section()
        axis = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])

        with self.assertRaisesRegex(ld.LadduError, 'at least one projection'):
            cross_section.projection_set({})
        with self.assertRaisesRegex(ld.LadduError, 'names must not be empty'):
            cross_section.projection_set({'': axis})
        with self.assertRaisesRegex(ld.LadduError, 'at least one axis'):
            cross_section.projection_set({'empty': []})
        with self.assertRaisesRegex(TypeError, 'mapping'):
            cross_section.projection_set(cast('Any', [axis]))
        with self.assertRaisesRegex(TypeError, 'Axis or a sequence'):
            cross_section.projection_set({'bad': cast('Any', object())})


if __name__ == '__main__':
    unittest.main()
