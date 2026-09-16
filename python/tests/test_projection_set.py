# ruff: noqa: PT027, S101

import unittest
from typing import TYPE_CHECKING, Any, cast

import laddu as ld
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

EXPECTED_TOTAL_SET_CACHE_COUNT = 2


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

    def test_projection_bins_match_dataset_histogram_assignments(self) -> None:
        cross_section = self.cross_section()
        fine = ld.Axis(ld.scalar('x'), edges=[0.0, 1.0, 2.0])

        results = cross_section.projection_set({'fine': fine, 'joint': [fine, fine]})

        expected = [1.5, 1.3265306122448979]
        np.testing.assert_allclose(results['fine'].data.central, expected)
        np.testing.assert_allclose(
            results['joint'].data.central,
            [expected[0], np.nan, np.nan, expected[1]],
            equal_nan=True,
        )

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

    def test_central_observed_total_skips_ensemble_draws(self) -> None:
        cross_section = self.cross_section()
        central = cross_section.observed_total_central()

        assert central.central == cross_section.observed_total().central
        assert len(central.draws) == 0
        assert central.source_id is None
        assert cross_section.diagnostics()['cached_integrals'] == 1

    def test_integral_retention_can_be_disabled_and_cleared(self) -> None:
        cross_section = self.cross_section()
        max_bytes = cross_section.diagnostics()['prepared_bytes']
        cross_section.configure_integral_retention(max_bytes=None)
        cross_section.clear_integral_cache()

        expected = cross_section.observed_total(tags=['signal']).central

        assert np.isfinite(expected)
        assert cross_section.diagnostics()['cached_integrals'] == 0
        cross_section.configure_integral_retention(max_bytes=max_bytes)
        np.testing.assert_allclose(
            cross_section.observed_total(tags=['signal']).central,
            expected,
        )
        assert cross_section.diagnostics()['prepared_bytes'] <= max_bytes
        cross_section.clear_integral_cache()
        assert np.isfinite(cross_section.observed_total().central)

    def test_combined_integral_limit_covers_all_members(self) -> None:
        first = self.cross_section()
        second = self.cross_section()
        first.observed_total(tags=['signal'])
        second.observed_total(tags=['signal'])
        combined = ld.CrossSection.combine([first, second])
        assert combined.diagnostics()['prepared_bytes'] == (
            first.diagnostics()['prepared_bytes'] + second.diagnostics()['prepared_bytes']
        )

        max_bytes = first.diagnostics()['prepared_bytes']
        combined.configure_integral_retention(max_bytes=max_bytes)
        combined.observed_total(tags=['signal'])

        assert combined.diagnostics()['prepared_bytes'] <= max_bytes

    def test_native_bootstrap_totals_share_preparation_under_small_budget(self) -> None:
        x = ld.scalar('x')
        model = ld.Model((ld.parameter('scale', initial=1.0) * x + 1.0).norm_sqr().tagged('signal'))

        values = np.linspace(0.1, 1.9, 20)

        def dataset(weights: np.ndarray[Any, np.dtype[np.float64]]) -> ld.Dataset:
            return ld.Dataset.from_arrays(
                p4s={},
                scalars={'x': values},
                weights=weights,
            )

        execution = ld.Execution(
            'cpu',
            threads=1,
            memory=ld.MemoryPlan(host='1 MiB'),
        )
        data = dataset(np.linspace(1.0, 2.0, 20))
        accepted = dataset(np.ones(20))
        generated = dataset(np.ones(20))
        likelihood = ld.Likelihood(
            [
                ld.NLL(
                    model,
                    data=data,
                    accepted_mc=accepted,
                    name='signal',
                )
            ],
            execution=execution,
        )
        replica_count = 3
        ensemble = likelihood.bootstrap_fit(
            replica_count,
            initial=[1.0],
            seed=42,
            terminators=[ld.ganesh.MaxSteps(1)],
        )
        cross_section = likelihood.cross_section(
            'signal',
            generated_mc=generated,
            luminosity=10.0,
            parameters=[1.0],
            ensemble=ensemble,
        )

        totals = cross_section.total_set(
            {
                'signal': ['signal'],
                'signal_alias': ['signal', 'signal'],
            }
        )
        total = totals.full
        expected_draws = []
        for index in range(replica_count):
            replica = ld.Likelihood(
                [
                    ld.NLL(
                        model,
                        data=data.bootstrap(seed=42 + index),
                        accepted_mc=accepted,
                        name='signal',
                    )
                ],
                execution=execution,
            )
            fit = replica.fit(
                initial=[1.0],
                terminators=[ld.ganesh.MaxSteps(1)],
            )
            expected_draws.append(
                replica.cross_section(
                    'signal',
                    generated_mc=generated,
                    luminosity=10.0,
                    parameters=fit.x,
                )
                .observed_total()
                .central
            )

        assert len(total.draws) == replica_count
        np.testing.assert_allclose(
            total.central,
            cross_section.observed_total_central().central,
        )
        np.testing.assert_allclose(total.draws, expected_draws)
        np.testing.assert_allclose(totals['signal'].draws, totals['signal_alias'].draws)
        assert set(totals.components) == {'signal', 'signal_alias'}
        assert total.source_id == ensemble.source_id
        assert cross_section.diagnostics()['cached_integrals'] == EXPECTED_TOTAL_SET_CACHE_COUNT


if __name__ == '__main__':
    unittest.main()
