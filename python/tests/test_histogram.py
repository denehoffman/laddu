# ruff: noqa: S101

import unittest

import laddu as ld
import numpy as np
import pytest

EXPECTED_OVERFLOW = 2.0


class HistogramTests(unittest.TestCase):
    def test_dataset_histogram_evaluates_observable_and_weight_expressions(self) -> None:
        dataset = ld.Dataset.from_arrays(
            p4s={},
            scalars={'x': [-1.0, 0.0, 1.0, 2.0]},
            weights=[0.5, 1.0, 1.5, 2.0],
        )

        weighted = dataset.histogram(
            ld.scalar('x'),
            bin_edges=[-2.0, 0.0, 2.0],
            weight=ld.scalar('x') + 2.0,
        )
        custom_only = dataset.histogram(
            ld.scalar('x'),
            bin_edges=[-2.0, 0.0, 2.0],
            event_weights=False,
            weight=ld.scalar('x') + 2.0,
        )

        assert weighted.counts == [0.5, 6.5]
        assert weighted.squared_weight_constituents() == ([0.25, 24.25], 0.0, 64.0)
        assert custom_only.counts == [1.0, 5.0]
        assert custom_only.squared_weight_constituents() == ([1.0, 13.0], 0.0, 16.0)

    def test_dataset_histogram_matches_streaming_batches_and_selected_views(self) -> None:
        def batches(**_plan: int | None):
            yield {
                'p4s': {},
                'scalars': {'x': np.array([-1.0, 0.0])},
                'weights': np.array([0.5, 1.0]),
            }
            yield {
                'p4s': {},
                'scalars': {'x': np.array([1.0, 2.0])},
                'weights': np.array([1.5, 2.0]),
            }

        streamed = ld.Dataset.from_batches(
            batches,
            schema={'p4s': [], 'scalars': ['x'], 'weights': True},
            cache='streaming',
        )
        selected = streamed.select(ld.scalar('x') >= 0.0)
        histogram = selected.histogram(ld.scalar('x'), bin_edges=[0.0, 1.0, 2.0])

        assert histogram.counts == [1.0, 1.5]
        assert histogram.overflow == EXPECTED_OVERFLOW
        assert histogram.squared_weight_constituents() == ([1.0, 2.25], 0.0, 4.0)

    def test_dataset_histogram_rejects_invalid_edges_values_and_weights(self) -> None:
        dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.0]}, weights=[1.0])
        with pytest.raises(ld.LadduError, match='strictly increasing'):
            dataset.histogram(ld.scalar('x'), bin_edges=[0.0, 0.0])

        nonfinite = ld.Dataset.from_arrays(p4s={}, scalars={'x': [np.nan]}, weights=[1.0])
        with pytest.raises(ld.LadduError, match='expected finite'):
            nonfinite.histogram(ld.scalar('x'), bin_edges=[0.0, 1.0])

        invalid_weight = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.0], 'w': [np.nan]}, weights=[1.0])
        with pytest.raises(ld.LadduError, match='expected finite'):
            invalid_weight.histogram(
                ld.scalar('x'),
                bin_edges=[-1.0, 1.0],
                weight=ld.scalar('w'),
            )

    def test_disjoint_weighted_histograms_merge_empirical_uncertainties(self) -> None:
        left = ld.Histogram.from_values([-0.1, 0.25], bins=2, limits=(0.0, 1.0), weights=[-2.0, 3.0])
        right = ld.Histogram.from_values([0.25, 0.75, 1.0], bins=2, limits=(0.0, 1.0), weights=[-4.0, 5.0, 6.0])

        left.merge(right)

        sumw2, underflow_sumw2, overflow_sumw2 = left.squared_weight_constituents()
        assert left.counts == [-1.0, 5.0]
        assert sumw2 == [25.0, 25.0]
        assert left.errors == [5.0, 5.0]
        expected_flow = (-2.0, 4.0, 6.0, 36.0)
        assert (left.underflow, underflow_sumw2, left.overflow, overflow_sumw2) == expected_flow

    def test_incompatible_merge_is_atomic(self) -> None:
        histogram = ld.Histogram.from_values([0.25], bins=2, limits=(0.0, 1.0))
        before = histogram.to_json()
        incompatible = ld.Histogram([0.0, 0.0], bin_edges=[0.0, 0.25, 1.0])

        with pytest.raises(ld.LadduError, match='identical bin edges'):
            histogram.merge(incompatible)

        assert histogram.to_json() == before

        empirical = ld.Histogram.from_values([], bins=2, limits=(0.0, 1.0))
        manual = ld.Histogram([0.0, 0.0], bin_edges=[0.0, 0.5, 1.0])
        with pytest.raises(ld.LadduError, match='fill and uncertainty policies'):
            empirical.merge(manual)


if __name__ == '__main__':
    unittest.main()
