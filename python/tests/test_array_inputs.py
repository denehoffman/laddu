# ruff: noqa: S101

import unittest

import laddu as ld
import numpy as np


class ArrayInputTests(unittest.TestCase):
    def test_float32_arrays_are_accepted_and_promoted(self) -> None:
        edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        assert ld.Axis(ld.scalar('x'), edges=edges).edges == [0.0, 1.0, 2.0]
        assert ld.Bin(edges).edges == [0.0, 1.0, 2.0]

        histogram = ld.Histogram(
            np.array([1.0, 2.0], dtype=np.float32),
            bin_edges=edges,
        )
        assert histogram.counts == [1.0, 2.0]

        ensemble = ld.Ensemble.from_arrays(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            parameter_names=['a', 'b'],
        )
        assert ensemble.draws.dtype == np.dtype(np.float64)

    def test_dataset_stats_expose_signed_and_squared_weight_diagnostics(self) -> None:
        dataset = ld.Dataset.from_arrays(
            p4s={},
            scalars={'x': np.arange(4, dtype=np.float32)},
            weights=np.array([2.0, -1.0, 3.0, -4.0], dtype=np.float32),
        )

        stats = dataset.stats()
        expected_stats = (4, 0.0, 30.0, 0.0, 5.0, -5.0)
        assert (
            stats.events,
            stats.sum_weights,
            stats.sum_squared_weights,
            stats.effective_entries,
            stats.positive_weights,
            stats.negative_weights,
        ) == expected_stats

        zero = ld.Dataset.from_arrays(
            p4s={},
            scalars={'x': np.arange(2, dtype=np.float32)},
            weights=np.zeros(2, dtype=np.float32),
        ).stats()
        assert zero.effective_entries is None

        unit = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.arange(3, dtype=np.float64)})
        first = unit.stats()
        second = unit.stats()
        expected_unit_weight = 3.0
        assert first.sum_weights == second.sum_weights == expected_unit_weight
        assert first.sum_squared_weights == second.sum_squared_weights == expected_unit_weight
        assert first.effective_entries == second.effective_entries == expected_unit_weight
        assert first.positive_weights == second.positive_weights == expected_unit_weight
        assert first.negative_weights == second.negative_weights == 0.0

        empty = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.array([], dtype=np.float64)}).stats()
        assert empty.events == 0
        assert empty.effective_entries is None


if __name__ == '__main__':
    unittest.main()
