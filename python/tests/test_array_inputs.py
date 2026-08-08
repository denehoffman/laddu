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


if __name__ == '__main__':
    unittest.main()
