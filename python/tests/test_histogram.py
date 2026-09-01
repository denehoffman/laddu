# ruff: noqa: S101

import unittest

import laddu as ld
import pytest


class HistogramTests(unittest.TestCase):
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
