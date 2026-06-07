import pickle
from fractions import Fraction

import laddu as ld
import numpy as np
import pytest
from laddu import math


def test_histogram_numpy_round_trip_and_pickle() -> None:
    hist = math.Histogram(np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0]))

    edges, counts = hist.to_numpy()
    np.testing.assert_allclose(edges, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(counts, [2.0, 3.0])
    np.testing.assert_allclose(hist.bin_edges, edges)
    np.testing.assert_allclose(hist.counts, counts)
    assert hist.total_weight == 5.0
    assert ld.Histogram is math.Histogram
    assert 'Histogram' in repr(hist)
    assert 'Histogram' in str(hist)

    restored = pickle.loads(pickle.dumps(hist))
    np.testing.assert_allclose(restored.bin_edges, hist.bin_edges)
    np.testing.assert_allclose(restored.counts, hist.counts)


def test_histogram_validation() -> None:
    with pytest.raises(RuntimeError, match=r'counts\.len'):
        math.Histogram([0.0, 1.0], [1.0, 2.0])

    hist = math.Histogram([0.0, 1.0, 2.0], [1.0, 1.0])
    assert hist.total_weight == 2.0


def test_clebsch_gordan() -> None:
    assert ld.clebsch_gordan is math.clebsch_gordan
    assert math.clebsch_gordan(
        Fraction(1, 2), Fraction(1, 2), 0.5, -0.5, 1, 0
    ) == pytest.approx(2.0**-0.5)
    assert (
        math.clebsch_gordan(ld.J(0), ld.M(0), ld.J(0), ld.M(0), ld.J(0), ld.M(0)) == 1.0
    )
    assert math.clebsch_gordan(Fraction(1, 2), Fraction(1, 2), 0.5, 0.5, 1, 0) == 0.0
