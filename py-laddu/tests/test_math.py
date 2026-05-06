import pickle

import laddu as ld
import numpy as np
import pytest
from laddu import generation, math


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


def test_histogram_validation_and_generation_constructors() -> None:
    with pytest.raises(RuntimeError, match=r'counts\.len'):
        math.Histogram([0.0, 1.0], [1.0, 2.0])

    hist = math.Histogram([0.0, 1.0, 2.0], [1.0, 1.0])
    assert generation.Distribution.histogram(hist) is not None
    assert generation.MandelstamTDistribution.histogram(hist) is not None
    assert generation.InitialGenerator.beam_with_energy_histogram(0.0, hist) is not None
