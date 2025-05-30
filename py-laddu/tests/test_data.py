import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from laddu import Dataset, Event, Mass, Vec3


def make_test_event() -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [Vec3(0.385, 0.022, 0.000)],
        0.48,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()])


def test_event_creation() -> None:
    event = make_test_event()
    assert len(event.p4s) == 4
    assert len(event.aux) == 1
    assert event.weight == 0.48


def test_event_p4_sum() -> None:
    event = make_test_event()
    p4_sum = event.get_p4_sum([2, 3])
    assert p4_sum.px == event.p4s[2].px + event.p4s[3].px
    assert p4_sum.py == event.p4s[2].py + event.p4s[3].py
    assert p4_sum.pz == event.p4s[2].pz + event.p4s[3].pz
    assert p4_sum.e == event.p4s[2].e + event.p4s[3].e


def test_dataset_conversion() -> None:
    data = {
        'p4_0_Px': [1.0, 2.0, 3.0, 4.0],
        'p4_0_Py': [2.0, 3.0, 4.0, 5.0],
        'p4_0_Pz': [3.0, 4.0, 5.0, 6.0],
        'p4_0_E': [4.0, 5.0, 6.0, 7.0],
        'p4_1_Px': [5.0, 6.0, 7.0, 8.0],
        'p4_1_Py': [6.0, 7.0, 8.0, 9.0],
        'p4_1_Pz': [7.0, 8.0, 9.0, 10.0],
        'p4_1_E': [80.0, 90.0, 100.0, 110.0],
        'aux_0_x': [9.0, 10.0, 11.0, 3.3],
        'aux_0_y': [10.1, 11.0, 12.0, 4.4],
        'aux_0_z': [0.1, 0.2, 0.3, 5.5],
        'weight': [1.0, 1.0, 2.0, 6.6],
    }
    ds_from_dict = Dataset.from_dict(data)
    np_data = {k: np.array(v) for k, v in data.items()}
    ds_from_numpy = Dataset.from_numpy(np_data)
    ds_from_pandas = Dataset.from_pandas(pd.DataFrame(data))
    ds_from_polars = Dataset.from_polars(pl.DataFrame(data))
    ds_from_arrow = Dataset.from_arrow(pa.Table.from_pydict(data))
    assert ds_from_dict[1].p4s[0].px == 2.0
    assert ds_from_numpy[0].aux[0].y == 10.1
    assert ds_from_pandas[2].weight == 2.0
    assert ds_from_polars[2].p4s[1].e == 100.0
    assert ds_from_arrow[2].aux[0].z == 0.3


def test_dataset_size_check() -> None:
    dataset = Dataset([])
    assert len(dataset) == 0
    dataset = make_test_dataset()
    assert len(dataset) == 1


def test_dataset_weights() -> None:
    dataset = Dataset(
        [
            make_test_event(),
            Event(
                make_test_event().p4s,
                make_test_event().aux,
                0.52,
            ),
        ]
    )
    weights = dataset.weights
    assert len(weights) == 2
    assert weights[0] == 0.48
    assert weights[1] == 0.52
    assert dataset.n_events_weighted == 1.0


def test_dataset_sum() -> None:
    dataset = make_test_dataset()
    dataset2 = Dataset([Event(make_test_event().p4s, make_test_event().aux, 0.52)])
    dataset_sum = dataset + dataset2
    assert dataset_sum[0].weight == dataset[0].weight
    assert dataset_sum[1].weight == dataset2[0].weight
    list_sum = sum([dataset, dataset2])
    assert list_sum != 0  # TODO: Pray the Python devs sort this out someday
    assert list_sum[0].weight == dataset_sum[0].weight
    assert list_sum[1].weight == dataset_sum[1].weight


# TODO: Dataset::filter requires free-threading or some other workaround (or maybe we make a non-parallel method)


def test_binned_dataset() -> None:
    dataset = Dataset(
        [
            Event(
                [Vec3(0.0, 0.0, 1.0).with_mass(1.0)],
                [],
                1.0,
            ),
            Event(
                [Vec3(0.0, 0.0, 2.0).with_mass(2.0)],
                [],
                2.0,
            ),
        ]
    )

    mass = Mass([0])
    binned = dataset.bin_by(mass, 2, (0.0, 3.0))

    assert binned.n_bins == 2
    assert len(binned.edges) == 3
    assert binned.edges[0] == 0.0
    assert binned.edges[2] == 3.0
    assert len(binned[0]) == 1
    assert binned[0].n_events_weighted == 1.0
    assert len(binned[1]) == 1
    assert binned[1].n_events_weighted == 2.0


def test_dataset_bootstrap() -> None:
    dataset = Dataset(
        [
            make_test_event(),
            Event(
                make_test_event().p4s,
                make_test_event().aux,
                1.0,
            ),
        ]
    )
    assert dataset[0].weight != dataset[1].weight

    bootstrapped = dataset.bootstrap(43)
    assert len(bootstrapped) == len(dataset)
    assert bootstrapped[0].weight == bootstrapped[1].weight

    empty_dataset = Dataset([])
    empty_bootstrap = empty_dataset.bootstrap(43)
    assert len(empty_bootstrap) == 0


def test_dataset_weighted_bootstrap() -> None:
    dataset = Dataset(
        [
            make_test_event(),
            Event(
                make_test_event().p4s,
                make_test_event().aux,
                -1.0,
            ),
        ]
    )
    assert dataset[0].weight != dataset[1].weight

    bootstrapped = dataset.weighted_bootstrap(43)
    assert len(bootstrapped) == len(dataset)
    assert bootstrapped[0].weight == bootstrapped[1].weight

    empty_dataset = Dataset([])
    empty_bootstrap = empty_dataset.weighted_bootstrap(43)
    assert len(empty_bootstrap) == 0


def test_dataset_boost() -> None:
    dataset = make_test_dataset()
    dataset_boosted = dataset.boost_to_rest_frame_of([1, 2, 3])
    p4_sum = dataset_boosted[0].get_p4_sum([1, 2, 3])
    assert pytest.approx(p4_sum.px) == 0.0
    assert pytest.approx(p4_sum.py) == 0.0
    assert pytest.approx(p4_sum.pz) == 0.0


def test_event_display() -> None:
    event = make_test_event()
    display_string = str(event)
    assert (
        display_string
        == 'Event:\n  p4s:\n    [e = 8.74700; p = (0.00000, 0.00000, 8.74700); m = 0.00000]\n    [e = 1.10334; p = (0.11900, 0.37400, 0.22200); m = 1.00700]\n    [e = 3.13671; p = (-0.11200, 0.29300, 3.08100); m = 0.49800]\n    [e = 5.50925; p = (-0.00700, -0.66700, 5.44600); m = 0.49800]\n  eps:\n    [0.385, 0.022, 0]\n  weight:\n    0.48\n'
    )
