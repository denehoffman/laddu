import numpy as np
import pandas as pd
import polars as pl
import pytest

from laddu import Dataset, Event, Mass, Vec3

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']


def make_test_event() -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [0.38562805, 1.93592989],
        0.48,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()], p4_names=P4_NAMES, aux_names=AUX_NAMES)


def test_event_creation() -> None:
    event = make_test_event()
    assert len(event.p4s) == 4
    assert len(event.aux_values) == 2
    assert event.weight == 0.48


def test_event_p4_sum() -> None:
    event = make_test_event()
    p4_sum = event.get_p4_sum(['kshort1', 'kshort2'])
    assert p4_sum.px == event.p4s[2].px + event.p4s[3].px
    assert p4_sum.py == event.p4s[2].py + event.p4s[3].py
    assert p4_sum.pz == event.p4s[2].pz + event.p4s[3].pz
    assert p4_sum.e == event.p4s[2].e + event.p4s[3].e


def test_event_boost() -> None:
    event = make_test_event()
    rest_frame = ['proton', 'kshort1', 'kshort2']
    event_boosted = event.boost_to_rest_frame_of(rest_frame)
    p4_sum = event_boosted.get_p4_sum(rest_frame)
    assert pytest.approx(p4_sum.px) == 0.0
    assert pytest.approx(p4_sum.py) == 0.0
    assert pytest.approx(p4_sum.pz) == 0.0


def test_event_name_lookup() -> None:
    event = make_test_event()
    proton_vec = event['proton']
    assert pytest.approx(proton_vec.e) == event.p4s[1].e
    proton_optional = event.p4('proton')
    assert proton_optional is not None
    assert pytest.approx(proton_optional.e) == event.p4s[1].e
    aux_value = event.aux('pol_angle')
    assert aux_value is not None
    assert pytest.approx(aux_value) == event.aux_values[1]
    assert isinstance(event.get('pol_magnitude'), float)
    assert event.get('unknown') is None
    with pytest.raises(KeyError):
        _ = event['unknown']


def test_event_evaluate() -> None:
    event = make_test_event()
    mass = Mass(['proton'])
    assert event.evaluate(mass) == 1.007


def test_event_evaluate_without_metadata() -> None:
    event = Event(
        [Vec3(0, 0, 1).with_mass(1.0)],
        [],
        1.0,
    )
    mass = Mass(['particle'])
    with pytest.raises(ValueError):
        event.evaluate(mass)


def test_dataset_conversion() -> None:
    data = {
        'beam_px': [1.0, 2.0, 3.0, 4.0],
        'beam_py': [2.0, 3.0, 4.0, 5.0],
        'beam_pz': [3.0, 4.0, 5.0, 6.0],
        'beam_e': [4.0, 5.0, 6.0, 7.0],
        'proton_px': [5.0, 6.0, 7.0, 8.0],
        'proton_py': [6.0, 7.0, 8.0, 9.0],
        'proton_pz': [7.0, 8.0, 9.0, 10.0],
        'proton_e': [80.0, 90.0, 100.0, 110.0],
        'pol_magnitude': [0.4, 0.5, 0.6, 0.7],
        'pol_angle': [0.1, 0.2, 0.3, 0.4],
        'weight': [1.0, 1.0, 2.0, 6.6],
        'extra_column': [10, 20, 30, 40],
    }
    ds_from_dict = Dataset.from_dict(data)
    np_data = {k: np.array(v) for k, v in data.items()}
    ds_from_numpy = Dataset.from_numpy(np_data)
    ds_from_pandas = Dataset.from_pandas(pd.DataFrame(data))
    ds_from_polars = Dataset.from_polars(pl.DataFrame(data))
    assert ds_from_dict[1].p4s[0].px == 2.0
    assert pytest.approx(ds_from_numpy[0].aux_values[1]) == 0.1
    assert ds_from_pandas[2].weight == 2.0
    assert ds_from_polars[2].p4s[1].e == 100.0


def test_dataset_size_check() -> None:
    dataset = Dataset([])
    assert len(dataset) == 0
    dataset = make_test_dataset()
    assert len(dataset) == 1


def test_dataset_weights() -> None:
    second_event = Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [0.38562805, 1.93592989],
        0.52,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )
    dataset = Dataset(
        [make_test_event(), second_event],
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )
    weights = dataset.weights
    assert len(weights) == 2
    assert weights[0] == 0.48
    assert weights[1] == 0.52
    assert dataset.n_events_weighted == 1.0


def test_dataset_sum() -> None:
    dataset = make_test_dataset()
    dataset2 = Dataset(
        [
            Event(
                [
                    Vec3(0.0, 0.0, 8.747).with_mass(0.0),
                    Vec3(0.119, 0.374, 0.222).with_mass(1.007),
                    Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
                    Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
                ],
                [0.38562805, 1.93592989],
                0.52,
                p4_names=P4_NAMES,
                aux_names=AUX_NAMES,
            )
        ],
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )
    dataset_sum = dataset + dataset2
    assert dataset_sum[0].weight == dataset[0].weight
    assert dataset_sum[1].weight == dataset2[0].weight
    list_sum = sum([dataset, dataset2])
    assert list_sum != 0  # TODO: Pray the Python devs sort this out someday
    assert list_sum[0].weight == dataset_sum[0].weight
    assert list_sum[1].weight == dataset_sum[1].weight


def test_dataset_filtering() -> None:
    dataset = Dataset(
        [
            Event([Vec3(0, 0, 5).with_mass(0.0)], [], 1.0, p4_names=['p'], aux_names=[]),
            Event([Vec3(0, 0, 5).with_mass(0.5)], [], 1.0, p4_names=['p'], aux_names=[]),
            Event([Vec3(0, 0, 5).with_mass(1.1)], [], 1.0, p4_names=['p'], aux_names=[]),
        ],
        p4_names=['p'],
        aux_names=[],
    )
    mass = Mass(['p'])
    expression = (mass > 0.0) & (mass < 1.0)

    filtered = dataset.filter(expression)
    assert filtered.n_events == 1
    assert mass.value(filtered[0]) == 0.5


def test_dataset_evaluate() -> None:
    dataset = make_test_dataset()
    mass = Mass(['proton'])
    assert dataset.evaluate(mass)[0] == 1.007


def test_dataset_index() -> None:
    dataset = make_test_dataset()
    assert isinstance(dataset[0], Event)
    mass = Mass(['proton'])
    assert isinstance(dataset[mass], np.ndarray)
    proton_vec = dataset[0]['proton']
    assert pytest.approx(proton_vec.e) == dataset[0].p4s[1].e


def test_binned_dataset() -> None:
    dataset = Dataset(
        [
            Event(
                [Vec3(0.0, 0.0, 1.0).with_mass(1.0)],
                [],
                1.0,
                p4_names=['p'],
                aux_names=[],
            ),
            Event(
                [Vec3(0.0, 0.0, 2.0).with_mass(2.0)],
                [],
                2.0,
                p4_names=['p'],
                aux_names=[],
            ),
        ],
        p4_names=['p'],
        aux_names=[],
    )

    mass = Mass(['p'])
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
                make_test_event().aux_values,
                1.0,
                p4_names=P4_NAMES,
                aux_names=AUX_NAMES,
            ),
        ],
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )
    assert dataset[0].weight != dataset[1].weight

    bootstrapped = dataset.bootstrap(43)
    assert len(bootstrapped) == len(dataset)
    assert bootstrapped[0].weight == bootstrapped[1].weight

    empty_dataset = Dataset([], p4_names=[], aux_names=[])
    empty_bootstrap = empty_dataset.bootstrap(43)
    assert len(empty_bootstrap) == 0


def test_dataset_iteration() -> None:
    dataset = make_test_dataset()
    events = list(dataset)
    assert len(events) == dataset.n_events
    assert all(isinstance(event, Event) for event in events)
    assert pytest.approx(events[0]['proton'].e) == dataset[0]['proton'].e


def test_dataset_boost() -> None:
    dataset = make_test_dataset()
    rest_frame = ['proton', 'kshort1', 'kshort2']
    dataset_boosted = dataset.boost_to_rest_frame_of(rest_frame)
    p4_sum = dataset_boosted[0].get_p4_sum(rest_frame)
    assert pytest.approx(p4_sum.px) == 0.0
    assert pytest.approx(p4_sum.py) == 0.0
    assert pytest.approx(p4_sum.pz) == 0.0


def test_event_display() -> None:
    event = make_test_event()
    display_string = str(event)
    assert 'Event:' in display_string
    assert 'p4s:' in display_string
    assert 'aux:' in display_string
    assert 'aux[0]: 0.38562805' in display_string
    assert 'aux[1]: 1.93592989' in display_string
    assert 'weight:' in display_string
