from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pytest

from laddu import Dataset, Event, Mass, Vec3, Vec4

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data_files'
DATA_F32_PARQUET = TEST_DATA_DIR / 'data_f32.parquet'
DATA_F64_PARQUET = TEST_DATA_DIR / 'data_f64.parquet'
DATA_F32_ROOT = TEST_DATA_DIR / 'data_f32.root'
DATA_F64_ROOT = TEST_DATA_DIR / 'data_f64.root'
DATA_AMPTOOLS_ROOT = TEST_DATA_DIR / 'data_amptools.root'
DATA_AMPTOOLS_POL_ROOT = TEST_DATA_DIR / 'data_amptools_pol.root'


def _assert_vec4_close(vec_left: Vec4 | None, vec_right: Vec4 | None) -> None:
    assert vec_left is not None
    assert vec_right is not None
    assert pytest.approx(vec_left.px) == vec_right.px
    assert pytest.approx(vec_left.py) == vec_right.py
    assert pytest.approx(vec_left.pz) == vec_right.pz
    assert pytest.approx(vec_left.e) == vec_right.e


def _assert_events_close(
    event_left: Event,
    event_right: Event,
    p4_names: list[str],
    aux_names: list[str],
) -> None:
    for name in p4_names:
        vec_left = event_left.p4(name)
        vec_right = event_right.p4(name)
        _assert_vec4_close(vec_left, vec_right)
    for name in aux_names:
        aux_left = event_left.aux[name]
        aux_right = event_right.aux[name]
        assert pytest.approx(aux_left) == aux_right
    assert pytest.approx(event_left.weight) == event_right.weight


def _shared_names(left: list[str], right: list[str]) -> list[str]:
    left_set = set(left)
    right_set = set(right)
    assert left_set == right_set
    return sorted(left_set)


def _assert_datasets_close(dataset_left: Dataset, dataset_right: Dataset) -> None:
    assert dataset_left.n_events == dataset_right.n_events
    shared_p4 = _shared_names(dataset_left.p4_names, dataset_right.p4_names)
    shared_aux = _shared_names(dataset_left.aux_names, dataset_right.aux_names)
    for idx in range(dataset_left.n_events):
        _assert_events_close(dataset_left[idx], dataset_right[idx], shared_p4, shared_aux)


def make_test_event() -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [0.38562805, 0.05708078],
        0.48,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()], p4_names=P4_NAMES, aux_names=AUX_NAMES)


def test_event_creation() -> None:
    event = make_test_event()
    assert len(event.p4s) == 4
    assert len(event.aux) == 2
    assert event.weight == 0.48


def test_event_p4_sum() -> None:
    event = make_test_event()
    p4_sum = event.get_p4_sum(['kshort1', 'kshort2'])
    first = event.p4s['kshort1']
    second = event.p4s['kshort2']
    assert p4_sum.px == first.px + second.px
    assert p4_sum.py == first.py + second.py
    assert p4_sum.pz == first.pz + second.pz
    assert p4_sum.e == first.e + second.e


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
    proton_vec = event.p4s['proton']
    assert isinstance(proton_vec, Vec4)
    assert pytest.approx(proton_vec.e) == event.p4s['proton'].e
    proton_optional = event.p4('proton')
    assert proton_optional is not None
    assert pytest.approx(proton_optional.e) == event.p4s['proton'].e
    aux_value = event.aux['pol_angle']
    assert pytest.approx(aux_value) == event.aux['pol_angle']
    assert isinstance(event.aux.get('pol_magnitude'), float)
    assert event.aux.get('unknown') is None
    with pytest.raises(KeyError):
        _ = event.p4s['unknown']


def test_event_alias_lookup() -> None:
    event = Event(
        [Vec3(0.0, 0.0, 1.0).with_mass(0.0), Vec3(0.1, 0.0, 2.0).with_mass(0.5)],
        [],
        1.0,
        p4_names=['beam', 'kshort'],
        aux_names=[],
        aliases={'resonance': ['beam', 'kshort'], 'projectile': 'beam'},
    )

    alias_vec = event.p4('resonance')
    expected = event.get_p4_sum(['beam', 'kshort'])
    _assert_vec4_close(alias_vec, expected)

    single_alias = event.p4('projectile')
    _assert_vec4_close(single_alias, event.p4s['beam'])


def test_event_alias_requires_p4_names() -> None:
    with pytest.raises(ValueError, match='p4_names'):
        Event([Vec3(0.0, 0.0, 1.0).with_mass(0.0)], [], 1.0, aliases={'p': 'beam'})


def test_dataset_alias_overrides_event_metadata() -> None:
    base_event = Event(
        [Vec3(0.0, 0.0, 1.0).with_mass(0.0), Vec3(0.1, 0.0, 2.0).with_mass(0.5)],
        [],
        1.0,
        p4_names=['beam', 'kshort'],
        aliases={'resonance': 'beam'},
    )

    dataset = Dataset(
        [base_event],
        aliases={'resonance': ['beam', 'kshort']},
    )
    expected = dataset[0].get_p4_sum(['beam', 'kshort'])
    alias_vec = dataset[0].p4('resonance')
    _assert_vec4_close(alias_vec, expected)


def test_dataset_constructor_metadata_precedence() -> None:
    event = Event(
        [Vec3(0.0, 0.0, 1.0).with_mass(0.0), Vec3(0.2, 0.1, 1.5).with_mass(0.3)],
        [],
        1.0,
        p4_names=['legacy_beam', 'legacy_kshort'],
        aliases={'pair': 'legacy_beam'},
    )

    dataset = Dataset(
        [event],
        p4_names=['beam', 'kshort'],
        aliases={'pair': ['beam', 'kshort']},
    )
    assert dataset.p4_names == ['beam', 'kshort']
    alias_vec = dataset[0].p4('pair')
    expected = dataset[0].get_p4_sum(['beam', 'kshort'])
    _assert_vec4_close(alias_vec, expected)
    with pytest.raises(KeyError):
        _ = dataset[0].p4s['legacy_beam']


def test_dataset_alias_requires_metadata() -> None:
    event = Event([Vec3(0.0, 0.0, 1.0).with_mass(0.0)], [], 1.0)

    with pytest.raises(ValueError, match='aliases'):
        Dataset([event], aliases={'p': 'beam'})


def test_event_evaluate() -> None:
    event = make_test_event()
    mass = Mass(['proton'])
    assert event.evaluate(mass) == 1.007


def test_mass_accepts_string_input() -> None:
    dataset = make_test_dataset()
    mass_list = Mass(['proton'])
    mass_str = Mass('proton')
    assert mass_list.value(dataset[0]) == mass_str.value(dataset[0])


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
    assert ds_from_dict[1].p4s['beam'].px == 2.0
    assert pytest.approx(ds_from_numpy[0].aux['pol_angle']) == 0.1
    assert ds_from_pandas[2].weight == 2.0
    assert ds_from_polars[2].p4s['proton'].e == 100.0


def test_dataset_rejects_duplicate_p4_names() -> None:
    event = make_test_event()
    with pytest.raises(ValueError):
        Dataset([event], p4_names=['beam', 'beam'], aux_names=AUX_NAMES)


def test_dataset_rejects_duplicate_aux_names() -> None:
    event = make_test_event()
    with pytest.raises(ValueError):
        Dataset([event], p4_names=P4_NAMES, aux_names=['aux', 'aux'])


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
        [0.38562805, 0.05708078],
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


def test_dataset_from_dict_missing_components() -> None:
    data = {'beam_px': [1.0]}
    with pytest.raises(KeyError):
        Dataset.from_dict(data)


def test_dataset_from_dict_requires_p4_columns() -> None:
    with pytest.raises(ValueError):
        Dataset.from_dict({'weight': [1.0]})


def test_from_dict_accepts_names_and_aliases() -> None:
    data = {
        'beam_px': [0.0, 1.0],
        'beam_py': [0.0, 0.0],
        'beam_pz': [1.0, 2.0],
        'beam_e': [1.5, 2.5],
        'aux': [3.0, 4.0],
    }

    dataset = Dataset.from_dict(
        data, p4s=['beam'], aux=['aux'], aliases={'primary': 'beam'}
    )

    assert dataset.p4_names == ['beam']
    assert dataset.aux_names == ['aux']
    beam = dataset[0].p4('beam')
    alias = dataset[0].p4('primary')
    _assert_vec4_close(alias, beam)


def test_from_numpy_propagates_aliases_and_names() -> None:
    data = {
        'beam_px': np.array([0.0]),
        'beam_py': np.array([0.0]),
        'beam_pz': np.array([1.0]),
        'beam_e': np.array([1.5]),
        'aux': np.array([2.0]),
    }

    dataset = Dataset.from_numpy(
        data,
        p4s=['beam'],
        aux=['aux'],
        aliases={'alias_beam': ['beam']},
    )

    assert dataset.p4_names == ['beam']
    assert dataset.aux_names == ['aux']
    _assert_vec4_close(dataset[0].p4('alias_beam'), dataset[0].p4('beam'))


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
                [0.38562805, 0.05708078],
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
    proton_vec = dataset[0].p4s['proton']
    assert isinstance(proton_vec, Vec4)
    assert pytest.approx(proton_vec.e) == dataset[0].p4s['proton'].e


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
                list(make_test_event().p4s.values()),
                list(make_test_event().aux.values()),
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
    proton_vec_from_events = events[0].p4s['proton']
    assert isinstance(proton_vec_from_events, Vec4)
    proton_vec_from_dataset = dataset[0].p4s['proton']
    assert isinstance(proton_vec_from_dataset, Vec4)
    assert pytest.approx(proton_vec_from_events.e) == proton_vec_from_dataset.e


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
    assert 'aux[1]: 0.05708078' in display_string
    assert 'weight:' in display_string


def test_dataset_from_parquet_auto_vs_named() -> None:
    auto = Dataset.from_parquet(DATA_F32_PARQUET)
    assert auto.p4_names == P4_NAMES
    assert auto.aux_names == AUX_NAMES

    explicit = Dataset.from_parquet(
        DATA_F32_PARQUET,
        p4s=P4_NAMES,
        aux=AUX_NAMES,
    )
    _assert_datasets_close(auto, explicit)


def test_dataset_from_root_matches_parquet() -> None:
    parquet = Dataset.from_parquet(DATA_F32_PARQUET)
    root_auto = Dataset.from_root(DATA_F32_ROOT)
    assert root_auto.p4_names == P4_NAMES
    assert root_auto.aux_names == AUX_NAMES
    _assert_datasets_close(root_auto, parquet)

    root_named = Dataset.from_root(
        DATA_F32_ROOT,
        p4s=P4_NAMES,
        aux=AUX_NAMES,
    )
    _assert_datasets_close(root_auto, root_named)


def test_dataset_from_parquet_with_aliases() -> None:
    dataset = Dataset.from_parquet(
        DATA_F32_PARQUET,
        aliases={'resonance': ['kshort1', 'kshort2']},
    )
    alias_vec = dataset[0].p4('resonance')
    expected = dataset[0].get_p4_sum(['kshort1', 'kshort2'])
    _assert_vec4_close(alias_vec, expected)


def test_mass_uses_alias_string() -> None:
    dataset = Dataset.from_parquet(
        DATA_F32_PARQUET,
        aliases={'resonance': ['kshort1', 'kshort2']},
    )
    mass_alias = Mass('resonance')
    mass_direct = Mass(['kshort1', 'kshort2'])
    alias_values = mass_alias.value_on(dataset)
    direct_values = mass_direct.value_on(dataset)
    np.testing.assert_allclose(alias_values, direct_values)


def test_dataset_from_amptools_matches_native_vectors() -> None:
    native = Dataset.from_parquet(DATA_F32_PARQUET)
    amptools = Dataset.from_amptools(DATA_AMPTOOLS_ROOT)
    assert amptools.p4_names == [
        'beam',
        'final_state_0',
        'final_state_1',
        'final_state_2',
    ]
    assert amptools.aux_names == []
    assert amptools.n_events == native.n_events
    for idx in range(native.n_events):
        amp_event = amptools[idx]
        native_event = native[idx]
        amp_vectors = list(amp_event.p4s.values())
        native_vectors = list(native_event.p4s.values())
        for amp_vec, native_vec in zip(amp_vectors, native_vectors):
            _assert_vec4_close(amp_vec, native_vec)
        assert pytest.approx(amp_event.weight) == native_event.weight


def test_dataset_from_amptools_pol_in_beam_columns() -> None:
    native = Dataset.from_parquet(DATA_F32_PARQUET)
    amptools = Dataset.from_amptools(DATA_AMPTOOLS_POL_ROOT, pol_in_beam=True)
    assert amptools.aux_names == AUX_NAMES
    for idx in range(native.n_events):
        amp_event = amptools[idx]
        native_event = native[idx]
        amp_mag = amp_event.aux['pol_magnitude']
        native_mag = native_event.aux['pol_magnitude']
        assert pytest.approx(amp_mag) == native_mag
        amp_angle = amp_event.aux['pol_angle']
        native_angle = native_event.aux['pol_angle']
        assert pytest.approx(amp_angle) == native_angle
        assert pytest.approx(amp_event.p4s['beam'].px) == 0.0
        assert pytest.approx(amp_event.p4s['beam'].py) == 0.0


def test_dataset_from_amptools_custom_polarization_names() -> None:
    dataset = Dataset.from_amptools(
        DATA_AMPTOOLS_ROOT,
        pol_angle=30.0,
        pol_magnitude=0.75,
        pol_angle_name='phi_pol',
        pol_magnitude_name='mag_pol',
    )
    assert dataset.aux_names == ['mag_pol', 'phi_pol']
    mag = dataset[0].aux['mag_pol']
    phi = dataset[0].aux['phi_pol']
    assert mag is not None
    assert phi is not None
    assert pytest.approx(mag) == 0.75
    assert pytest.approx(phi) == np.deg2rad(30.0)


def test_dataset_from_amptools_rejects_unknown_option() -> None:
    with pytest.raises(TypeError):
        Dataset.from_amptools(DATA_AMPTOOLS_ROOT, unknown_option=True)  # type: ignore[arg-type]


def test_dataset_from_parquet_f64_matches_f32() -> None:
    f32 = Dataset.from_parquet(DATA_F32_PARQUET)
    f64 = Dataset.from_parquet(DATA_F64_PARQUET)
    assert f64.p4_names == P4_NAMES
    assert f64.aux_names == AUX_NAMES
    _assert_datasets_close(f32, f64)


def test_dataset_from_root_f64_matches_parquet() -> None:
    parquet = Dataset.from_parquet(DATA_F32_PARQUET)
    root_f64 = Dataset.from_root(DATA_F64_ROOT)
    assert root_f64.p4_names == P4_NAMES
    assert root_f64.aux_names == AUX_NAMES
    _assert_datasets_close(root_f64, parquet)


def test_dataset_to_numpy_precision() -> None:
    dataset = make_test_dataset()
    arrays64 = dataset.to_numpy()
    assert arrays64['beam_px'].dtype == np.float64
    assert arrays64['weight'].dtype == np.float64

    arrays32 = dataset.to_numpy(precision='f32')
    assert arrays32['beam_px'].dtype == np.float32
    assert arrays32['weight'].dtype == np.float32


def test_dataset_parquet_roundtrip_tempfile() -> None:
    dataset = Dataset.from_parquet(DATA_F32_PARQUET)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'roundtrip.parquet'
        dataset.to_parquet(path)
        reopened = Dataset.from_parquet(path)
    _assert_datasets_close(dataset, reopened)


def test_dataset_root_roundtrip_tempfile() -> None:
    dataset = Dataset.from_parquet(DATA_F32_PARQUET)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'roundtrip.root'
        dataset.to_root(path)
        reopened = Dataset.from_root(path)
    _assert_datasets_close(dataset, reopened)
