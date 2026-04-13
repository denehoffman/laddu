import pytest
from laddu import Dataset, Event, Mass, Vec3
from laddu.amplitudes.lookup_table import LookupTable

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']
AUX_VALUES = [0.38562805, 0.05708078]


def make_test_event() -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        AUX_VALUES.copy(),
        0.48,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()], p4_names=P4_NAMES, aux_names=AUX_NAMES)


def test_lookup_table_1d_nearest() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 0.25, 0.75, 1.0]],
        [1.0 + 0.0j, 2.0 + 3.0j, 4.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 3.0


def test_lookup_table_2d_row_major() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1']), Mass(['kshort1', 'kshort2'])],
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_zero_boundary() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 0.5, 1.0]],
        [1.0 + 0.0j, 2.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 0.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_clamp_boundary() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 0.5, 1.0]],
        [1.0 + 0.0j, 2.0 + 0.0j],
        boundary_mode='clamp',
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='lookup-table values'):
        LookupTable(
            'lookup',
            [Mass(['kshort1'])],
            [[0.0, 0.5, 1.0]],
            [1.0 + 0.0j],
        )
