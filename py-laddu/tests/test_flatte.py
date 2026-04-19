import pytest
from laddu import Dataset, Event, Flatte, Mass, Vec3, parameter

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


def test_flatte_evaluation() -> None:
    amp = Flatte(
        'flatte',
        parameter('mass'),
        parameter('g_obs'),
        parameter('g_alt'),
        (Mass(['kshort1']), Mass(['kshort2'])),
        (0.1349768, 0.547862),
        Mass(['kshort1', 'kshort2']),
    )
    evaluator = amp.load(make_test_dataset())
    result = evaluator.evaluate([0.98, 0.7, 0.2])
    assert pytest.approx(result[0].real) == -0.7338320342780681
    assert pytest.approx(result[0].imag) == 0.5018145529787819


def test_flatte_gradient() -> None:
    amp = Flatte(
        'flatte',
        parameter('mass'),
        parameter('g_obs'),
        parameter('g_alt'),
        (Mass(['kshort1']), Mass(['kshort2'])),
        (0.1349768, 0.547862),
        Mass(['kshort1', 'kshort2']),
    )
    evaluator = amp.load(make_test_dataset())
    result = evaluator.evaluate_gradient([0.98, 0.7, 0.2])
    assert pytest.approx(result[0][0].real) == -0.08473788905152731
    assert pytest.approx(result[0][0].imag) == 1.6292790093139917
    assert pytest.approx(result[0][1].real) == 0.497349582793617
    assert pytest.approx(result[0][1].imag) == 0.19360065665801518
    assert pytest.approx(result[0][2].real) == 0.597447011338709
    assert pytest.approx(result[0][2].imag) == 0.23256505627570476
