import pytest
from laddu import Dataset, Event, Mass, Vec3, Voigt, parameter

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


def test_voigt_evaluation() -> None:
    amp = Voigt(
        'voigt',
        parameter('mass'),
        parameter('width'),
        parameter('sigma'),
        Mass(['kshort1', 'kshort2']),
    )
    evaluator = amp.load(make_test_dataset())
    result = evaluator.evaluate([0.98, 0.08, 0.02])
    assert pytest.approx(result[0].real) == 0.2857389147779551
    assert pytest.approx(result[0].imag) == 0.0


def test_voigt_gradient() -> None:
    amp = Voigt(
        'voigt',
        parameter('mass'),
        parameter('width'),
        parameter('sigma'),
        Mass(['kshort1', 'kshort2']),
    )
    evaluator = amp.load(make_test_dataset())
    result = evaluator.evaluate_gradient([0.98, 0.08, 0.02])
    assert pytest.approx(result[0][0].real) == 0.7225730704295464
    assert pytest.approx(result[0][0].imag) == 0.0
    assert pytest.approx(result[0][1].real) == 1.7488427782862053
    assert pytest.approx(result[0][1].imag) == 0.0
    assert pytest.approx(result[0][2].real) == 0.10952492310922711
    assert pytest.approx(result[0][2].imag) == 0.0
