from laddu import Event, Dataset, Vector3, Zlm, Angles, Manager, Polarization
import pytest


def make_test_event() -> Event:
    return Event(
        [
            Vector3(0.0, 0.0, 8.747).with_mass(0.0),
            Vector3(0.119, 0.374, 0.222).with_mass(1.007),
            Vector3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vector3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [Vector3(0.385, 0.022, 0.000)],
        0.48,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()])


def test_zlm_evaluation():
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    polarization = Polarization(0, [1])
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 0.04284127
    assert pytest.approx(result[0].imag) == -0.2385963


def test_ylm_gradient():
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    polarization = Polarization(0, [1])
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
