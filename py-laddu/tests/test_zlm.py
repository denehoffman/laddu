import pytest

from laddu import Angles, Dataset, Event, Manager, Polarization, Vec3, Zlm
from laddu.amplitudes.zlm import PolPhase


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


def test_zlm_evaluation() -> None:
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    polarization = Polarization(0, [1], 0)
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 0.04284127
    assert pytest.approx(result[0].imag) == -0.2385963


def test_zlm_gradient() -> None:
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    polarization = Polarization(0, [1], 0)
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters


def test_polphase_evaluation() -> None:
    manager = Manager()
    polarization = Polarization(0, [1], 0)
    amp = PolPhase('polphase', polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == -0.28729145
    assert pytest.approx(result[0].imag) == -0.25724039


def test_polphase_gradient() -> None:
    manager = Manager()
    polarization = Polarization(0, [1], 0)
    amp = PolPhase('polphase', polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
