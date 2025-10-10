import pytest

from laddu import Angles, Dataset, Event, Manager, Vec3, Ylm


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


def test_ylm_evaluation() -> None:
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    amp = Ylm('ylm', 1, 1, angles)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 0.2713394
    assert pytest.approx(result[0].imag) == 0.1426897


def test_ylm_gradient() -> None:
    manager = Manager()
    angles = Angles(0, [1], [2], [2, 3], 'Helicity')
    amp = Ylm('ylm', 1, 1, angles)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
