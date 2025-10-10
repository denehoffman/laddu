import pytest

from laddu import BreitWigner, Dataset, Event, Manager, Mass, Vec3, parameter


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


def test_bw_evaluation() -> None:
    manager = Manager()
    amp = BreitWigner(
        'bw', parameter('mass'), parameter('width'), 2, Mass([2]), Mass([3]), Mass([2, 3])
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 1.4585691
    assert pytest.approx(result[0].imag) == 1.4107341


def test_bw_gradient() -> None:
    manager = Manager()
    amp = BreitWigner(
        'bw', parameter('mass'), parameter('width'), 2, Mass([2]), Mass([3]), Mass([2, 3])
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([1.5, 0.3])
    assert pytest.approx(result[0][0].real) == 1.3252039
    assert pytest.approx(result[0][0].imag) == -11.6827505
    assert pytest.approx(result[0][1].real) == -2.2688852
    assert pytest.approx(result[0][1].imag) == 2.5079719
