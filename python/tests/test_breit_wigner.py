from laddu import Event, Dataset, Vector3, BreitWigner, Mass, Manager, parameter
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


def test_bw_evaluation():
    manager = Manager()
    amp = BreitWigner(
        'bw', parameter('mass'), parameter('width'), 2, Mass([2]), Mass([3]), Mass([2, 3])
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 1.4585691
    assert pytest.approx(result[0].imag) == 1.4107341


def test_bw_gradient():
    manager = Manager()
    amp = BreitWigner(
        'bw', parameter('mass'), parameter('width'), 2, Mass([2]), Mass([3]), Mass([2, 3])
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([1.5, 0.3])
    assert pytest.approx(result[0][0].real) == 1.3252039
    assert pytest.approx(result[0][0].imag) == -11.6827505
    assert pytest.approx(result[0][1].real) == -2.2688852
    assert pytest.approx(result[0][1].imag) == 2.5079719
