import numpy as np
import pytest

from laddu import Dataset, Event, Manager, Mass, Vector3, parameter
from laddu.amplitudes.piecewise import (
    PiecewiseComplexScalar,
    PiecewisePolarComplexScalar,
    PiecewiseScalar,
)


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


def test_piecewise_scalar_evaluation() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewiseScalar(
        "test_scalar",
        v,
        3,
        (0.0, 1.0),
        [parameter("test_param0"), parameter("test_param1"), parameter("test_param2")],
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([1.1, 2.2, 3.3])
    assert pytest.approx(result[0].real) == 2.2
    assert pytest.approx(result[0].imag) == 0.0


def test_piecewise_scalar_gradient() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewiseScalar(
        "test_scalar",
        v,
        3,
        (0.0, 1.0),
        [parameter("test_param0"), parameter("test_param1"), parameter("test_param2")],
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = aid.norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([1.0, 2.0, 3.0])
    assert pytest.approx(result[0][0].real) == 0.0
    assert pytest.approx(result[0][0].imag) == 0.0
    assert pytest.approx(result[0][1].real) == 4.0
    assert pytest.approx(result[0][1].imag) == 0.0
    assert pytest.approx(result[0][2].real) == 0.0
    assert pytest.approx(result[0][2].imag) == 0.0


def test_piecewise_complex_scalar_evaluation() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewiseComplexScalar(
        "test_complex",
        v,
        3,
        (0.0, 1.0),
        [
            (parameter("re_param0"), parameter("im_param0")),
            (parameter("re_param1"), parameter("im_param1")),
            (parameter("re_param2"), parameter("im_param2")),
        ],
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([1.1, 1.2, 2.1, 2.2, 3.1, 3.2])
    assert pytest.approx(result[0].real) == 2.1
    assert pytest.approx(result[0].imag) == 2.2


def test_piecewise_complex_scalar_gradient() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewiseComplexScalar(
        "test_complex",
        v,
        3,
        (0.0, 1.0),
        [
            (parameter("re_param0"), parameter("im_param0")),
            (parameter("re_param1"), parameter("im_param1")),
            (parameter("re_param2"), parameter("im_param2")),
        ],
    )

    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = aid.norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([1.1, 1.2, 2.1, 2.2, 3.1, 3.2])
    assert pytest.approx(result[0][0].real) == 0.0
    assert pytest.approx(result[0][0].imag) == 0.0
    assert pytest.approx(result[0][1].real) == 0.0
    assert pytest.approx(result[0][1].imag) == 0.0
    assert pytest.approx(result[0][2].real) == 4.2
    assert pytest.approx(result[0][2].imag) == 0.0
    assert pytest.approx(result[0][3].real) == 4.4
    assert pytest.approx(result[0][3].imag) == 0.0
    assert pytest.approx(result[0][4].real) == 0.0
    assert pytest.approx(result[0][4].imag) == 0.0
    assert pytest.approx(result[0][5].real) == 0.0
    assert pytest.approx(result[0][5].imag) == 0.0


def test_piecewise_polar_complex_scalar_evaluation() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewisePolarComplexScalar(
        "test_polar",
        v,
        3,
        (0.0, 1.0),
        [
            (parameter("r_param0"), parameter("theta_param0")),
            (parameter("r_param1"), parameter("theta_param1")),
            (parameter("r_param2"), parameter("theta_param2")),
        ],
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    result = evaluator.evaluate([1.1 * r, 1.2 * theta, 2.1 * r, 2.2 * theta, 3.1 * r, 3.2 * theta])
    assert pytest.approx(result[0].real) == 2.1 * r * np.cos(2.2 * theta)
    assert pytest.approx(result[0].imag) == 2.1 * r * np.sin(2.2 * theta)


def test_piecewise_polar_complex_scalar_gradient() -> None:
    manager = Manager()
    v = Mass([2])
    amp = PiecewisePolarComplexScalar(
        "test_polar",
        v,
        3,
        (0.0, 1.0),
        [
            (parameter("r_param0"), parameter("theta_param0")),
            (parameter("r_param1"), parameter("theta_param1")),
            (parameter("r_param2"), parameter("theta_param2")),
        ],
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    result = evaluator.evaluate_gradient([1.1 * r, 1.2 * theta, 2.1 * r, 2.2 * theta, 3.1 * r, 3.2 * theta])
    assert pytest.approx(result[0][0].real) == 0.0
    assert pytest.approx(result[0][0].imag) == 0.0
    assert pytest.approx(result[0][1].real) == 0.0
    assert pytest.approx(result[0][1].imag) == 0.0
    assert pytest.approx(result[0][2].real) == np.cos(2.2 * theta)
    assert pytest.approx(result[0][2].imag) == np.sin(2.2 * theta)
    assert pytest.approx(result[0][3].real) == -2.1 * r * np.sin(2.2 * theta)
    assert pytest.approx(result[0][3].imag) == 2.1 * r * np.cos(2.2 * theta)
    assert pytest.approx(result[0][4].real) == 0.0
    assert pytest.approx(result[0][4].imag) == 0.0
    assert pytest.approx(result[0][5].real) == 0.0
    assert pytest.approx(result[0][5].imag) == 0.0
