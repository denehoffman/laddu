from laddu import (
    Manager,
    Scalar,
    ComplexScalar,
    PolarComplexScalar,
    Event,
    Vector3,
    Dataset,
    parameter,
)
import numpy as np
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


def test_scalar_creation_and_evaluation():
    manager = Manager()
    amp = Scalar('test_scalar', parameter('test_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([2.5])
    assert result[0].real == 2.5
    assert result[0].imag == 0.0


def test_scalar_gradient():
    manager = Manager()
    amp = Scalar('test_scalar', parameter('test_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = aid.norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)
    gradient = evaluator.evaluate_gradient([2.0])
    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0


def test_complex_scalar_creation_and_evaluation():
    manager = Manager()
    amp = ComplexScalar('test_complex', parameter('re_param'), parameter('im_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([1.5, 2.5])
    assert result[0].real == 1.5
    assert result[0].imag == 2.5


def test_complex_scalar_gradient():
    manager = Manager()
    amp = ComplexScalar('test_complex', parameter('re_param'), parameter('im_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = aid.norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)
    gradient = evaluator.evaluate_gradient([3.0, 4.0])
    assert gradient[0][0].real == 6.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 8.0
    assert gradient[0][1].imag == 0.0


def test_polar_complex_scalar_creation_and_evaluation():
    manager = Manager()
    amp = PolarComplexScalar('test_polar', parameter('r_param'), parameter('theta_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    result = evaluator.evaluate([r, theta])
    assert pytest.approx(result[0].real) == r * np.cos(theta)
    assert pytest.approx(result[0].imag) == r * np.sin(theta)


def test_polar_complex_scalar_gradient():
    manager = Manager()
    amp = PolarComplexScalar('test_polar', parameter('r_param'), parameter('theta_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    gradient = evaluator.evaluate_gradient([r, theta])
    assert pytest.approx(gradient[0][0].real) == np.cos(theta)
    assert pytest.approx(gradient[0][0].imag) == np.sin(theta)
    assert pytest.approx(gradient[0][1].real) == -r * np.sin(theta)
    assert pytest.approx(gradient[0][1].imag) == r * np.cos(theta)
