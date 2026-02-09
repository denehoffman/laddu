import numpy as np
import pytest
from laddu import (
    ComplexScalar,
    Dataset,
    Event,
    PolarComplexScalar,
    Scalar,
    Vec3,
    parameter,
)

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


def test_scalar_creation_and_evaluation() -> None:
    amp = Scalar('test_scalar', parameter('test_param'))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([2.5])
    assert result[0].real == 2.5
    assert result[0].imag == 0.0


def test_scalar_gradient() -> None:
    amp = Scalar('test_scalar', parameter('test_param'))
    dataset = make_test_dataset()
    expr = amp.norm_sqr()
    evaluator = expr.load(dataset)
    gradient = evaluator.evaluate_gradient([2.0])
    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0


def test_complex_scalar_creation_and_evaluation() -> None:
    amp = ComplexScalar('test_complex', parameter('re_param'), parameter('im_param'))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([1.5, 2.5])
    assert result[0].real == 1.5
    assert result[0].imag == 2.5


def test_complex_scalar_gradient() -> None:
    amp = ComplexScalar('test_complex', parameter('re_param'), parameter('im_param'))
    dataset = make_test_dataset()
    expr = amp.norm_sqr()
    evaluator = expr.load(dataset)
    gradient = evaluator.evaluate_gradient([3.0, 4.0])
    assert gradient[0][0].real == 6.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 8.0
    assert gradient[0][1].imag == 0.0


def test_polar_complex_scalar_creation_and_evaluation() -> None:
    amp = PolarComplexScalar('test_polar', parameter('r_param'), parameter('theta_param'))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    result = evaluator.evaluate([r, theta])
    assert pytest.approx(result[0].real) == r * np.cos(theta)
    assert pytest.approx(result[0].imag) == r * np.sin(theta)


def test_polar_complex_scalar_gradient() -> None:
    amp = PolarComplexScalar('test_polar', parameter('r_param'), parameter('theta_param'))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    r = 2.0
    theta = np.pi / 4.3
    gradient = evaluator.evaluate_gradient([r, theta])
    assert pytest.approx(gradient[0][0].real) == np.cos(theta)
    assert pytest.approx(gradient[0][0].imag) == np.sin(theta)
    assert pytest.approx(gradient[0][1].real) == -r * np.sin(theta)
    assert pytest.approx(gradient[0][1].imag) == r * np.cos(theta)
