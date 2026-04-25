import math

import numpy as np
import pytest
from laddu import Dataset, Event, Mass, Vec3, parameter
from laddu.amplitudes.lookup import (
    LookupTable,
    LookupTableComplex,
    LookupTablePolar,
    LookupTableScalar,
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


def test_lookup_table_1d_nearest() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 0.25, 0.75, 1.0]],
        [1.0 + 0.0j, 2.0 + 3.0j, 4.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 3.0


def test_lookup_table_2d_row_major() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1']), Mass(['kshort1', 'kshort2'])],
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_zero_boundary() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 0.5, 1.0]],
        [1.0 + 0.0j, 2.0 + 0.0j],
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 0.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_clamp_boundary() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 0.5, 1.0]],
        [1.0 + 0.0j, 2.0 + 0.0j],
        boundary_mode='clamp',
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 2.0
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_1d_linear() -> None:
    amp = LookupTable(
        'lookup',
        variables=[Mass(['kshort1'])],
        axis_coordinates=[[0.0, 1.0]],
        values=[1.0 + 0.0j, 3.0 + 0.0j],
        interpolation='linear',
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 1.0 + 2.0 * 0.498
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_accepts_numpy_arrays() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1'])],
        axis_coordinates=np.array([[0.0, 1.0]], dtype=np.float64),
        values=np.array([1.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
        interpolation='linear',
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 1.0 + 2.0 * 0.498
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_2d_linear_row_major() -> None:
    amp = LookupTable(
        'lookup',
        [Mass(['kshort1']), Mass(['proton'])],
        [[0.0, 1.0], [1.0, 2.0]],
        [1.0 + 0.0j, 4.0 + 0.0j, 3.0 + 0.0j, 6.0 + 0.0j],
        interpolation='linear',
    )

    result = amp.load(make_test_dataset()).evaluate([])

    assert pytest.approx(result[0].real) == 1.0 + 2.0 * 0.498 + 3.0 * 0.007
    assert pytest.approx(result[0].imag) == 0.0


def test_lookup_table_linear_boundaries() -> None:
    zero = LookupTable(
        'lookup_zero',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 1.0]],
        [1.0 + 0.0j, 3.0 + 0.0j],
        interpolation='linear',
    )
    clamp = LookupTable(
        'lookup_clamp',
        [Mass(['kshort1', 'kshort2'])],
        [[0.0, 1.0]],
        [1.0 + 0.0j, 3.0 + 0.0j],
        interpolation='linear',
        boundary_mode='clamp',
    )

    dataset = make_test_dataset()
    zero_result = zero.load(dataset).evaluate([])
    clamp_result = clamp.load(dataset).evaluate([])

    assert pytest.approx(zero_result[0].real) == 0.0
    assert pytest.approx(zero_result[0].imag) == 0.0
    assert pytest.approx(clamp_result[0].real) == 3.0
    assert pytest.approx(clamp_result[0].imag) == 0.0


def test_lookup_table_scalar_parameters_and_gradient() -> None:
    amp = LookupTableScalar(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 0.25, 0.75, 1.0]],
        [parameter('p0'), parameter('p1'), parameter('p2')],
    ).norm_sqr()

    gradient = amp.load(make_test_dataset()).evaluate_gradient([1.0, 2.0, 3.0])

    assert pytest.approx(gradient[0][0].real) == 0.0
    assert pytest.approx(gradient[0][0].imag) == 0.0
    assert pytest.approx(gradient[0][1].real) == 4.0
    assert pytest.approx(gradient[0][1].imag) == 0.0
    assert pytest.approx(gradient[0][2].real) == 0.0
    assert pytest.approx(gradient[0][2].imag) == 0.0


def test_lookup_table_linear_scalar_parameters_and_gradient() -> None:
    amp = LookupTableScalar(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 1.0]],
        [parameter('p0'), parameter('p1')],
        interpolation='linear',
    ).norm_sqr()

    gradient = amp.load(make_test_dataset()).evaluate_gradient([1.0, 3.0])
    value = 0.502 * 1.0 + 0.498 * 3.0

    assert pytest.approx(gradient[0][0].real) == 2.0 * value * 0.502
    assert pytest.approx(gradient[0][0].imag) == 0.0
    assert pytest.approx(gradient[0][1].real) == 2.0 * value * 0.498
    assert pytest.approx(gradient[0][1].imag) == 0.0


def test_lookup_table_linear_complex_parameters_and_gradient() -> None:
    amp = LookupTableComplex(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 1.0]],
        [
            (parameter('re0'), parameter('im0')),
            (parameter('re1'), parameter('im1')),
        ],
        interpolation='linear',
    )

    evaluator = amp.load(make_test_dataset())
    params = np.array([1.0, 2.0, 3.0, 4.0])
    gradient = evaluator.evaluate_gradient(params)

    assert pytest.approx(gradient[0][0].real) == 0.502
    assert pytest.approx(gradient[0][0].imag) == 0.0
    assert pytest.approx(gradient[0][1].real) == 0.0
    assert pytest.approx(gradient[0][1].imag) == 0.502
    assert pytest.approx(gradient[0][2].real) == 0.498
    assert pytest.approx(gradient[0][2].imag) == 0.0
    assert pytest.approx(gradient[0][3].real) == 0.0
    assert pytest.approx(gradient[0][3].imag) == 0.498

    eps = 1e-6
    for iparam in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[iparam] += eps
        minus[iparam] -= eps
        finite_difference = (
            evaluator.evaluate(plus)[0] - evaluator.evaluate(minus)[0]
        ) / (2.0 * eps)
        assert pytest.approx(gradient[0][iparam].real) == finite_difference.real
        assert pytest.approx(gradient[0][iparam].imag) == finite_difference.imag


def test_lookup_table_complex_parameters() -> None:
    amp = LookupTableComplex(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 0.25, 0.75, 1.0]],
        [
            (parameter('re0'), parameter('im0')),
            (parameter('re1'), parameter('im1')),
            (parameter('re2'), parameter('im2')),
        ],
    )

    result = amp.load(make_test_dataset()).evaluate([1.1, 1.2, 2.1, 2.2, 3.1, 3.2])

    assert pytest.approx(result[0].real) == 2.1
    assert pytest.approx(result[0].imag) == 2.2


def test_lookup_table_polar_parameters() -> None:
    amp = LookupTablePolar(
        'lookup',
        [Mass(['kshort1'])],
        [[0.0, 0.25, 0.75, 1.0]],
        [
            (parameter('r0'), parameter('theta0')),
            (parameter('r1'), parameter('theta1')),
            (parameter('r2'), parameter('theta2')),
        ],
    )

    result = amp.load(make_test_dataset()).evaluate([1.1, 1.2, 2.1, 2.2, 3.1, 3.2])

    assert pytest.approx(result[0].real) == 2.1 * math.cos(2.2)
    assert pytest.approx(result[0].imag) == 2.1 * math.sin(2.2)


def test_lookup_table_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='lookup-table values'):
        LookupTable(
            'lookup',
            [Mass(['kshort1'])],
            [[0.0, 0.5, 1.0]],
            [1.0 + 0.0j],
        )
