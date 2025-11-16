import pickle

import pytest

from laddu import Dataset, Event, Manager, Mass, Scalar, Vec3, parameter
from laddu.amplitudes.kmatrix import (
    KopfKMatrixF0,
    KopfKMatrixF2,
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


def test_serde() -> None:
    manager = Manager()
    res_mass = Mass(['kshort1', 'kshort2'])
    f0 = KopfKMatrixF0(
        'f0',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
            (parameter('p4'), parameter('p5')),
            (parameter('p6'), parameter('p7')),
            (parameter('p8'), parameter('p9')),
        ),
        1,
        res_mass,
    )
    f2 = KopfKMatrixF2(
        'f2',
        (
            (parameter('g0'), parameter('g1')),
            (parameter('g2'), parameter('g3')),
            (parameter('g4'), parameter('g5')),
            (parameter('g6'), parameter('g7')),
        ),
        1,
        res_mass,
    )
    s = Scalar('s', parameter('s'))
    f0_aid = manager.register(f0)
    f2_aid = manager.register(f2)
    s_aid = manager.register(s)
    expr = (f0_aid * s_aid + f2_aid).norm_sqr()
    model = manager.model(expr)
    dataset = make_test_dataset()
    evaluator = model.load(dataset)
    p = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
    result = evaluator.evaluate(p)
    pickled_model = pickle.dumps(model)
    unpickled_model = pickle.loads(pickled_model)
    unpickled_evaluator = unpickled_model.load(dataset)
    unpickled_result = unpickled_evaluator.evaluate(p)

    assert pytest.approx(result[0].real) == pytest.approx(unpickled_result[0].real)
    assert pytest.approx(result[0].imag) == pytest.approx(unpickled_result[0].imag)
