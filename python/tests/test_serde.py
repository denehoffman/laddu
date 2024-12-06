from laddu import Event, Dataset, Vector3, Mass, Manager, parameter, Scalar
from laddu.amplitudes.kmatrix import (
    KopfKMatrixF0,
    KopfKMatrixF2,
)
import pytest
import pickle


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


def test_serde():
    manager = Manager()
    res_mass = Mass([2, 3])
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
