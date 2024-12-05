from laddu import Event, Dataset, Vector3, Mass, Manager, parameter
from laddu.amplitudes.kmatrix import (
    KopfKMatrixF0,
    KopfKMatrixF2,
    KopfKMatrixA0,
    KopfKMatrixA2,
    KopfKMatrixRho,
    KopfKMatrixPi1,
)
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


def test_f0_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixF0(
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
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert pytest.approx(result[0].real) == 0.2674945
    assert pytest.approx(result[0].imag) == 0.7289451


def test_f0_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixF0(
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
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert pytest.approx(result[0][0].real) == -0.0324912
    assert pytest.approx(result[0][0].imag) == -0.01107348
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.0241053
    assert pytest.approx(result[0][2].imag) == 0.007918499
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)
    assert pytest.approx(result[0][4].real) == -0.0316345
    assert pytest.approx(result[0][4].imag) == 0.01491556
    assert pytest.approx(result[0][5].real) == pytest.approx(-result[0][4].imag)
    assert pytest.approx(result[0][5].imag) == pytest.approx(result[0][4].real)
    assert pytest.approx(result[0][6].real) == 0.5838982
    assert pytest.approx(result[0][6].imag) == 0.2071617
    assert pytest.approx(result[0][7].real) == pytest.approx(-result[0][6].imag)
    assert pytest.approx(result[0][7].imag) == pytest.approx(result[0][6].real)
    assert pytest.approx(result[0][8].real) == 0.0914546
    assert pytest.approx(result[0][8].imag) == 0.03607718
    assert pytest.approx(result[0][9].real) == pytest.approx(-result[0][8].imag)
    assert pytest.approx(result[0][9].imag) == pytest.approx(result[0][8].real)


def test_f2_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixF2(
        'f2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
            (parameter('p4'), parameter('p5')),
            (parameter('p6'), parameter('p7')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert pytest.approx(result[0].real) == 0.02523304
    assert pytest.approx(result[0].imag) == 0.3971239


def test_f2_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixF2(
        'f2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
            (parameter('p4'), parameter('p5')),
            (parameter('p6'), parameter('p7')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert pytest.approx(result[0][0].real) == -0.3078948
    assert pytest.approx(result[0][0].imag) == 0.3808689
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.4290085
    assert pytest.approx(result[0][2].imag) == 0.0799660
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)
    assert pytest.approx(result[0][4].real) == 0.1657487
    assert pytest.approx(result[0][4].imag) == -0.00413829
    assert pytest.approx(result[0][5].real) == pytest.approx(-result[0][4].imag)
    assert pytest.approx(result[0][5].imag) == pytest.approx(result[0][4].real)
    assert pytest.approx(result[0][6].real) == 0.0594691
    assert pytest.approx(result[0][6].imag) == 0.1143819
    assert pytest.approx(result[0][7].real) == pytest.approx(-result[0][6].imag)
    assert pytest.approx(result[0][7].imag) == pytest.approx(result[0][6].real)


def test_a0_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixA0(
        'a0',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == -0.8002759
    assert pytest.approx(result[0].imag) == -0.1359306


def test_a0_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixA0(
        'a0',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == 0.2906192
    assert pytest.approx(result[0][0].imag) == -0.0998906
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == -1.3136838
    assert pytest.approx(result[0][2].imag) == 1.1380269
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_a2_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixA2(
        'a2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == -0.2092661
    assert pytest.approx(result[0].imag) == -0.0985062


def test_a2_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixA2(
        'a2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == -0.5756896
    assert pytest.approx(result[0][0].imag) == 0.9398863
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == -0.0811143
    assert pytest.approx(result[0][2].imag) == -0.1522787
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_rho_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixRho(
        'rho',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == 0.0948355
    assert pytest.approx(result[0].imag) == 0.2609183


def test_rho_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixRho(
        'rho',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == 0.0265203
    assert pytest.approx(result[0][0].imag) == -0.02660265
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.5172379
    assert pytest.approx(result[0][2].imag) == 0.1707373
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_pi1_evaluation():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixPi1(
        'pi1',
        ((parameter('p0'), parameter('p1')),),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([0.1, 0.2])
    assert pytest.approx(result[0].real) == -0.1101758
    assert pytest.approx(result[0].imag) == 0.2638717


def test_pi1_gradient():
    manager = Manager()
    res_mass = Mass([2, 3])
    amp = KopfKMatrixPi1(
        'pi1',
        ((parameter('p0'), parameter('p1')),),
        1,
        res_mass,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2])
    assert pytest.approx(result[0][0].real) == -14.7987174
    assert pytest.approx(result[0][0].imag) == -5.8430094
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
