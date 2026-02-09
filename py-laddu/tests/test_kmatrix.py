import pytest
from laddu import Dataset, Event, Mass, Vec3, parameter
from laddu.amplitudes.kmatrix import (
    KopfKMatrixA0,
    KopfKMatrixA2,
    KopfKMatrixF0,
    KopfKMatrixF2,
    KopfKMatrixPi1,
    KopfKMatrixRho,
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


def test_f0_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
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
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert pytest.approx(result[0].real) == 0.2674945594859745
    assert pytest.approx(result[0].imag) == 0.7289451151846622


def test_f0_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
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
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert pytest.approx(result[0][0].real) == -0.032491219879072594
    assert pytest.approx(result[0][0].imag) == -0.011073489047324615
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.02410530896582612
    assert pytest.approx(result[0][2].imag) == 0.007918499653925656
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)
    assert pytest.approx(result[0][4].real) == -0.031634528397387424
    assert pytest.approx(result[0][4].imag) == 0.01491556758888564
    assert pytest.approx(result[0][5].real) == pytest.approx(-result[0][4].imag)
    assert pytest.approx(result[0][5].imag) == pytest.approx(result[0][4].real)
    assert pytest.approx(result[0][6].real) == 0.5838982754419436
    assert pytest.approx(result[0][6].imag) == 0.20716175256804892
    assert pytest.approx(result[0][7].real) == pytest.approx(-result[0][6].imag)
    assert pytest.approx(result[0][7].imag) == pytest.approx(result[0][6].real)
    assert pytest.approx(result[0][8].real) == 0.09145465471022667
    assert pytest.approx(result[0][8].imag) == 0.03607718440586096
    assert pytest.approx(result[0][9].real) == pytest.approx(-result[0][8].imag)
    assert pytest.approx(result[0][9].imag) == pytest.approx(result[0][8].real)


def test_f2_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
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
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert pytest.approx(result[0].real) == 0.025233045240226293
    assert pytest.approx(result[0].imag) == 0.39712393858386263


def test_f2_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
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
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert pytest.approx(result[0][0].real) == -0.3078948637910409
    assert pytest.approx(result[0][0].imag) == 0.38086899234534155
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.42900856602686843
    assert pytest.approx(result[0][2].imag) == 0.0799660634186314
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)
    assert pytest.approx(result[0][4].real) == 0.1657487556679948
    assert pytest.approx(result[0][4].imag) == -0.004138290603113022
    assert pytest.approx(result[0][5].real) == pytest.approx(-result[0][4].imag)
    assert pytest.approx(result[0][5].imag) == pytest.approx(result[0][4].real)
    assert pytest.approx(result[0][6].real) == 0.059469124136208376
    assert pytest.approx(result[0][6].imag) == 0.11438194180427544
    assert pytest.approx(result[0][7].real) == pytest.approx(-result[0][6].imag)
    assert pytest.approx(result[0][7].imag) == pytest.approx(result[0][6].real)


def test_a0_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixA0(
        'a0',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == -0.8002759157259999
    assert pytest.approx(result[0].imag) == -0.1359306632058216


def test_a0_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixA0(
        'a0',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == 0.2906192438344459
    assert pytest.approx(result[0][0].imag) == -0.09989060459904309
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == -1.313683875655594
    assert pytest.approx(result[0][2].imag) == 1.1380269958314373
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_a2_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixA2(
        'a2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == -0.2092661754354623
    assert pytest.approx(result[0].imag) == -0.09850621309829852


def test_a2_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixA2(
        'a2',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == -0.575689604769787
    assert pytest.approx(result[0][0].imag) == 0.9398863940931068
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == -0.08111430722946257
    assert pytest.approx(result[0][2].imag) == -0.15227874234387567
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_rho_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixRho(
        'rho',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0].real) == 0.09483558754117698
    assert pytest.approx(result[0].imag) == 0.2609183741271106


def test_rho_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixRho(
        'rho',
        (
            (parameter('p0'), parameter('p1')),
            (parameter('p2'), parameter('p3')),
        ),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2, 0.3, 0.4])
    assert pytest.approx(result[0][0].real) == 0.026520319348816407
    assert pytest.approx(result[0][0].imag) == -0.026602652559793133
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
    assert pytest.approx(result[0][2].real) == 0.5172379289201292
    assert pytest.approx(result[0][2].imag) == 0.17073733305788397
    assert pytest.approx(result[0][3].real) == pytest.approx(-result[0][2].imag)
    assert pytest.approx(result[0][3].imag) == pytest.approx(result[0][2].real)


def test_pi1_evaluation() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixPi1(
        'pi1',
        ((parameter('p0'), parameter('p1')),),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([0.1, 0.2])
    assert pytest.approx(result[0].real) == -0.11017586807747382
    assert pytest.approx(result[0].imag) == 0.2638717244927622


def test_pi1_gradient() -> None:
    res_mass = Mass(['kshort1', 'kshort2'])
    amp = KopfKMatrixPi1(
        'pi1',
        ((parameter('p0'), parameter('p1')),),
        1,
        res_mass,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([0.1, 0.2])
    assert pytest.approx(result[0][0].real) == -14.798717468937502
    assert pytest.approx(result[0][0].imag) == -5.843009428873981
    assert pytest.approx(result[0][1].real) == pytest.approx(-result[0][0].imag)
    assert pytest.approx(result[0][1].imag) == pytest.approx(result[0][0].real)
