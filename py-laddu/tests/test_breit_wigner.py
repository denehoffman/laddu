import pytest

from laddu import BreitWigner, Dataset, Event, Manager, Mass, Vec3, parameter

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


def test_bw_evaluation() -> None:
    manager = Manager()
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 1.458569174900372
    assert pytest.approx(result[0].imag) == 1.4107341131495694


def test_bw_gradient() -> None:
    manager = Manager()
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([1.7, 0.3])
    assert pytest.approx(result[0][0].real) == -2.4105851202988857
    assert pytest.approx(result[0][0].imag) == -1.8880913749138584
    assert pytest.approx(result[0][1].real) == 1.0467031328673773
    assert pytest.approx(result[0][1].imag) == 1.3683612879088032
