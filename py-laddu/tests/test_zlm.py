import pytest

from laddu import Angles, Dataset, Event, Manager, Polarization, Vec3, Zlm
from laddu.amplitudes.zlm import PolPhase

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']
AUX_VALUES = [0.38562805, 1.93592989]


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


def test_zlm_evaluation() -> None:
    manager = Manager()
    angles = Angles('beam', ['proton'], ['kshort1'], ['kshort1', 'kshort2'], 'Helicity')
    polarization = Polarization('beam', ['proton'], 'pol_magnitude', 'pol_angle')
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 0.042841277026400094
    assert pytest.approx(result[0].imag) == -0.23859639145706923


def test_zlm_gradient() -> None:
    manager = Manager()
    angles = Angles('beam', ['proton'], ['kshort1'], ['kshort1', 'kshort2'], 'Helicity')
    polarization = Polarization('beam', ['proton'], 'pol_magnitude', 'pol_angle')
    amp = Zlm('zlm', 1, 1, '+', angles, polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters


def test_polphase_evaluation() -> None:
    manager = Manager()
    polarization = Polarization('beam', ['proton'], 'pol_magnitude', 'pol_angle')
    amp = PolPhase('polphase', polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == -0.28729144623530045
    assert pytest.approx(result[0].imag) == -0.2572403892603803


def test_polphase_gradient() -> None:
    manager = Manager()
    polarization = Polarization('beam', ['proton'], 'pol_magnitude', 'pol_angle')
    amp = PolPhase('polphase', polarization)
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
