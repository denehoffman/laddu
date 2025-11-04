import pytest

from laddu import Dataset, Event, Manager, Mandelstam, Mass, PhaseSpaceFactor, Vec3

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


def test_phase_space_factor_evaluation() -> None:
    manager = Manager()
    recoil_mass = Mass(['proton'])
    daughter_1_mass = Mass(['kshort1'])
    daughter_2_mass = Mass(['kshort2'])
    resonance_mass = Mass(['kshort1', 'kshort2'])
    mandelstam_s = Mandelstam(['beam'], [], ['kshort1', 'kshort2'], ['proton'], 's')
    amp = PhaseSpaceFactor(
        'kappa',
        recoil_mass,
        daughter_1_mass,
        daughter_2_mass,
        resonance_mass,
        mandelstam_s,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert (
        pytest.approx(result[0].real) == 0.0000702841757
    )  # NOTE: change in precision from Rust test
    assert pytest.approx(result[0].imag) == 0.0


def test_phase_space_factor_gradient() -> None:
    manager = Manager()
    recoil_mass = Mass(['proton'])
    daughter_1_mass = Mass(['kshort1'])
    daughter_2_mass = Mass(['kshort2'])
    resonance_mass = Mass(['kshort1', 'kshort2'])
    mandelstam_s = Mandelstam(['beam'], [], ['kshort1', 'kshort2'], ['proton'], 's')
    amp = PhaseSpaceFactor(
        'kappa',
        recoil_mass,
        daughter_1_mass,
        daughter_2_mass,
        resonance_mass,
        mandelstam_s,
    )
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
