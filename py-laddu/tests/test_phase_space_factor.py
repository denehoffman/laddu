import pytest
from laddu import (
    Dataset,
    Event,
    Particle,
    PhaseSpaceFactor,
    Reaction,
    Vec3,
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


def reaction_context() -> tuple[Reaction, Particle, Particle, Particle, Particle]:
    beam = Particle.measured('beam', 'beam')
    target = Particle.missing('target')
    kshort1 = Particle.measured('K_S1', 'kshort1')
    kshort2 = Particle.measured('K_S2', 'kshort2')
    kk = Particle.composite('KK', [kshort1, kshort2])
    proton = Particle.measured('proton', 'proton')
    return Reaction.two_to_two(beam, target, kk, proton), kk, kshort1, kshort2, proton


def test_phase_space_factor_evaluation() -> None:
    reaction, kk, _, _, proton = reaction_context()
    decay = reaction.decay(kk)
    recoil_mass = reaction.mass(proton)
    daughter_1_mass = decay.daughter_1_mass()
    daughter_2_mass = decay.daughter_2_mass()
    resonance_mass = decay.parent_mass()
    mandelstam_s = reaction.mandelstam('s')
    amp = PhaseSpaceFactor(
        'kappa',
        recoil_mass,
        daughter_1_mass,
        daughter_2_mass,
        resonance_mass,
        mandelstam_s,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 7.028417575882146e-05
    assert pytest.approx(result[0].imag) == 0.0


def test_phase_space_factor_gradient() -> None:
    reaction, kk, _, _, proton = reaction_context()
    decay = reaction.decay(kk)
    recoil_mass = reaction.mass(proton)
    daughter_1_mass = decay.daughter_1_mass()
    daughter_2_mass = decay.daughter_2_mass()
    resonance_mass = decay.parent_mass()
    mandelstam_s = reaction.mandelstam('s')
    amp = PhaseSpaceFactor(
        'kappa',
        recoil_mass,
        daughter_1_mass,
        daughter_2_mass,
        resonance_mass,
        mandelstam_s,
    )
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
