import pytest
from laddu import (
    BreitWigner,
    BreitWignerNonRelativistic,
    Dataset,
    Event,
    Flatte,
    Mass,
    Particle,
    PhaseSpaceFactor,
    Reaction,
    Vec3,
    Voigt,
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


def reaction_context() -> tuple[Reaction, Particle, Particle]:
    beam = Particle.stored('beam')
    target = Particle.missing('target')
    kshort1 = Particle.stored('kshort1')
    kshort2 = Particle.stored('kshort2')
    kk = Particle.composite('kk', [kshort1, kshort2])
    proton = Particle.stored('proton')
    return Reaction.two_to_two(beam, target, kk, proton), kk, proton


def test_bw_evaluation() -> None:
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 1.4308791652435884
    assert pytest.approx(result[0].imag) == 1.3839522217669178


def test_bw_gradient() -> None:
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([1.7, 0.3])
    assert pytest.approx(result[0][0].real) == -2.4885111876269255
    assert pytest.approx(result[0][0].imag) == -1.8242624730389174
    assert pytest.approx(result[0][1].real) == -0.5492978554232557
    assert pytest.approx(result[0][1].imag) == 0.7828010830313784


def test_bw_no_bwbf_evaluation() -> None:
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
        barrier_factors=False,
    )
    result = amp.load(make_test_dataset()).evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 2.0654840145948157
    assert pytest.approx(result[0].imag) == 1.2058262598870575


def test_bw_no_bwbf_gradient() -> None:
    amp = BreitWigner(
        'bw',
        parameter('mass'),
        parameter('width'),
        2,
        Mass(['kshort1']),
        Mass(['kshort2']),
        Mass(['kshort1', 'kshort2']),
        barrier_factors=False,
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([1.7, 0.3])
    assert pytest.approx(result[0][0].real) == -3.2382865275566544
    assert pytest.approx(result[0][0].imag) == -0.9544869810058523
    assert pytest.approx(result[0][1].real) == -0.06116353148223782
    assert pytest.approx(result[0][1].imag) == 0.3131899140692953


def test_bw_nonrel_evaluation() -> None:
    amp = BreitWignerNonRelativistic(
        'bw',
        parameter('mass'),
        parameter('width'),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate([1.5, 0.3])
    assert pytest.approx(result[0].real) == 1.084721431628924
    assert pytest.approx(result[0].imag) == 1.3518336007116172


def test_bw_nonrel_gradient() -> None:
    amp = BreitWignerNonRelativistic(
        'bw',
        parameter('mass'),
        parameter('width'),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([1.7, 0.3])
    assert pytest.approx(result[0][0].real) == -1.7757650016553739
    assert pytest.approx(result[0][0].imag) == -2.0392238297998153
    assert pytest.approx(result[0][1].real) == -1.0894724338203443
    assert pytest.approx(result[0][1].imag) == 0.7917525805669601


def test_flatte_evaluation() -> None:
    amp = Flatte(
        'flatte',
        parameter('mass'),
        parameter('g_obs'),
        parameter('g_alt'),
        (Mass(['kshort1']), Mass(['kshort2'])),
        (0.1349768, 0.547862),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate([0.98, 0.7, 0.2])
    assert pytest.approx(result[0].real) == -0.7338320342780681
    assert pytest.approx(result[0].imag) == 0.5018145529787819


def test_flatte_gradient() -> None:
    amp = Flatte(
        'flatte',
        parameter('mass'),
        parameter('g_obs'),
        parameter('g_alt'),
        (Mass(['kshort1']), Mass(['kshort2'])),
        (0.1349768, 0.547862),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([0.98, 0.7, 0.2])
    assert pytest.approx(result[0][0].real) == -0.08473788905152731
    assert pytest.approx(result[0][0].imag) == 1.6292790093139917
    assert pytest.approx(result[0][1].real) == 0.497349582793617
    assert pytest.approx(result[0][1].imag) == 0.19360065665801518
    assert pytest.approx(result[0][2].real) == 0.597447011338709
    assert pytest.approx(result[0][2].imag) == 0.23256505627570476


def test_voigt_evaluation() -> None:
    amp = Voigt(
        'voigt',
        parameter('mass'),
        parameter('width'),
        parameter('sigma'),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate([0.98, 0.08, 0.02])
    assert pytest.approx(result[0].real) == 0.2857389147779551
    assert pytest.approx(result[0].imag) == 0.0


def test_voigt_gradient() -> None:
    amp = Voigt(
        'voigt',
        parameter('mass'),
        parameter('width'),
        parameter('sigma'),
        Mass(['kshort1', 'kshort2']),
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([0.98, 0.08, 0.02])
    assert pytest.approx(result[0][0].real) == 0.7225730704295464
    assert pytest.approx(result[0][0].imag) == 0.0
    assert pytest.approx(result[0][1].real) == 1.7488427782862053
    assert pytest.approx(result[0][1].imag) == 0.0
    assert pytest.approx(result[0][2].real) == 0.10952492310922711
    assert pytest.approx(result[0][2].imag) == 0.0


def test_phase_space_factor_evaluation() -> None:
    reaction, _, _ = reaction_context()
    decay = reaction.decay('kk')
    amp = PhaseSpaceFactor(
        'kappa',
        reaction.mass('proton'),
        decay.daughter_1_mass(),
        decay.daughter_2_mass(),
        decay.parent_mass(),
        reaction.mandelstam('s'),
    )
    result = amp.load(make_test_dataset()).evaluate([])
    assert pytest.approx(result[0].real) == 7.028417575882146e-05
    assert pytest.approx(result[0].imag) == 0.0


def test_phase_space_factor_gradient() -> None:
    reaction, _, _ = reaction_context()
    decay = reaction.decay('kk')
    amp = PhaseSpaceFactor(
        'kappa',
        reaction.mass('proton'),
        decay.daughter_1_mass(),
        decay.daughter_2_mass(),
        decay.parent_mass(),
        reaction.mandelstam('s'),
    )
    result = amp.load(make_test_dataset()).evaluate_gradient([])
    assert len(result[0]) == 0
