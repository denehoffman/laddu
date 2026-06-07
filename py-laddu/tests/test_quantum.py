from fractions import Fraction

import pytest
from laddu import (
    Channel,
    Charge,
    Isospin,
    J,
    L,
    M,
    Parity,
    PartialWave,
    ParticleProperties,
    RuleSet,
    S,
    SelectionRules,
    allowed_partial_waves,
    coupled_spins,
)


def test_sign_like_quantum_numbers_parse_consistent_aliases() -> None:
    assert str(Parity('+')) == '+'
    assert str(Parity('positive')) == '+'
    assert str(Parity('neg')) == '-'
    assert Parity.positive().value == 1
    assert Parity.negative().value == -1


def test_charge_and_isospin_use_physical_values() -> None:
    charge = Charge(Fraction(2, 3))
    isospin = Isospin(Fraction(1, 2), projection=Fraction(-1, 2))

    assert charge.value == Fraction(2, 3)
    assert isospin.isospin == Fraction(1, 2)
    assert isospin.projection == Fraction(-1, 2)

    typed_isospin = Isospin(J.half(1), projection=M.half(-1))
    assert typed_isospin.isospin == Fraction(1, 2)
    assert typed_isospin.projection == Fraction(-1, 2)


def test_particle_properties_accept_keyword_quantum_numbers() -> None:
    pi_plus = ParticleProperties(
        'pi+',
        spin=0,
        parity='-',
        charge=Charge(1),
        isospin=Isospin(1, projection=1),
    )

    assert pi_plus.name == 'pi+'
    assert pi_plus.spin == 0
    assert str(pi_plus.parity) == '-'
    assert pi_plus.charge_unchecked is not None
    assert pi_plus.charge.value == 1


def test_coupled_spins_and_partial_wave_validation() -> None:
    assert coupled_spins(Fraction(1, 2), Fraction(1, 2)) == [0, 1]
    assert SelectionRules.coupled_spins(1, 0) == [1]

    wave = PartialWave(j=1, l=0, s=1)
    assert wave.j == 1
    assert wave.l == 0
    assert wave.s == 1
    assert wave.label == '3S1'

    typed_wave = PartialWave(j=J.int(1), l=L.int(0), s=S.int(1))
    assert typed_wave.label == '3S1'

    with pytest.raises(RuntimeError, match='compatible'):
        PartialWave(j=2, l=0, s=1)


def test_allowed_partial_waves_for_one_plus_to_rho_pi() -> None:
    parent = ParticleProperties('X(1+)', spin=1, parity='+', charge=Charge(0))
    rho = ParticleProperties(
        'rho0',
        spin=1,
        parity='-',
        charge=Charge(0),
        isospin=Isospin(1, projection=0),
        self_conjugate=True,
    )
    pion = ParticleProperties(
        'pi0',
        spin=0,
        parity='-',
        charge=Charge(0),
        isospin=Isospin(1, projection=0),
        self_conjugate=True,
    )

    waves = allowed_partial_waves(parent, rho, pion, max_l=2, rules=RuleSet.strong())

    assert [(allowed.wave.label, allowed.wave.l) for allowed in waves] == [
        ('3S1', 0),
        ('3D1', 2),
    ]
    assert [str(allowed.parity) for allowed in waves] == ['+', '+']


def test_channel_two_body_couplings_for_identical_ksks() -> None:
    channel = Channel()
    channel.create_decay('x_decay', 'X', ['Ks1', 'Ks2'], rules='strong')
    kshort = ParticleProperties(
        'K_S',
        species='K_S',
        spin=0,
        parity='-',
        charge=Charge(0),
        strangeness=0,
        baryon_number=0,
        statistics='boson',
    )
    channel.edit_particle('Ks1', properties=kshort)
    channel.edit_particle('Ks2', properties=kshort)

    couplings = channel.two_body_couplings('x_decay', j_max=2, l_max=2)

    assert [coupling.wave.label for coupling in couplings] == ['1S0', '1D2']
    assert [(coupling.j, coupling.l, coupling.s) for coupling in couplings] == [
        (0, 0, 0),
        (2, 2, 0),
    ]
