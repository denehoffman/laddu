from fractions import Fraction

import pytest
from laddu import (
    BlattWeisskopf,
    ClebschGordan,
    Dataset,
    Event,
    Particle,
    PhotonSDME,
    Reaction,
    Vec3,
    WignerD,
)

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']
AUX_VALUES = [0.38562805, 0.05708078]


def make_test_dataset() -> Dataset:
    return Dataset(
        [
            Event(
                [
                    Vec3(0.0, 0.0, 8.747).with_mass(0.0),
                    Vec3(0.119, 0.374, 0.222).with_mass(1.007),
                    Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
                    Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
                ],
                AUX_VALUES.copy(),
                1.0,
                p4_names=P4_NAMES,
                aux_names=AUX_NAMES,
            ),
        ],
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def reaction() -> tuple[Reaction, Particle, Particle, Particle]:
    beam = Particle.measured('beam', 'beam')
    target = Particle.missing('target')
    recoil = Particle.measured('recoil', 'proton')
    ks1 = Particle.measured('K_S1', 'kshort1')
    ks2 = Particle.measured('K_S2', 'kshort2')
    x = Particle.composite('X', [ks1, ks2])
    return Reaction.two_to_two(beam, target, x, recoil), x, ks1, ks2


def test_reaction_variables_feed_wigner_d_and_barrier() -> None:
    rxn, x, ks1, _ = reaction()
    decay = rxn.decay(x)
    angles = decay.angles(ks1, 'Helicity')
    d = WignerD('d', 2, 0, 0, angles)
    b = BlattWeisskopf('b', decay, 2, 1.5)
    evaluator = (d * b).load(make_test_dataset())
    value = evaluator.evaluate([])[0]

    assert value.real == pytest.approx(value.real)
    assert value.imag == pytest.approx(value.imag)


def test_clebsch_gordan_and_photon_sdme_are_expression_terms() -> None:
    cg = ClebschGordan(
        'cg',
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(-1, 2),
        1,
        0,
    )
    rho = PhotonSDME('rho', 1, 1)
    value = (cg * rho).load(make_test_dataset()).evaluate([])[0]

    assert value.real == pytest.approx(0.5 / 2.0**0.5)
    assert value.imag == pytest.approx(0.0)


def test_half_integer_quantum_numbers_accept_fraction_and_float() -> None:
    dataset = make_test_dataset()
    rxn, x, ks1, _ = reaction()
    angles = rxn.decay(x).angles(ks1, 'Helicity')
    d_fraction = WignerD(
        'd_fraction',
        Fraction(3, 2),
        Fraction(1, 2),
        Fraction(-1, 2),
        angles,
    )
    d_float = WignerD('d_float', 1.5, 0.5, -0.5, angles)
    cg = ClebschGordan(
        'cg_half',
        Fraction(1, 2),
        Fraction(1, 2),
        1,
        0,
        1.5,
        0.5,
    )

    values = (d_fraction + d_float + cg).load(dataset).evaluate([])

    assert values[0].real == pytest.approx(values[0].real)
    assert values[0].imag == pytest.approx(values[0].imag)


def test_quantum_number_inputs_reject_invalid_values() -> None:
    dataset = make_test_dataset()
    rxn, x, ks1, _ = reaction()
    angles = rxn.decay(x).angles(ks1, 'Helicity')

    with pytest.raises(RuntimeError, match='integer or half-integer'):
        WignerD('bad_float', 1.25, 0, 0, angles).load(dataset)

    with pytest.raises(RuntimeError, match='integer or half-integer'):
        ClebschGordan('bad_fraction', Fraction(1, 3), 0, 1, 0, 1, 0)

    with pytest.raises(RuntimeError, match='orbital angular momentum must be an integer'):
        BlattWeisskopf('bad_l', rxn.decay(x), 1.5, 1.5)


def test_decay_helicity_factor_matches_explicit_wigner_d() -> None:
    dataset = make_test_dataset()
    rxn, x, ks1, _ = reaction()
    decay = rxn.decay(x)
    factor = decay.helicity_factor('h', 2, 1, ks1, 1, 0)
    explicit = WignerD('d', 2, 1, 1, decay.angles(ks1, 'Helicity')).conj()

    factor_value = factor.load(dataset).evaluate([])[0]
    explicit_value = explicit.load(dataset).evaluate([])[0]

    assert factor_value.real == pytest.approx(explicit_value.real)
    assert factor_value.imag == pytest.approx(explicit_value.imag)


def test_decay_canonical_factor_matches_explicit_product() -> None:
    dataset = make_test_dataset()
    rxn, x, ks1, _ = reaction()
    decay = rxn.decay(x)
    factor = decay.canonical_factor('c', 2, 0, 2, 0, ks1, 0, 0, 0, 0)
    explicit = (
        ClebschGordan('ls_cg', 2, 0, 0, 0, 2, 0)
        * ClebschGordan('spin_cg', 0, 0, 0, 0, 0, 0)
        * WignerD('d', 2, 0, 0, decay.angles(ks1, 'Helicity')).conj()
    )

    factor_value = factor.load(dataset).evaluate([])[0]
    explicit_value = explicit.load(dataset).evaluate([])[0] * (5.0**0.5)

    assert factor_value.real == pytest.approx(explicit_value.real)
    assert factor_value.imag == pytest.approx(explicit_value.imag)
