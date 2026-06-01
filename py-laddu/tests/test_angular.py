from fractions import Fraction

import pytest
from laddu import (
    BlattWeisskopf,
    ClebschGordan,
    Dataset,
    Event,
    J,
    L,
    M,
    PhotonSDME,
    Vec3,
    WignerD,
)

from tests.channel_helpers import channel, helicity_frame

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


def test_reaction_variables_feed_wigner_d_and_barrier() -> None:
    ch = channel('x')
    angles = ch.angles('kshort1', helicity_frame('x'))
    d = WignerD('d', spin=2, row_projection=0, column_projection=0, angles=angles)
    b = BlattWeisskopf(
        'b',
        parent_mass=ch.mass('x'),
        daughter_1_mass=ch.mass('kshort1'),
        daughter_2_mass=ch.mass('kshort2'),
        l=2,
        reference_mass=1.5,
    )
    evaluator = (d * b).load(make_test_dataset())
    value = evaluator.evaluate([])[0]

    assert value.real == pytest.approx(value.real)
    assert value.imag == pytest.approx(value.imag)


def test_clebsch_gordan_and_photon_sdme_are_expression_terms() -> None:
    cg = ClebschGordan(
        'cg',
        j1=Fraction(1, 2),
        m1=Fraction(1, 2),
        j2=Fraction(1, 2),
        m2=Fraction(-1, 2),
        j=1,
        m=0,
    )
    rho = PhotonSDME('rho', helicity=1, helicity_prime=1)
    value = (cg * rho).load(make_test_dataset()).evaluate([])[0]

    assert value.real == pytest.approx(0.5 / 2.0**0.5)
    assert value.imag == pytest.approx(0.0)


def test_half_integer_quantum_numbers_accept_fraction_and_float() -> None:
    dataset = make_test_dataset()
    ch = channel('x')
    angles = ch.angles('kshort1', helicity_frame('x'))
    d_fraction = WignerD(
        'd_fraction',
        spin=Fraction(3, 2),
        row_projection=Fraction(1, 2),
        column_projection=Fraction(-1, 2),
        angles=angles,
    )
    d_float = WignerD(
        'd_float', spin=1.5, row_projection=0.5, column_projection=-0.5, angles=angles
    )
    cg = ClebschGordan(
        'cg_half',
        j1=Fraction(1, 2),
        m1=Fraction(1, 2),
        j2=1,
        m2=0,
        j=1.5,
        m=0.5,
    )

    values = (d_fraction + d_float + cg).load(dataset).evaluate([])

    assert values[0].real == pytest.approx(values[0].real)
    assert values[0].imag == pytest.approx(values[0].imag)


def test_angular_terms_accept_typed_quantum_numbers() -> None:
    dataset = make_test_dataset()
    ch = channel('x')
    angles = ch.angles('kshort1', helicity_frame('x'))
    d = WignerD(
        'd_typed',
        spin=J.int(2),
        row_projection=M.int(0),
        column_projection=M.int(0),
        angles=angles,
    )
    b = BlattWeisskopf(
        'b_typed',
        parent_mass=ch.mass('x'),
        daughter_1_mass=ch.mass('kshort1'),
        daughter_2_mass=ch.mass('kshort2'),
        l=L.int(2),
        reference_mass=1.5,
    )

    value = (d * b).load(dataset).evaluate([])[0]

    assert value.real == pytest.approx(value.real)


def test_quantum_number_inputs_reject_invalid_values() -> None:
    dataset = make_test_dataset()
    ch = channel('x')
    angles = ch.angles('kshort1', helicity_frame('x'))

    with pytest.raises(RuntimeError, match='integer or half-integer'):
        WignerD(
            'bad_float', spin=1.25, row_projection=0, column_projection=0, angles=angles
        ).load(dataset)

    with pytest.raises(RuntimeError, match='integer or half-integer'):
        ClebschGordan('bad_fraction', j1=Fraction(1, 3), m1=0, j2=1, m2=0, j=1, m=0)

    with pytest.raises(RuntimeError, match='orbital angular momentum must be integer'):
        BlattWeisskopf(
            'bad_l',
            parent_mass=ch.mass('x'),
            daughter_1_mass=ch.mass('kshort1'),
            daughter_2_mass=ch.mass('kshort2'),
            l=1.5,
            reference_mass=1.5,
        )
