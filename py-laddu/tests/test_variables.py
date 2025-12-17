import pytest

from laddu import (
    Angles,
    CosTheta,
    Dataset,
    Event,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    Topology,
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


def reaction_topology() -> Topology:
    return Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton')


def test_mass_single_particle() -> None:
    event = make_test_event()
    mass = Mass(['proton'])
    assert mass.value(event) == 1.007


def test_mass_multiple_particles() -> None:
    event = make_test_event()
    mass = Mass(['kshort1', 'kshort2'])
    assert pytest.approx(mass.value(event)) == 1.3743786309153077


def test_costheta_helicity() -> None:
    event = make_test_event()
    costheta = CosTheta(reaction_topology(), 'kshort1', 'Helicity')
    assert pytest.approx(costheta.value(event)) == -0.4611175068834238


def test_phi_helicity() -> None:
    event = make_test_event()
    phi = Phi(reaction_topology(), 'kshort1', 'Helicity')
    assert pytest.approx(phi.value(event)) == -2.657462587335066


def test_costheta_gottfried_jackson() -> None:
    event = make_test_event()
    costheta = CosTheta(reaction_topology(), 'kshort1', 'Gottfried-Jackson')
    assert pytest.approx(costheta.value(event)) == 0.09198832278031577


def test_phi_gottfried_jackson() -> None:
    event = make_test_event()
    phi = Phi(reaction_topology(), 'kshort1', 'Gottfried-Jackson')
    assert pytest.approx(phi.value(event)) == -2.713913199133907


def test_angles() -> None:
    event = make_test_event()
    angles = Angles(reaction_topology(), 'kshort1', 'Helicity')
    assert pytest.approx(angles.costheta.value(event)) == -0.4611175068834238
    assert pytest.approx(angles.phi.value(event)) == -2.657462587335066


def test_pol_angle() -> None:
    event = make_test_event()
    pol_angle = PolAngle(reaction_topology(), 'pol_angle')
    assert pytest.approx(pol_angle.value(event)) == 1.93592989


def test_pol_magnitude() -> None:
    event = make_test_event()
    pol_magnitude = PolMagnitude('pol_magnitude')
    assert pytest.approx(pol_magnitude.value(event)) == 0.38562805


def test_polarization() -> None:
    event = make_test_event()
    polarization = Polarization(
        reaction_topology(), pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    assert pytest.approx(polarization.pol_angle.value(event)) == 1.93592989
    assert pytest.approx(polarization.pol_magnitude.value(event)) == 0.38562805


def test_mandelstam() -> None:
    event = make_test_event()
    base = reaction_topology()
    s = Mandelstam(base, 's')
    t = Mandelstam(reaction_topology(), 't')
    u = Mandelstam(reaction_topology(), 'u')
    alt = Topology.missing_k1('beam', 'proton', ['kshort1', 'kshort2'])
    sp = Mandelstam(alt, 's')
    tp = Mandelstam(alt, 't')
    up = Mandelstam(alt, 'u')
    assert pytest.approx(s.value(event)) == 18.504011052120063
    assert pytest.approx(s.value(event)) == pytest.approx(sp.value(event))
    assert pytest.approx(t.value(event)) == -0.19222859969898076
    assert pytest.approx(t.value(event)) == pytest.approx(tp.value(event))
    assert pytest.approx(u.value(event)) == -14.404198931464428
    assert pytest.approx(u.value(event)) == pytest.approx(up.value(event))
    m2_beam = event.get_p4_sum(['beam']).m2
    m2_recoil = event.get_p4_sum(['proton']).m2
    m2_res = event.get_p4_sum(['kshort1', 'kshort2']).m2
    assert (
        pytest.approx(
            s.value(event)
            + t.value(event)
            + u.value(event)
            - m2_beam
            - m2_recoil
            - m2_res,
            abs=1e-2,
        )
        == 1.0
    )


def test_variable_value_on() -> None:
    dataset = make_test_dataset()
    mass = Mass(['kshort1', 'kshort2'])
    values = mass.value_on(dataset)
    assert len(values) == 1
    assert pytest.approx(values[0]) == 1.3743786309153077
