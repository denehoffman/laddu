from laddu import (
    Mass,
    CosTheta,
    Phi,
    Angles,
    PolMagnitude,
    PolAngle,
    Polarization,
    Mandelstam,
    Event,
    Vector3,
    Dataset,
)
import pytest


def make_test_event() -> Event:
    return Event(
        [
            Vector3(0.0, 0.0, 8.747).with_mass(0.0),
            Vector3(0.119, 0.374, 0.222).with_mass(1.007),
            Vector3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vector3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [Vector3(0.385, 0.022, 0.000)],
        0.48,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()])


def test_mass_single_particle():
    event = make_test_event()
    mass = Mass([1])
    assert mass.value(event) == 1.007


def test_mass_multiple_particles():
    event = make_test_event()
    mass = Mass([2, 3])
    assert pytest.approx(mass.value(event)) == 1.3743786


def test_costheta_helicity():
    event = make_test_event()
    costheta = CosTheta(0, [1], [2], [2, 3], "Helicity")
    assert pytest.approx(costheta.value(event)) == -0.4611175


def test_phi_helicity():
    event = make_test_event()
    phi = Phi(0, [1], [2], [2, 3], "Helicity")
    assert pytest.approx(phi.value(event)) == -2.6574625


def test_costheta_gottfried_jackson():
    event = make_test_event()
    costheta = CosTheta(0, [1], [2], [2, 3], "Gottfried-Jackson")
    assert pytest.approx(costheta.value(event)) == 0.09198832


def test_phi_gottfried_jackson():
    event = make_test_event()
    phi = Phi(0, [1], [2], [2, 3], "Gottfried-Jackson")
    assert pytest.approx(phi.value(event)) == -2.7139131


def test_angles():
    event = make_test_event()
    angles = Angles(0, [1], [2], [2, 3], "Helicity")
    assert pytest.approx(angles.costheta.value(event)) == -0.4611175
    assert pytest.approx(angles.phi.value(event)) == -2.6574625


def test_pol_angle():
    event = make_test_event()
    pol_angle = PolAngle(0, [1])
    assert pytest.approx(pol_angle.value(event)) == 1.9359298


def test_pol_magnitude():
    event = make_test_event()
    pol_magnitude = PolMagnitude(0)
    assert pytest.approx(pol_magnitude.value(event)) == 0.3856280


def test_polarization():
    event = make_test_event()
    polarization = Polarization(0, [1])
    assert pytest.approx(polarization.pol_angle.value(event)) == 1.9359298
    assert pytest.approx(polarization.pol_magnitude.value(event)) == 0.3856280


def test_mandelstam():
    event = make_test_event()
    s = Mandelstam([0], [], [2, 3], [1], "s")
    t = Mandelstam([0], [], [2, 3], [1], "t")
    u = Mandelstam([0], [], [2, 3], [1], "u")
    sp = Mandelstam([], [0], [1], [2, 3], "s")
    tp = Mandelstam([], [0], [1], [2, 3], "t")
    up = Mandelstam([], [0], [1], [2, 3], "u")
    assert pytest.approx(s.value(event)) == 18.504011
    assert pytest.approx(s.value(event)) == pytest.approx(sp.value(event))
    assert pytest.approx(t.value(event)) == -0.1922285
    assert pytest.approx(t.value(event)) == pytest.approx(tp.value(event))
    assert pytest.approx(u.value(event)) == -14.4041989
    assert pytest.approx(u.value(event)) == pytest.approx(up.value(event))
    m2_beam = event.get_p4_sum([0]).m2
    m2_recoil = event.get_p4_sum([1]).m2
    m2_res = event.get_p4_sum([2, 3]).m2
    assert (
        pytest.approx(
            s.value(event) + t.value(event) + u.value(event) - m2_beam - m2_recoil - m2_res,
            abs=1e-2,
        )
        == 1.0
    )


def test_variable_value_on():
    dataset = make_test_dataset()
    mass = Mass([2, 3])
    values = mass.value_on(dataset)
    assert len(values) == 1
    assert pytest.approx(values[0]) == 1.3743786
