from laddu import Vector3, Vector4
import numpy as np
import pytest


def test_three_to_four_momentum_conversion():
    p3 = Vector3(1.0, 2.0, 3.0)
    target_p4 = Vector4(1.0, 2.0, 3.0, 10.0)
    p4_from_mass = p3.with_mass(target_p4.m)
    assert target_p4.e == p4_from_mass.e
    assert target_p4.px == p4_from_mass.px
    assert target_p4.py == p4_from_mass.py
    assert target_p4.pz == p4_from_mass.pz
    p4_from_energy = p3.with_energy(target_p4.e)
    assert target_p4.e == p4_from_energy.e
    assert target_p4.px == p4_from_energy.px
    assert target_p4.py == p4_from_energy.py
    assert target_p4.pz == p4_from_energy.pz


def test_four_momentum_basics():
    p = Vector4(3.0, 4.0, 5.0, 10.0)
    assert p.e == 10.0
    assert p.px == 3.0
    assert p.py == 4.0
    assert p.pz == 5.0
    assert p.momentum.px == 3.0
    assert p.momentum.py == 4.0
    assert p.momentum.pz == 5.0
    assert p.beta.px == 0.3
    assert p.beta.py == 0.4
    assert p.beta.pz == 0.5
    assert p.m2 == 50.0
    assert p.m == np.sqrt(50.0)
    assert repr(p) == '[e = 10.00000; p = (3.00000, 4.00000, 5.00000); m = 7.07107]'


def test_three_momentum_basics():
    p = Vector4(3.0, 4.0, 5.0, 10.0)
    q = Vector4(1.2, -3.4, 7.6, 0.0)
    p3_view = p.momentum
    q3_view = q.momentum
    assert p3_view.px == 3.0
    assert p3_view.py == 4.0
    assert p3_view.pz == 5.0
    assert p3_view.mag2 == 50.0
    assert p3_view.mag == np.sqrt(50.0)
    assert p3_view.costheta == 5.0 / np.sqrt(50.0)
    assert p3_view.theta == np.acos(5.0 / np.sqrt(50.0))
    assert p3_view.phi == np.atan2(4.0, 3.0)
    assert pytest.approx(p3_view.unit.px) == 3.0 / np.sqrt(50.0)
    assert pytest.approx(p3_view.unit.py) == 4.0 / np.sqrt(50.0)
    assert pytest.approx(p3_view.unit.pz) == 5.0 / np.sqrt(50.0)
    assert pytest.approx(p3_view.cross(q3_view).px) == 47.4
    assert pytest.approx(p3_view.cross(q3_view).py) == -16.8
    assert pytest.approx(p3_view.cross(q3_view).pz) == -15.0
    p3 = Vector3(3.0, 4.0, 5.0)
    q3 = Vector3(1.2, -3.4, 7.6)
    assert p3.px == 3.0
    assert p3.py == 4.0
    assert p3.pz == 5.0
    assert p3.mag2 == 50.0
    assert p3.mag == np.sqrt(50.0)
    assert p3.costheta == 5.0 / np.sqrt(50.0)
    assert p3.theta == np.acos(5.0 / np.sqrt(50.0))
    assert p3.phi == np.atan2(4.0, 3.0)
    assert pytest.approx(p3.unit.px) == 3.0 / np.sqrt(50.0)
    assert pytest.approx(p3.unit.py) == 4.0 / np.sqrt(50.0)
    assert pytest.approx(p3.unit.pz) == 5.0 / np.sqrt(50.0)
    assert pytest.approx(p3.cross(q3).px) == 47.4
    assert pytest.approx(p3.cross(q3).py) == -16.8
    assert pytest.approx(p3.cross(q3).pz) == -15.0


def test_boost_com():
    p = Vector4(3.0, 4.0, 5.0, 10.0)
    zero = p.boost(-p.beta)
    assert zero.px == 0.0
    assert zero.py == 0.0
    assert zero.pz == 0.0


def test_boost():
    p1 = Vector4(3.0, 4.0, 5.0, 10.0)
    p2 = Vector4(3.4, 2.3, 1.2, 9.0)
    p1_boosted = p1.boost(-p2.beta)
    assert p1_boosted.e == 8.157632144622882
    assert p1_boosted.px == -0.6489200627053444
    assert p1_boosted.py == 1.5316128987581492
    assert p1_boosted.pz == 3.712145860221643
