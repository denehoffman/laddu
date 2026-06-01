import pytest
from laddu import Angles, Channel, Dataset, Event, Vec3, Ylm, Zlm
from laddu.amplitudes.angular import PolPhase

from tests.channel_helpers import channel, helicity_frame

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


def reaction_context() -> tuple[Channel, Angles]:
    ch = channel()
    return ch, ch.angles('kshort1', helicity_frame())


def test_ylm_evaluation() -> None:
    _, angles = reaction_context()
    result = Ylm('ylm', l=1, m=1, angles=angles).load(make_test_dataset()).evaluate([])
    assert pytest.approx(result[0].real) == 0.2713394403451028
    assert pytest.approx(result[0].imag) == 0.1426897184196572


def test_ylm_gradient() -> None:
    _, angles = reaction_context()
    result = (
        Ylm('ylm', l=1, m=1, angles=angles)
        .load(make_test_dataset())
        .evaluate_gradient([])
    )
    assert len(result[0]) == 0


def test_zlm_evaluation() -> None:
    reaction, angles = reaction_context()
    polarization = reaction.polarization(
        'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    result = (
        Zlm('zlm', l=1, m=1, r='+', angles=angles, polarization=polarization)
        .load(make_test_dataset())
        .evaluate([])
    )
    assert pytest.approx(result[0].real) == 0.042841277026400094
    assert pytest.approx(result[0].imag) == -0.23859639145706923


def test_zlm_gradient() -> None:
    reaction, angles = reaction_context()
    polarization = reaction.polarization(
        'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    result = (
        Zlm('zlm', l=1, m=1, r='+', angles=angles, polarization=polarization)
        .load(make_test_dataset())
        .evaluate_gradient([])
    )
    assert len(result[0]) == 0


def test_polphase_evaluation() -> None:
    reaction, _ = reaction_context()
    polarization = reaction.polarization(
        'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    result = (
        PolPhase('polphase', polarization=polarization)
        .load(make_test_dataset())
        .evaluate([])
    )
    assert pytest.approx(result[0].real) == -0.28729144623530045
    assert pytest.approx(result[0].imag) == -0.2572403892603803


def test_polphase_gradient() -> None:
    reaction, _ = reaction_context()
    polarization = reaction.polarization(
        'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    result = (
        PolPhase('polphase', polarization=polarization)
        .load(make_test_dataset())
        .evaluate_gradient([])
    )
    assert len(result[0]) == 0
