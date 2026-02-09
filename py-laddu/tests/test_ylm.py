import pytest
from laddu import Angles, Dataset, Event, Topology, Vec3, Ylm

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


def test_ylm_evaluation() -> None:
    angles = Angles(reaction_topology(), 'kshort1', 'Helicity')
    amp = Ylm('ylm', 1, 1, angles)
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([])
    assert pytest.approx(result[0].real) == 0.2713394403451028
    assert pytest.approx(result[0].imag) == 0.1426897184196572


def test_ylm_gradient() -> None:
    angles = Angles(reaction_topology(), 'kshort1', 'Helicity')
    amp = Ylm('ylm', 1, 1, angles)
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_gradient([])
    assert len(result[0]) == 0  # amplitude has no parameters
