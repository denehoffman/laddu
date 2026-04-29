from typing import Any, cast

import laddu as ld
import laddu.amplitude
import laddu.likelihood
import laddu.optimize
import laddu.quantum
import laddu.reaction
import laddu.variables
import laddu.vectors
import pytest


def test_domain_modules_export_expected_core_types() -> None:
    assert ld.amplitude.Expression is ld.Expression
    assert ld.amplitude.Evaluator is ld.Evaluator
    assert ld.amplitude.Parameter is ld.Parameter
    assert ld.vectors.Vec3 is ld.Vec3
    assert ld.vectors.Vec4 is ld.Vec4


def test_decay_exposes_enclosing_reaction() -> None:
    beam = ld.Particle.stored('beam')
    target = ld.Particle.missing('target')
    recoil = ld.Particle.stored('recoil')
    daughter_1 = ld.Particle.stored('d1')
    daughter_2 = ld.Particle.stored('d2')
    parent = ld.Particle.composite('x', (daughter_1, daughter_2))
    reaction = ld.Reaction.two_to_two(beam, target, parent, recoil)

    decay = reaction.decay('x')

    assert isinstance(decay.reaction, ld.Reaction)
    assert isinstance(decay.reaction.mass('x'), ld.Mass)


def test_reaction_rejects_invalid_particle_queries() -> None:
    beam = ld.Particle.stored('beam')
    target = ld.Particle.missing('target')
    recoil = ld.Particle.stored('recoil')
    daughter_1 = ld.Particle.stored('d1')
    daughter_2 = ld.Particle.stored('d2')
    parent = ld.Particle.composite('x', (daughter_1, daughter_2))
    reaction = ld.Reaction.two_to_two(beam, target, parent, recoil)

    with pytest.raises(RuntimeError, match="unknown reaction particle 'missing'"):
        reaction.decay('missing')

    with pytest.raises(RuntimeError, match='isobar decays must contain exactly two'):
        reaction.decay('d1')


def test_composite_particles_require_two_daughters() -> None:
    daughter_1 = ld.Particle.stored('d1')
    daughter_2 = ld.Particle.stored('d2')
    daughter_3 = ld.Particle.stored('d3')

    with pytest.raises(ValueError, match='exactly two ordered daughters'):
        ld.Particle.composite('too_few', cast(Any, (daughter_1,)))

    with pytest.raises(ValueError, match='exactly two ordered daughters'):
        ld.Particle.composite('too_many', cast(Any, (daughter_1, daughter_2, daughter_3)))

    with pytest.raises(TypeError):
        ld.Particle.composite('list_input', cast(Any, [daughter_1, daughter_2]))


def test_domain_modules_export_expected_analysis_types() -> None:
    assert ld.reaction.Particle is ld.Particle
    assert ld.reaction.Reaction is ld.Reaction
    assert ld.reaction.Decay is ld.Decay
    assert ld.variables.Mass is ld.Mass
    assert ld.variables.CosTheta is ld.CosTheta
    assert ld.likelihood.NLL is ld.NLL
    assert ld.optimize.ControlFlow is ld.ControlFlow
    assert ld.quantum.allowed_projections is ld.allowed_projections
