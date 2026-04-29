import laddu as ld
import laddu.amplitude
import laddu.likelihood
import laddu.optimize
import laddu.quantum
import laddu.reaction
import laddu.variables
import laddu.vectors


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
    parent = ld.Particle.composite('x', [daughter_1, daughter_2])
    reaction = ld.Reaction.two_to_two(beam, target, parent, recoil)

    decay = reaction.decay('x')

    assert isinstance(decay.reaction, ld.Reaction)
    assert isinstance(decay.reaction.mass('x'), ld.Mass)


def test_domain_modules_export_expected_analysis_types() -> None:
    assert ld.reaction.Particle is ld.Particle
    assert ld.reaction.Reaction is ld.Reaction
    assert ld.reaction.Decay is ld.Decay
    assert ld.variables.Mass is ld.Mass
    assert ld.variables.CosTheta is ld.CosTheta
    assert ld.likelihood.NLL is ld.NLL
    assert ld.optimize.ControlFlow is ld.ControlFlow
    assert ld.quantum.allowed_projections is ld.allowed_projections
