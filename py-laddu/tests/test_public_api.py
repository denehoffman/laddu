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


def test_domain_modules_export_expected_analysis_types() -> None:
    assert ld.reaction.Particle is ld.Particle
    assert ld.reaction.Reaction is ld.Reaction
    assert ld.reaction.Decay is ld.Decay
    assert ld.variables.Mass is ld.Mass
    assert ld.variables.CosTheta is ld.CosTheta
    assert ld.likelihood.NLL is ld.NLL
    assert ld.optimize.ControlFlow is ld.ControlFlow
    assert ld.quantum.allowed_projections is ld.allowed_projections
