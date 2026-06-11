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
    assert ld.amplitude.ParameterMap is ld.ParameterMap
    assert ld.amplitude.CompiledExpression is ld.CompiledExpression
    assert ld.amplitude.TestAmplitude is ld.TestAmplitude
    assert ld.vectors.Vec3 is ld.Vec3
    assert ld.vectors.Vec4 is ld.Vec4


def make_channel() -> ld.Channel:
    channel = ld.Channel()
    channel.create_production('production', ['beam', 'target'], ['x', 'recoil'])
    channel.create_decay('x_decay', 'x', ['d1', 'd2'])
    channel.edit_particle('beam', source=ld.ParticleSource.Stored)
    channel.edit_particle('target', source=ld.ParticleSource.Missing)
    channel.edit_particle('recoil', source=ld.ParticleSource.Stored)
    channel.edit_particle('d1', source=ld.ParticleSource.Stored)
    channel.edit_particle('d2', source=ld.ParticleSource.Stored)
    return channel


def test_channel_creates_topology_variables() -> None:
    channel = make_channel()
    frame = ld.Frame(
        'x_decay',
        ld.Axes.from_y_z(
            ld.Axis.normal('beam', 'recoil').at('production').flipped(),
            ld.Axis.opposite('recoil').at('x_decay'),
        ),
    )

    assert isinstance(channel.mass('x'), ld.Mass)
    assert isinstance(channel.angles('d1', frame), ld.Angles)
    assert isinstance(channel.mandelstam('production', 's'), ld.Mandelstam)
    assert isinstance(channel.pol_angle('production', 'pol_angle'), ld.PolAngle)
    assert isinstance(
        channel.polarization(
            'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
        ),
        ld.Polarization,
    )
    assert 'Channel' in repr(channel)


def test_channel_exposes_particle_and_vertex_snapshots() -> None:
    channel = make_channel()

    particles = channel.particles()
    vertices = channel.vertices()
    x = channel.particle('x')
    production = channel.vertex('production')

    assert [particle.label for particle in particles] == [
        'beam',
        'target',
        'x',
        'recoil',
        'd1',
        'd2',
    ]
    assert [vertex.label for vertex in vertices] == ['production', 'x_decay']
    assert x.from_endpoint == 'production'
    assert x.to_endpoint == 'x_decay'
    assert x.source is ld.ParticleSource.Inferred
    assert production.label == 'production'
    assert [particle.label for particle in channel.incoming_particles('production')] == [
        'beam',
        'target',
    ]
    assert [particle.label for particle in channel.outgoing_particles('x_decay')] == [
        'd1',
        'd2',
    ]
    assert [vertex.label for vertex in channel.decay_vertices('x')] == ['x_decay']


def test_channel_rejects_invalid_particle_queries() -> None:
    channel = make_channel()

    with pytest.raises(RuntimeError, match='Unknown particle'):
        channel.mass('missing')

    with pytest.raises(RuntimeError, match='Unknown particle'):
        channel.angles(
            'missing',
            ld.Frame(
                'x_decay',
                ld.Axes.from_y_z(
                    ld.Axis.normal('beam', 'recoil').at('production').flipped(),
                    ld.Axis.opposite('recoil').at('x_decay'),
                ),
            ),
        )


def test_domain_modules_export_expected_analysis_types() -> None:
    assert ld.reaction.Axis is ld.Axis
    assert ld.reaction.Axes is ld.Axes
    assert ld.reaction.Frame is ld.Frame
    assert ld.reaction.Channel is ld.Channel
    assert ld.reaction.Particle is ld.Particle
    assert ld.reaction.Vertex is ld.Vertex
    assert ld.variables.Mass is ld.Mass
    assert ld.variables.CosTheta is ld.CosTheta
    assert ld.likelihood.NLL is ld.NLL
    assert ld.optimize.ControlFlow is ld.ControlFlow
    assert ld.quantum.allowed_projections is ld.allowed_projections
    assert ld.quantum.allowed_partial_waves is ld.allowed_partial_waves
    assert ld.quantum.ParticleProperties is ld.ParticleProperties
    assert ld.quantum.Reflectivity is ld.Reflectivity


def test_user_facing_objects_have_readable_display() -> None:
    channel = make_channel()
    dataset = ld.Dataset.empty_local(p4_names=['beam', 'recoil', 'd1', 'd2'])

    assert 'Channel' in repr(channel)
    assert 'production' in str(channel)
    assert 'Dataset' in repr(dataset)
    assert 'beam' in str(dataset)
