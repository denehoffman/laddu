import laddu as ld
import numpy as np
import pytest
from laddu import generation


def make_generation_channel() -> ld.Channel:
    channel = ld.Channel()
    channel.create_production(
        'production',
        ['beam', 'target'],
        ['kk', 'recoil'],
        generator=ld.gen.t_exponential(0.1),
    )
    channel.create_decay('kk_decay', 'kk', ['kshort1', 'kshort2'])

    channel.edit_particle('beam', mass=0.0, momentum=ld.gen.energy(8.0))
    channel.edit_particle('target', mass=0.938272, momentum=ld.gen.rest())
    channel.edit_particle('kk', mass_sampler=ld.gen.uniform_mass(1.1, 1.6))
    channel.edit_particle('recoil', mass=0.938272)
    channel.edit_particle('kshort1', mass=0.497611)
    channel.edit_particle('kshort2', mass=0.497611)
    return channel


def test_generation_module_exports_event_generator() -> None:
    assert generation.EventGenerator is ld.EventGenerator
    assert generation.GeneratedEvent is ld.GeneratedEvent
    assert generation.GenerationPlan is ld.GenerationPlan
    assert generation.MomentumSource is ld.MomentumSource
    assert generation.MassSampler is ld.MassSampler
    assert generation.VertexGenerator is ld.VertexGenerator


def test_generation_smoke() -> None:
    channel = make_generation_channel()
    generator = generation.EventGenerator(channel, seed=12345)
    result = generator.generate(4, generation.DatasetSink())
    dataset = result.output

    assert dataset.n_events == 4
    assert result.stats.written_events == 4
    assert dataset.p4_names == ['beam', 'target', 'kk', 'kshort1', 'kshort2', 'recoil']
    assert generator.p4_labels() == dataset.p4_names

    plan = generator.plan
    assert plan.production.vertex == 'production'
    assert [particle.label for particle in plan.production.incoming] == ['beam', 'target']
    assert [particle.mass for particle in plan.production.incoming] == [0.0, 0.938272]
    assert [particle.label for particle in plan.production.outgoing] == ['kk', 'recoil']
    assert plan.production.outgoing[0].mass.kind == 'sampled'
    assert plan.production.outgoing[1].mass.kind == 'properties'
    assert plan.production.outgoing[1].mass.value == 0.938272
    assert plan.production.outgoing[0].decay is not None
    assert plan.production.outgoing[0].decay.vertex == 'kk_decay'
    assert [
        particle.label for particle in plan.production.outgoing[0].decay.daughters
    ] == [
        'kshort1',
        'kshort2',
    ]

    generated_event = generator.generate_event()
    assert generated_event.labels() == dataset.p4_names
    assert generated_event.p4('beam') is not None
    assert generated_event.p4('missing') is None
    assert [label for label, _ in generated_event.p4s()] == dataset.p4_names

    repeated = generator.generate(4, generation.DatasetSink()).output
    for left, right in zip(dataset.events_global, repeated.events_global, strict=True):
        for name in dataset.p4_names:
            assert left.p4(name).e == right.p4(name).e
            assert left.p4(name).px == right.p4(name).px
            assert left.p4(name).py == right.p4(name).py
            assert left.p4(name).pz == right.p4(name).pz


def test_seeded_generate_event_is_deterministic_and_stateful() -> None:
    channel = make_generation_channel()
    first_generator = generation.EventGenerator(channel, seed=12345)
    second_generator = generation.EventGenerator(channel, seed=12345)

    first_event = first_generator.generate_event()
    repeated_first_event = second_generator.generate_event()
    second_event = first_generator.generate_event()

    for label, first_p4 in first_event.p4s():
        repeated_p4 = repeated_first_event.p4(label)
        assert repeated_p4 is not None
        assert first_p4.e == repeated_p4.e
        assert first_p4.px == repeated_p4.px
        assert first_p4.py == repeated_p4.py
        assert first_p4.pz == repeated_p4.pz

    first_p4s = dict(first_event.p4s())
    second_p4s = dict(second_event.p4s())
    assert any(first_p4s[label].pz != second_p4s[label].pz for label in first_p4s)


def test_histogram_generation_annotations() -> None:
    channel = make_generation_channel()
    histogram = ld.Histogram([1.1, 1.3, 1.6], [1.0, 2.0])
    channel.edit_particle('kk', mass_sampler=ld.gen.histogram_mass(histogram))
    channel.edit_particle(
        'beam', momentum=ld.gen.histogram_energy(ld.Histogram([8.0, 8.5], [1.0]))
    )
    channel.edit_vertex(
        'production',
        generator=ld.gen.t_histogram(ld.Histogram([-0.4, -0.1, 0.0], [1.0, 1.0])),
    )

    dataset = (
        generation.EventGenerator(channel, seed=12345)
        .generate(3, generation.DatasetSink())
        .output
    )
    assert dataset.n_events == 3
    assert np.all(np.isfinite(dataset.p4_column_global('kk')))


def test_generation_requires_particle_masses() -> None:
    missing_mass_channel = ld.Channel()
    missing_mass_channel.create_production(
        'production',
        ['beam', 'target'],
        ['kk', 'recoil'],
        generator=ld.gen.t_exponential(0.1),
    )
    missing_mass_channel.create_decay('kk_decay', 'kk', ['kshort1', 'kshort2'])
    missing_mass_channel.edit_particle('beam', momentum=ld.gen.energy(8.0))
    missing_mass_channel.edit_particle('target', momentum=ld.gen.rest())

    with pytest.raises(RuntimeError, match='ParticleProperties mass'):
        generation.EventGenerator(missing_mass_channel)
