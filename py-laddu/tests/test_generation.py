import laddu as ld
from laddu import generation


def test_generation_module_exports_event_generator() -> None:
    assert generation.EventGenerator is ld.EventGenerator
    assert generation.GeneratedBatch is ld.GeneratedBatch
    assert generation.GeneratedEventLayout is ld.GeneratedEventLayout
    assert generation.GeneratedReaction is ld.GeneratedReaction


def test_generation_smoke() -> None:
    beam = generation.GeneratedParticle.initial(
        'beam',
        generation.InitialGenerator.beam_with_fixed_energy(0.0, 8.0),
        generation.Reconstruction.stored(),
    )
    target = generation.GeneratedParticle.initial(
        'target',
        generation.InitialGenerator.target(0.938272),
        generation.Reconstruction.missing(),
    )
    kshort1 = generation.GeneratedParticle.stable(
        'kshort1',
        generation.StableGenerator(0.497611),
        generation.Reconstruction.stored(),
    )
    kshort2 = generation.GeneratedParticle.stable(
        'kshort2',
        generation.StableGenerator(0.497611),
        generation.Reconstruction.stored(),
    )
    kk = generation.GeneratedParticle.composite(
        'kk',
        generation.CompositeGenerator(1.1, 1.6),
        (kshort1, kshort2),
        generation.Reconstruction.composite(),
    )
    recoil = generation.GeneratedParticle.stable(
        'recoil',
        generation.StableGenerator(0.938272),
        generation.Reconstruction.stored(),
    )
    reaction = generation.GeneratedReaction.two_to_two(
        beam,
        target,
        kk,
        recoil,
        generation.MandelstamTDistribution.exponential(0.1),
    )
    generator = generation.EventGenerator(reaction, seed=12345)
    batch = generator.generate_batch(4)
    dataset = batch.dataset

    assert dataset.n_events == 4
    assert dataset.p4_names == ['beam', 'target', 'kk', 'kshort1', 'kshort2', 'recoil']
    assert batch.layout.p4_labels == dataset.p4_names
    assert batch.layout.aux_labels == []
    assert batch.reaction.p4_labels() == dataset.p4_names
    assert [particle.id for particle in batch.layout.particles] == [
        'beam',
        'target',
        'kk',
        'kshort1',
        'kshort2',
        'recoil',
    ]
    assert [particle.product_id for particle in batch.layout.particles] == list(range(6))
    assert [particle.parent_id for particle in batch.layout.particles] == [
        None,
        None,
        None,
        2,
        2,
        None,
    ]
    assert [particle.p4_label for particle in batch.layout.particles] == dataset.p4_names
    assert reaction.reconstructed_reaction().decay('kk').daughters() == [
        'kshort1',
        'kshort2',
    ]
    assert generator.generate_dataset(4).n_events == 4
