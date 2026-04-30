from typing import cast

import laddu as ld
from laddu import generation


def test_generation_module_exports_event_generator() -> None:
    assert generation.EventGenerator is ld.EventGenerator
    assert generation.GeneratedBatch is ld.GeneratedBatch
    assert generation.GeneratedBatchIter is ld.GeneratedBatchIter
    assert generation.GeneratedEventLayout is ld.GeneratedEventLayout
    assert generation.GeneratedReaction is ld.GeneratedReaction
    assert generation.GeneratedStorage is ld.GeneratedStorage
    assert generation.ParticleSpecies is ld.ParticleSpecies


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
    assert [particle.produced_vertex_id for particle in batch.layout.particles] == [
        None,
        None,
        0,
        1,
        1,
        0,
    ]
    assert [particle.decay_vertex_id for particle in batch.layout.particles] == [
        None,
        None,
        1,
        None,
        None,
        None,
    ]
    assert [particle.p4_label for particle in batch.layout.particles] == dataset.p4_names
    assert [vertex.vertex_id for vertex in batch.layout.vertices] == [0, 1]
    assert [vertex.kind for vertex in batch.layout.vertices] == ['Production', 'Decay']
    assert [vertex.incoming_product_ids for vertex in batch.layout.vertices] == [
        [0, 1],
        [2],
    ]
    assert [vertex.outgoing_product_ids for vertex in batch.layout.vertices] == [
        [2, 5],
        [3, 4],
    ]
    assert reaction.reconstructed_reaction().decay('kk').daughters() == [
        'kshort1',
        'kshort2',
    ]
    assert generator.generate_dataset(4).n_events == 4

    one_shot = generator.generate_dataset(7)
    batches = list(generator.generate_batches(7, 3))
    assert [generated.dataset.n_events for generated in batches] == [3, 3, 1]
    for batch_offset, generated in zip([0, 3, 6], batches, strict=True):
        for local_index in range(generated.dataset.n_events):
            expected = one_shot[batch_offset + local_index]
            actual = generated.dataset[local_index]
            for name in one_shot.p4_names:
                assert actual.p4(name).e == expected.p4(name).e
                assert actual.p4(name).px == expected.p4(name).px
                assert actual.p4(name).py == expected.p4(name).py
                assert actual.p4(name).pz == expected.p4(name).pz

    import pytest

    with pytest.raises(RuntimeError, match='batch_size'):
        list(generator.generate_batches(1, 0))

    projected = generation.EventGenerator(
        reaction,
        seed=12345,
        storage=generation.GeneratedStorage.only(
            ('beam', 'target', 'kshort1', 'kshort2', 'recoil')
        ),
    ).generate_batch(2)
    assert projected.dataset.p4_names == [
        'beam',
        'target',
        'kshort1',
        'kshort2',
        'recoil',
    ]
    assert projected.layout.p4_labels == projected.dataset.p4_names
    assert projected.reaction.p4_labels() == [
        'beam',
        'target',
        'kk',
        'kshort1',
        'kshort2',
        'recoil',
    ]
    assert [particle.p4_label for particle in projected.layout.particles] == [
        'beam',
        'target',
        None,
        'kshort1',
        'kshort2',
        'recoil',
    ]

    with pytest.raises(RuntimeError, match='unknown particle ID'):
        generation.EventGenerator(
            reaction,
            storage=generation.GeneratedStorage.only(('beam', 'does_not_exist')),
        )


def test_particle_species_metadata_propagates_to_layout() -> None:
    photon = generation.ParticleSpecies.code(22)
    proton = generation.ParticleSpecies.with_namespace('pdg', 2212)
    kshort = generation.ParticleSpecies.label('KShort')
    kk_species = generation.ParticleSpecies.label('KK')

    assert photon.id == 22
    assert photon.namespace is None
    assert photon.label_value is None
    assert proton.id == 2212
    assert proton.namespace == 'pdg'
    assert kshort.id is None
    assert kshort.label_value == 'KShort'

    beam = generation.GeneratedParticle.initial(
        'beam',
        generation.InitialGenerator.beam_with_fixed_energy(0.0, 8.0),
        generation.Reconstruction.stored(),
    ).with_species(photon)
    target = generation.GeneratedParticle.initial(
        'target',
        generation.InitialGenerator.target(0.938272),
        generation.Reconstruction.missing(),
    ).with_species(proton)
    kshort1 = generation.GeneratedParticle.stable(
        'kshort1',
        generation.StableGenerator(0.497611),
        generation.Reconstruction.stored(),
    ).with_species(kshort)
    kshort2 = generation.GeneratedParticle.stable(
        'kshort2',
        generation.StableGenerator(0.497611),
        generation.Reconstruction.stored(),
    ).with_species(kshort)
    kk = generation.GeneratedParticle.composite(
        'kk',
        generation.CompositeGenerator(1.1, 1.6),
        (kshort1, kshort2),
        generation.Reconstruction.composite(),
    ).with_species(kk_species)
    recoil = generation.GeneratedParticle.stable(
        'recoil',
        generation.StableGenerator(0.938272),
        generation.Reconstruction.stored(),
    ).with_species(proton)
    reaction = generation.GeneratedReaction.two_to_two(
        beam,
        target,
        kk,
        recoil,
        generation.MandelstamTDistribution.exponential(0.1),
    )
    batch = generation.EventGenerator(reaction, seed=12345).generate_batch(1)

    maybe_species = [particle.species for particle in batch.layout.particles]
    assert all(item is not None for item in maybe_species)
    species = cast('list[generation.ParticleSpecies]', maybe_species)
    assert [item.id for item in species] == [22, 2212, None, None, None, 2212]
    assert [item.namespace for item in species] == [None, 'pdg', None, None, None, 'pdg']
    assert [item.label_value for item in species] == [
        None,
        None,
        'KK',
        'KShort',
        'KShort',
        None,
    ]
