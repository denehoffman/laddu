import laddu as ld
from laddu import generation


def test_generation_module_exports_event_generator() -> None:
    assert generation.EventGenerator is ld.EventGenerator
    assert generation.GenReaction is ld.GenReaction


def test_generation_smoke() -> None:
    beam = generation.InitialStateParticle(
        'beam',
        generation.GenInitialState.beam_with_fixed_energy(0.0, 8.0),
        generation.Reconstruction.reconstructed(['beam']),
    )
    target = generation.InitialStateParticle(
        'target',
        generation.GenInitialState.target(0.938272),
        generation.Reconstruction.missing(),
    )
    kshort1 = generation.FinalStateParticle(
        'kshort1',
        generation.GenFinalState(0.497611),
        generation.Reconstruction.reconstructed(['kshort1']),
    )
    kshort2 = generation.FinalStateParticle(
        'kshort2',
        generation.GenFinalState(0.497611),
        generation.Reconstruction.reconstructed(['kshort2']),
    )
    kk = generation.FinalStateParticle.composite(
        'kk',
        generation.GenComposite(1.1, 1.6),
        (kshort1, kshort2),
    )
    recoil = generation.FinalStateParticle(
        'recoil',
        generation.GenFinalState(0.938272),
        generation.Reconstruction.reconstructed(['recoil']),
    )
    reaction = generation.GenReaction.two_to_two(
        beam,
        target,
        kk,
        recoil,
        generation.MandelstamTDistribution.exponential(0.1),
    )
    generator = generation.EventGenerator(reaction, seed=12345)
    dataset = generator.generate_dataset(4)

    assert dataset.n_events == 4
    assert dataset.p4_names == ['beam', 'target', 'kk', 'kshort1', 'kshort2', 'recoil']
