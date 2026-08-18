# ruff: noqa: PT027, S101

import unittest

import laddu as ld

DECAY_DIMENSIONS = 2
N_EVENTS = 16


def decay_channel() -> ld.Channel:
    return ld.Channel(
        'proven envelope decay',
        edges=[
            ld.Edge(
                'parent',
                p4='parent',
                particle=ld.Particle(mass=2.0),
                initial_momentum=ld.InitialMomentum.p4([2.0, 0.0, 0.0, 0.0]),
            ),
            ld.Edge('a', p4='a', particle=ld.Particle(mass=0.2), output=True),
            ld.Edge('b', p4='b', particle=ld.Particle(mass=0.4), output=True),
        ],
        vertices=[
            ld.Vertex(
                'decay',
                incoming=['parent'],
                outgoing=['a', 'b'],
                generation=ld.VertexProposal.isotropic(),
            ),
        ],
    )


def decay_generator() -> ld.Generator:
    return ld.Generator(decay_channel())


class ProvenEnvelopeTests(unittest.TestCase):
    def test_scalar_sources_are_exported_and_accepted(self) -> None:
        source = ld.ScalarSource.uniform(0.2, 0.3)
        assert source.to_json() == '{"kind":"uniform","min":0.2,"max":0.3}'
        restored = ld.ScalarSource.from_json(source.to_json())
        generator = ld.Generator(decay_channel(), scalars={'polarization': restored})
        dataset, _ = generator.weighted(1, seed=17)
        value = dataset.evaluate(ld.scalar('polarization'), real=True)[0]
        assert 0.2 <= value < 0.3  # noqa: PLR2004

    def test_report_and_model_less_unweighting(self) -> None:
        generator = decay_generator()
        proven = generator.phase_space_envelope()

        assert proven.weight_interval[0] == 0.0
        assert proven.maximum_weight == proven.weight_interval[1]
        assert proven.continuous_dimensions == DECAY_DIMENSIONS
        assert proven.piecewise_regions == 1
        assert proven.subdivisions == 0

        dataset, report = generator.unweighted(
            N_EVENTS,
            proven_envelope=True,
            max_proposals=100,
            seed=17,
        )
        assert len(dataset) == N_EVENTS
        assert report.proven_weight_interval == proven.weight_interval
        assert report.proven_continuous_dimensions == DECAY_DIMENSIONS
        assert report.proven_piecewise_regions == 1
        assert report.proven_subdivisions == 0
        assert report.maximum_weight <= proven.maximum_weight

    def test_proven_envelope_rejects_a_model(self) -> None:
        generator = decay_generator()
        model = ld.Model(ld.Expr(1.0))
        with self.assertRaisesRegex(ValueError, 'only when model is omitted'):
            generator.unweighted(1, model, proven_envelope=True)


if __name__ == '__main__':
    unittest.main()
