# ruff: noqa: PT027, S101, PLR2004

import unittest

import laddu as ld
import numpy as np
import pytest

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


@pytest.mark.parametrize('method', ['weighted', 'unweighted'])
def test_generated_indices_are_contiguous_output_ordinals(method) -> None:
    generator = decay_generator()
    options = {'max_weight': generator.phase_space_envelope().maximum_weight * 2} if method == 'unweighted' else {}
    original, baseline = getattr(generator, method)(16, seed=17, **options)
    indexed, report = getattr(generator, method)(
        16, seed=17, index_column='id', index_dtype=np.uint16, index_start=100, **options
    )
    np.testing.assert_array_equal(indexed.column('id'), np.arange(100, 116, dtype=np.uint16))
    assert indexed.column('id').dtype == np.dtype('uint16')
    np.testing.assert_array_equal(indexed.weights(), original.weights())
    assert baseline.proposals == report.proposals
    if method == 'unweighted':
        assert report.proposals > report.produced
    for left, right in zip(original.batches(), indexed.batches(), strict=True):
        for name in original.p4_names():
            np.testing.assert_array_equal(left.p4s[name], right.p4s[name])


@pytest.mark.parametrize('dtype', ['u8', 'uint16', np.uint32, np.dtype('u8')])
def test_generated_index_dtype_descriptors(dtype) -> None:
    data, _ = decay_generator().weighted(4, index_column='id', index_dtype=dtype, index_start=20)
    assert data.column('id').tolist() == [20, 21, 22, 23]
    expected = 'uint8' if isinstance(dtype, str) and dtype == 'u8' else np.dtype(dtype).name
    assert data.schema.columns == {'id': expected}


@pytest.mark.parametrize(
    'options',
    [
        {'index_column': 'id', 'index_dtype': 'i64'},
        {'index_column': 'id', 'index_dtype': 'u8', 'index_start': 250},
        {'index_column': 'id', 'index_start': 2**64 - 1},
        {'index_column': 'id', 'index_start': -1},
        {'index_start': 7},
        {'index_column': 'a_e'},
    ],
)
def test_generated_index_validation(options) -> None:
    with pytest.raises((ld.LadduError, ValueError, OverflowError)):
        decay_generator().weighted(16, **options)


def test_generated_indices_after_envelope_growth_and_across_chunks() -> None:
    generator = decay_generator()
    dataset, report = generator.unweighted(
        64,
        max_weight=1e-12,
        grow_envelope=True,
        seed=17,
        index_column='id',
        index_dtype='u64',
        index_start=2**63,
        memory='64KiB',
    )
    assert report.envelope is not None
    assert report.envelope > 1e-12
    np.testing.assert_array_equal(dataset.column('id'), np.arange(64, dtype=np.uint64) + 2**63)
    weighted, report = generator.weighted(300, memory='64KiB', index_column='id', index_start=500)
    assert report.chunk_events < 300
    np.testing.assert_array_equal(weighted.column('id'), np.arange(500, 800, dtype=np.uint64))


@pytest.mark.parametrize('seed', [3, 5, 17])
def test_indices_preserve_adaptive_sampling_under_a_tight_budget(seed) -> None:
    generator = ld.Generator(decay_channel(), scalars={'v': ld.ScalarSource.uniform(0.001, 1.0)})
    model = ld.Model(ld.scalar('v'))

    def generate(index_column: str | None = None, index_start: int = 0):
        return generator.unweighted(
            20,
            model=model,
            max_weight=1e-12,
            grow_envelope=True,
            seed=seed,
            memory='8KiB',
            index_column=index_column,
            index_start=index_start,
        )

    baseline, original_report = generate()
    indexed, indexed_report = generate('id', 100)
    np.testing.assert_array_equal(indexed.column('v'), baseline.column('v'))
    assert indexed_report.chunk_events == original_report.chunk_events
    assert indexed_report.estimated_peak_bytes <= 8 * 1024
    assert indexed_report.envelope == original_report.envelope
    assert indexed_report.proposals == original_report.proposals
    for left, right in zip(baseline.batches(), indexed.batches(), strict=True):
        for name in baseline.p4_names():
            np.testing.assert_array_equal(left.p4s[name], right.p4s[name])
    np.testing.assert_array_equal(indexed.column('id'), np.arange(100, 120, dtype=np.uint64))


@pytest.mark.parametrize('format_name', ['parquet', 'root', 'custom'])
def test_generated_ids_align_after_roundtrip_views_and_mapping_evaluation(format_name, tmp_path) -> None:
    generator = ld.Generator(decay_channel(), scalars={'v': ld.ScalarSource.uniform(0.001, 1.0)})
    data, _ = generator.weighted(20, seed=17, index_column='id', index_start=2**63)
    path = tmp_path / ('generated.' + format_name)
    if format_name == 'custom':
        schema = data.schema

        def writer(path, batches, **_kwargs):
            batches = list(batches)
            arrays = {name: np.concatenate([batch.p4s[name] for batch in batches]) for name in schema.p4s}
            arrays.update({name: np.concatenate([batch.scalars[name] for batch in batches]) for name in schema.scalars})
            arrays.update({name: np.concatenate([batch.columns[name] for batch in batches]) for name in schema.columns})
            with path.open('wb') as file:
                np.savez(
                    file, allow_pickle=False, **arrays, weights=np.concatenate([batch.weights for batch in batches])
                )

        def reader(path, **_kwargs):
            with np.load(path, allow_pickle=False) as arrays:
                yield ld.io.EventBatch(
                    p4s={name: arrays[name] for name in schema.p4s},
                    scalars={name: arrays[name] for name in schema.scalars},
                    columns={name: arrays[name] for name in schema.columns},
                    weights=arrays['weights'],
                )

        fmt = ld.io.FormatSpec(
            name='generated-numpy',
            describe=lambda *_args, **_kwargs: ld.io.SourceInfo(schema),
            read_batches=reader,
            write_batches=writer,
        )
        fmt.write(data, path, chunk_size=3)
        restored = fmt.read(path, cache='streaming')
    else:
        sink = ld.ParquetSink(path) if format_name == 'parquet' else ld.RootSink(path)
        data.write_to(sink, chunk_size=3)
        restored = (ld.read_parquet if format_name == 'parquet' else ld.read_root)(path, cache='streaming')

    rows = np.flatnonzero(data.column('v') > 0.5)
    view = restored.select(ld.scalar('v') > 0.5)
    expected = data.column('v')[rows]
    values = view.evaluate({'v': ld.scalar('v'), 'squared': ld.scalar('v') * ld.scalar('v')}, real=True)
    np.testing.assert_array_equal(view.column('id'), data.column('id')[rows])
    assert view.column('id').dtype == np.dtype('uint64')
    np.testing.assert_array_equal(view.weights(), np.asarray(data.weights())[rows])
    np.testing.assert_array_equal(values['v'], expected)
    np.testing.assert_array_equal(values['squared'], expected**2)


class ProvenEnvelopeTests(unittest.TestCase):
    def test_scalar_sources_are_exported_and_accepted(self) -> None:
        source = ld.ScalarSource.uniform(0.2, 0.3)
        assert source.to_json() == '{"kind":"uniform","min":0.2,"max":0.3}'
        restored = ld.ScalarSource.from_json(source.to_json())
        generator = ld.Generator(decay_channel(), scalars={'polarization': restored})
        dataset, _ = generator.weighted(1, seed=17)
        value = dataset.evaluate(ld.scalar('polarization'), real=True)[0]
        assert 0.2 <= value < 0.3

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
