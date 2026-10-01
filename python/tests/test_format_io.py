# ruff: noqa: S101, PT027, PLR2004, ARG001, EM101
"""Public extension contracts exercised through an independent record format."""

import json
import unittest
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import laddu as ld
import numpy as np


@dataclass(frozen=True)
class Options:
    particle: str = 'beam'
    scale: float = 1.0


def describe(path, *, options):
    return ld.io.SourceInfo(ld.io.Schema(p4s=[options.particle], scalars=['run'], weights=True))


def records(path, *, options, plan):
    with Path(path).open() as file:
        for line in file:
            record = json.loads(line)
            yield ld.io.EventBatch(
                p4s={options.particle: np.array([record['momentum']]) * options.scale},
                scalars={'run': np.array([record['run']], dtype=float)},
                weights=np.array([record['weight']], dtype=float),
            )


def write_records(path, batches, *, options, schema, plan):
    with Path(path).open('w') as file:
        for batch in batches:
            momenta = batch.p4s[options.particle] / options.scale
            runs = batch.scalars['run']
            file.writelines(
                json.dumps(
                    {
                        'momentum': momenta[i].tolist(),
                        'run': runs[i].item(),
                        'weight': batch.weights[i].item(),
                    }
                )
                + '\n'
                for i in range(len(batch))
            )


def record_format(**kwargs):
    return ld.io.FormatSpec(
        name='records',
        describe=describe,
        read_batches=records,
        write_batches=write_records,
        extension='jsonl',
        **kwargs,
    )


def sample():
    return ld.Dataset.from_arrays(
        p4s={'beam': np.array([[5.0, 0.0, 0.0, 5.0], [6.0, 0.0, 0.0, 6.0], [7.0, 0.0, 0.0, 7.0]])},
        scalars={'run': np.array([1.0, 2.0, 3.0])},
        weights=np.array([0.5, -1.0, 2.0]),
    )


class FormatTests(unittest.TestCase):
    def test_public_module_import(self):
        assert import_module('laddu.io').FormatSpec is ld.io.FormatSpec

    def test_owned_batches_and_column_free_events(self):
        array = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        batch = ld.io.EventBatch(p4s={'beam': array})
        array[:] = 0
        np.testing.assert_array_equal(batch.p4s['beam'], [[1.0, 2.0, 3.0, 4.0]])
        assert not batch.p4s['beam'].flags.writeable
        assert batch.weights is None
        assert len(ld.io.EventBatch(length=3)) == 3
        schema = ld.io.Schema()
        data = ld.Dataset.from_batches(lambda **_: iter([ld.io.EventBatch(length=3)]), schema=schema, cache='streaming')
        assert len(data) == 3
        assert schema.p4s == ()

    def test_spec_round_trip_options_and_conversion(self):
        fmt = record_format()
        options = Options(scale=1000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            result = fmt.write(sample(), path, options=options, chunk_size=1)
            assert result.events == 3
            assert result.paths == [path]
            data = ld.Dataset(fmt.source(path, options=options, cache='streaming'))
            np.testing.assert_array_equal(data.weights(), sample().weights())
            for original, restored in zip(sample().batches(chunk_size=1), data.batches(chunk_size=1), strict=True):
                np.testing.assert_array_equal(original.p4s['beam'], restored.p4s['beam'])
            target = Path(tmp) / 'output.jsonl'
            result = ld.io.convert(
                path, target, reader=fmt, writer=fmt, read_options=options, write_options=options, chunk_size=2
            )
            assert result.events == 3
            np.testing.assert_array_equal(fmt.read(target, options=options).weights(), data.weights())
            assert data.write_to(fmt.sink(Path(tmp) / 'bound.jsonl', options=options)).events == 3

    def test_named_columns_are_reordered(self):
        data = ld.Dataset.from_batches(
            lambda **_: iter([ld.io.EventBatch(scalars={'b': np.array([2.0]), 'a': np.array([1.0])})]),
            schema={'p4s': [], 'scalars': ['a', 'b']},
            cache='streaming',
        )
        batch = next(data.batches())
        assert tuple(batch.scalars) == ('a', 'b')
        np.testing.assert_array_equal(batch.scalars['a'], [1.0])

    def test_particle_array_records(self):
        particles = ('beam', 'recoil')
        components = ('energy', 'px', 'py', 'pz')

        def reader(path, *, options, plan):
            with Path(path).open() as file:
                for line in file:
                    record = json.loads(line)
                    vectors = np.array([record[key] for key in components], dtype=float).T
                    yield ld.io.EventBatch(p4s={name: vectors[i : i + 1] for i, name in enumerate(options)})

        def writer(path, batches, *, options, schema, plan):
            with Path(path).open('w') as file:
                for batch in batches:
                    vectors = np.stack([batch.p4s[name] for name in options], axis=1)
                    for event in vectors:
                        record = {key: event[:, i].tolist() for i, key in enumerate(components)}
                        file.write(json.dumps(record) + '\n')

        fmt = ld.io.FormatSpec(
            name='particle-arrays',
            describe=lambda _, *, options: ld.io.SourceInfo(ld.io.Schema(p4s=options)),
            read_batches=reader,
            write_batches=writer,
        )
        data = ld.Dataset.from_arrays(
            p4s={
                'beam': np.array([[9.0, 0.0, 0.0, 9.0], [8.0, 0.0, 0.0, 8.0]]),
                'recoil': np.array([[1.2, 0.1, 0.2, 0.3], [1.3, -0.1, -0.2, -0.3]]),
            },
            scalars={},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'array-records.jsonl'
            fmt.write(data, path, options=particles, chunk_size=2)
            restored = next(fmt.read(path, options=particles).batches())
            original = next(data.batches())
            for name in particles:
                np.testing.assert_array_equal(restored.p4s[name], original.p4s[name])
            np.testing.assert_array_equal(restored.weights, [1.0, 1.0])

    def test_effective_weights_and_transformations(self):
        data = sample().select(ld.scalar('run') > 1).bootstrap(seed=31)
        batches = list(data.batches(chunk_size=1))
        np.testing.assert_array_equal(np.concatenate([b.weights for b in batches]), data.weights())
        assert data.schema.weights
        np.testing.assert_array_equal(np.concatenate([b.scalars['run'] for b in batches]), [2.0, 3.0])
        retained = batches[0].p4s['beam']
        list(data.batches())
        assert retained.shape == (1, 4)
        subsampled = sample().subsample(0.7, seed=3)
        assert sum(map(len, subsampled.batches(chunk_size=1))) == len(subsampled)

    def test_early_close_releases_reader(self):
        closed = []

        def reader(source, *, options, plan):
            try:
                for _ in range(3):
                    yield ld.io.EventBatch(scalars={'x': np.array([1.0])})
            finally:
                closed.append(True)

        fmt = ld.io.FormatSpec(
            name='cleanup', describe=lambda _, **__: ld.io.SourceInfo(ld.io.Schema(scalars=['x'])), read_batches=reader
        )
        data = fmt.read(None, cache='streaming')
        with data.batches(chunk_size=1) as batches:
            next(batches)
        assert closed == [True]
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            next(batches)

    def test_callback_exception_preserves_cause(self):
        def reader(source, *, options, plan):
            yield ld.io.EventBatch(scalars={'x': np.array([1.0])})
            raise OSError('broken record')

        fmt = ld.io.FormatSpec(
            name='failing', describe=lambda _, **__: ld.io.SourceInfo(ld.io.Schema(scalars=['x'])), read_batches=reader
        )
        batches = fmt.read(None, cache='streaming').batches(chunk_size=1)
        next(batches)
        with self.assertRaisesRegex(ld.LadduError, 'failing read_batches') as caught:
            next(batches)
        assert isinstance(caught.exception.__cause__, OSError)
        with self.assertRaises(ld.LadduError) as caught:
            fmt.read(None, cache='streaming').weights()
        assert isinstance(caught.exception.__cause__, OSError)

    def test_declared_length_and_export_memory_budget(self):
        fmt = ld.io.FormatSpec(
            name='wrong-count',
            describe=lambda _, **__: ld.io.SourceInfo(ld.io.Schema(scalars=['x']), length=2),
            read_batches=lambda _, **__: iter([ld.io.EventBatch(scalars={'x': np.array([1.0])})]),
        )
        with self.assertRaisesRegex(ld.LadduError, 'declared length'):
            list(fmt.read(None, cache='streaming').batches(chunk_size=1))
        fmt = ld.io.FormatSpec(
            name='budget',
            describe=lambda _, **__: ld.io.SourceInfo(ld.io.Schema(scalars=['x'])),
            read_batches=lambda _, **__: iter([ld.io.EventBatch(scalars={'x': np.array([1.0])})]),
        )
        with self.assertRaisesRegex(ld.LadduError, 'one exported event'):
            fmt.read(None, memory=1).batches()

    def test_invalid_batches_and_bounds(self):
        for batch, message in [
            (ld.io.EventBatch(scalars={'wrong': np.array([1.0])}), 'schema'),
            (ld.io.EventBatch(scalars={'x': np.array([1.0, 2.0])}), 'chunk_size'),
        ]:
            fmt = ld.io.FormatSpec(
                name='invalid',
                describe=lambda _, **__: ld.io.SourceInfo(ld.io.Schema(scalars=['x'])),
                read_batches=lambda _, batch=batch, **__: iter([batch]),
            )
            with self.assertRaisesRegex(ld.LadduError, message):
                list(fmt.read(None, cache='streaming').batches(chunk_size=1))
        with self.assertRaises(ValueError):
            ld.io.EventBatch(p4s={'p': np.ones((2, 3))})
        with self.assertRaises(ld.LadduError):
            ld.io.EventBatch(scalars={'x': np.ones(2)}, length=3)
        with self.assertRaisesRegex(ValueError, 'positive'):
            sample().batches(chunk_size=0)

    def test_writer_early_return_and_failure(self):
        for writer, message in [
            (lambda *_args, **_kwargs: None, 'before consuming'),
            (self.fail_writer, 'writer broke'),
        ]:
            fmt = ld.io.FormatSpec(name='writer', write_batches=writer)
            with TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, message):
                fmt.write(sample(), Path(tmp) / 'output.dat')

    @staticmethod
    def fail_writer(*_args, **_kwargs):
        raise OSError('writer broke')

    def test_empty_and_builtin_output(self):
        with TemporaryDirectory() as tmp:
            for sink, read in [(ld.ParquetSink, ld.read_parquet), (ld.RootSink, ld.read_root)]:
                path = Path(tmp) / ('output.parquet' if sink == ld.ParquetSink else 'output.root')
                data = sample().bootstrap(seed=4)
                result = data.write_to(sink(path), chunk_size=1)
                assert result.events == 3
                np.testing.assert_array_equal(read(path).weights(), data.weights())
            path = Path(tmp) / 'empty.jsonl'
            empty = sample().subsample(0)
            assert record_format().write(empty, path, options=Options()).events == 0
            assert path.read_text() == ''

    def test_spec_capability_declaration(self):
        assert set(record_format(native_partitioning={'rows', 'contiguous'}).native_partitioning) == {
            'contiguous',
            'rows',
        }
        with self.assertRaisesRegex(ValueError, 'together'):
            ld.io.FormatSpec(name='bad', read_batches=records)

    def test_partitioning_helper_detects_overlap_and_omission(self):
        def reader(source, *, options, plan):
            rows = np.arange(len(source))
            if plan.partitioning == 'rows':
                rows = rows[rows % plan.nranks == plan.rank]
            else:
                rows = rows[len(source) * plan.rank // plan.nranks : len(source) * (plan.rank + 1) // plan.nranks]
            for start in range(0, len(rows), plan.chunk_size):
                yield ld.io.EventBatch(
                    scalars={'x': np.array(source, dtype=float)[rows[start : start + plan.chunk_size]]}
                )

        def describe(_, **__):
            return ld.io.SourceInfo(ld.io.Schema(scalars=['x']))

        good = ld.io.FormatSpec(
            name='native', describe=describe, read_batches=reader, native_partitioning={'rows', 'contiguous'}
        )
        for nranks in [1, 2, 3, 8]:
            for chunk_size in [1, 2, 3]:
                good.check_partitioning([1.0, 2.0, 3.0, 4.0, 5.0], nranks=nranks, chunk_size=chunk_size)
        bad = ld.io.FormatSpec(
            name='bad',
            describe=describe,
            read_batches=lambda source, **_: iter([ld.io.EventBatch(scalars={'x': np.array(source)})]),
            native_partitioning={'rows'},
        )
        with self.assertRaisesRegex(ValueError, 'disagrees|extra|omitted'):
            bad.check_partitioning([1.0, 2.0, 3.0], chunk_size=3)


if __name__ == '__main__':
    unittest.main()
