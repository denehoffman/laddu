# ruff: noqa: S101, PLR2004

from itertools import pairwise

import laddu as ld
import numpy as np
import pytest


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_integer_columns_preserve_exact_dtype_and_owned_storage(dtype) -> None:
    info = np.iinfo(dtype)
    ids = np.array([info.min, info.max, info.max - 1], dtype=dtype)
    expected = ids.copy()
    dataset = ld.Dataset.from_arrays(
        p4s={},
        scalars={'x': [1.0, 2.0, 3.0]},
        columns={'event_id': ids},
        weights=[0.5, 1.5, 2.0],
    )
    ids[0] = 1
    result = dataset.column('event_id')
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.dtype(dtype)
    assert not result.flags.writeable
    try:
        result.setflags(write=True)
    except ValueError:
        pass  # NumPy may forbid enabling writes for a capsule-owned copy.
    else:
        result[0] = 2
    np.testing.assert_array_equal(dataset.column('event_id'), expected)
    np.testing.assert_array_equal(dataset.column('x'), [1.0, 2.0, 3.0])
    assert dataset.column_names() == ['x', 'event_id']
    assert dataset.schema.columns == {'event_id': np.dtype(dtype).name}


def test_integer_columns_align_with_values_through_views() -> None:
    x = np.arange(20, dtype=np.float64)
    dataset = ld.Dataset.from_arrays(
        p4s={}, scalars={'x': x}, columns={'id': np.arange(20, dtype=np.uint64) + 2**63}, weights=x + 1
    )
    bins = dataset.bin_by(ld.scalar('x'), bins=ld.Bin([0.0, 5.0, 10.0, 20.0]))
    for view in [
        dataset,
        dataset.subsample(0.5, seed=13),
        dataset.bootstrap(seed=7),
        dataset.select(ld.scalar('x') > 8),
        *[bin_view.dataset for bin_view in bins],
    ]:
        values = view.evaluate({'x': ld.scalar('x')}, real=True)['x']
        np.testing.assert_array_equal(view.column('id') - np.uint64(2**63), values)
        np.testing.assert_array_equal(view.column('x'), values)
    np.testing.assert_array_equal(
        dataset.subsample(0.5, seed=13).weights(), dataset.subsample(0.5, seed=13).column('x') + 1
    )


@pytest.mark.parametrize(
    ('description', 'canonical'), [('u8', 'uint8'), ('i64', 'int64'), (np.dtype('u8'), 'uint64'), (np.uint16, 'uint16')]
)
def test_schema_integer_dtype_descriptors(description, canonical) -> None:
    assert ld.Schema(p4s=[], scalars=[], columns={'id': description}).columns == {'id': canonical}


@pytest.mark.parametrize('values', [[1.0], [True], np.array([1], dtype=object), [[1, 2]]])
def test_reject_non_integer_or_non_vector_columns(values) -> None:
    with pytest.raises((ValueError, TypeError)):
        ld.Dataset.from_arrays(p4s={}, scalars={}, columns={'id': values})


def test_empty_and_integer_only_dataset() -> None:
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={}, columns={'id': np.array([4, 5], dtype=np.uint8)})
    assert len(dataset) == 2
    empty = ld.Dataset.from_arrays(p4s={}, scalars={}, columns={'id': np.array([], dtype=np.uint64)})
    assert empty.column('id').shape == (0,)
    assert empty.column('id').dtype == np.dtype('uint64')
    with pytest.raises(ld.LadduError):
        empty.column('missing')


def test_batch_factory_reorders_exact_columns_and_exports_them() -> None:
    def batches(**_plan):
        for start in [0, 2]:
            yield {
                'p4s': {},
                'scalars': {'x': np.arange(start, start + 2, dtype=float)},
                'columns': {
                    'run': np.array([7, 7], dtype=np.int16),
                    'id': np.arange(start, start + 2, dtype=np.uint64) + 2**63,
                },
            }

    dataset = ld.Dataset.from_batches(
        batches,
        schema={'p4s': [], 'scalars': ['x'], 'columns': {'id': 'u64', 'run': 'i16'}},
        length=4,
        cache='streaming',
    )
    np.testing.assert_array_equal(dataset.column('id'), np.arange(4, dtype=np.uint64) + 2**63)
    exported = list(dataset.batches(chunk_size=3))
    assert exported[0].schema.columns == {'id': 'uint64', 'run': 'int16'}
    assert list(exported[0].columns) == ['id', 'run']
    np.testing.assert_array_equal(np.concatenate([b.columns['id'] for b in exported]), dataset.column('id'))


def test_custom_format_carries_exact_schema_and_batches(tmp_path) -> None:
    schema = ld.Schema(columns={'id': 'u64', 'run': 'i8'})

    def reader(path, **_kwargs):
        with np.load(path) as data:
            yield ld.io.EventBatch(columns={'id': data['id'], 'run': data['run']}, weights=data['weights'])

    def writer(path, batches, *, schema, **_kwargs):
        assert schema.columns == {'id': 'uint64', 'run': 'int8'}
        batches = list(batches)
        with path.open('wb') as file:
            np.savez(
                file,
                allow_pickle=False,
                **{name: np.concatenate([b.columns[name] for b in batches]) for name in schema.columns},
                weights=np.concatenate([b.weights for b in batches]),
            )

    fmt = ld.io.FormatSpec(
        name='integer-test',
        describe=lambda *_args, **_kwargs: ld.io.SourceInfo(ld.Schema(columns=schema.columns, weights=True)),
        read_batches=reader,
        write_batches=writer,
    )
    dataset = ld.Dataset.from_arrays(
        p4s={},
        scalars={},
        columns={'id': np.array([2**63, 2**63 + 1], dtype=np.uint64), 'run': np.array([-2, 7], dtype=np.int8)},
    )
    path = tmp_path / 'custom.bin'
    fmt.write(dataset, path, chunk_size=1)
    restored = fmt.read(path)
    for name in schema.columns:
        np.testing.assert_array_equal(restored.column(name), dataset.column(name))
        assert restored.column(name).dtype == dataset.column(name).dtype


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_parquet_integer_roundtrip(dtype, tmp_path) -> None:
    info = np.iinfo(dtype)
    ids = np.array([info.min, info.max - 1, info.max], dtype=dtype)
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.arange(3, dtype=float)}, columns={'id': ids})
    path = tmp_path / 'integer.parquet'
    dataset.write_to(ld.ParquetSink(path, precision='f32'), chunk_size=2)
    restored = ld.read_parquet(path, cache='streaming')
    np.testing.assert_array_equal(restored.column('id'), ids)
    assert restored.column('id').dtype == np.dtype(dtype)
    np.testing.assert_array_equal(restored.column('x'), [0, 1, 2])


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_root_integer_roundtrip(dtype, tmp_path) -> None:
    info = np.iinfo(dtype)
    ids = np.array([info.min, info.max - 1, info.max], dtype=dtype)
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.arange(3, dtype=float)}, columns={'id': ids})
    path = tmp_path / 'integer.root'
    dataset.write_to(ld.RootSink(path, precision='f32'), chunk_size=2)
    restored = ld.read_root(path, cache='streaming')
    np.testing.assert_array_equal(restored.column('id'), ids)
    assert restored.column('id').dtype == np.dtype(dtype)
    np.testing.assert_array_equal(restored.column('x'), [0, 1, 2])


@pytest.mark.parametrize(
    ('sink_type', 'reader', 'suffix'),
    [(ld.ParquetSink, ld.read_parquet, 'parquet'), (ld.RootSink, ld.read_root, 'root')],
)
def test_native_explicit_schema_projects_columns(sink_type, reader, suffix, tmp_path) -> None:
    dataset = ld.Dataset.from_arrays(
        p4s={},
        scalars={'x': [2.0, 3.0]},
        columns={'id': np.array([2**63, 2**63 + 1], dtype=np.uint64), 'other': np.array([4, 5], dtype=np.int8)},
    )
    path = tmp_path / ('input.' + suffix)
    dataset.write_to(sink_type(path))
    restored = reader(path, schema=ld.Schema(columns={'id': 'u64'}))
    assert restored.column_names() == ['id']
    np.testing.assert_array_equal(restored.column('id'), dataset.column('id'))


@pytest.mark.parametrize('name', ['x', 'p', 'p_e', 'weight', ''])
def test_integer_names_do_not_collide_with_event_fields(name) -> None:
    with pytest.raises(ld.LadduError):
        ld.Dataset.from_arrays(
            p4s={'p': np.zeros((1, 4))}, scalars={'x': [1.0]}, columns={name: np.array([1], dtype=np.uint64)}
        )


def test_integer_shapes_lengths_and_batch_dtypes_are_validated() -> None:
    with pytest.raises((ValueError, ld.LadduError)):
        ld.Dataset.from_arrays(p4s={}, scalars={'x': [1.0]}, columns={'id': np.array([1, 2], dtype=np.uint8)})
    for columns in [{}, {'id': np.array([1], dtype=np.int64)}, {'id': np.array([1], dtype=np.uint32)}]:
        dataset = ld.Dataset.from_batches(
            lambda columns=columns, **_: iter([{'p4s': {}, 'scalars': {}, 'columns': columns}]),
            schema=ld.Schema(columns={'id': 'u64'}),
            cache='streaming',
        )
        with pytest.raises(ld.LadduError, match='does not match declared schema'):
            dataset.column('id')


@pytest.mark.parametrize(
    ('sink_type', 'reader', 'suffix'),
    [(ld.ParquetSink, ld.read_parquet, 'parquet'), (ld.RootSink, ld.read_root, 'root')],
)
def test_native_files_require_identical_integer_dtype(sink_type, reader, suffix, tmp_path) -> None:
    for index, dtype in enumerate([np.uint64, np.uint32]):
        data = ld.Dataset.from_arrays(p4s={}, scalars={}, columns={'id': np.array([1], dtype=dtype)})
        data.write_to(sink_type(tmp_path / (str(index) + '.' + suffix)))
    with pytest.raises(ld.LadduError):
        reader(tmp_path / ('*.' + suffix))


def test_replay_alignment_is_independent_of_batch_boundaries() -> None:
    traversals = []
    ids = np.array([2**63, 2**63, 2**63 + 1, 2**63 + 2, 2**63 + 2], dtype=np.uint64)
    x = np.arange(5, dtype=float)

    def batches(**_plan):
        traversals.append(1)
        edges = [0, 2, 5] if len(traversals) % 2 else [0, 1, 3, 5]
        for start, end in pairwise(edges):
            yield {
                'p4s': {},
                'scalars': {'x': x[start:end]},
                'columns': {'id': ids[start:end], 'run': np.ones(end - start, dtype=np.uint16)},
                'weights': x[start:end] + 1,
            }

    dataset = ld.Dataset.from_batches(
        batches, schema=ld.Schema(scalars=['x'], columns={'id': 'u64', 'run': 'u16'}, weights=True), cache='streaming'
    )
    view = dataset.select(ld.scalar('x') >= 1)
    np.testing.assert_array_equal(view.column('id'), ids[1:])
    np.testing.assert_array_equal(view.evaluate({'x': ld.scalar('x')}, real=True)['x'], x[1:])
    np.testing.assert_array_equal(view.weights(), x[1:] + 1)
    exported = list(view.batches(chunk_size=1))
    resident = ld.Dataset.from_batches(lambda **_: iter(exported), schema=view.schema, cache='resident')
    np.testing.assert_array_equal(resident.column('id'), ids[1:])


@pytest.mark.parametrize(
    ('sink_type', 'reader', 'suffix'),
    [(ld.ParquetSink, ld.read_parquet, 'parquet'), (ld.RootSink, ld.read_root, 'root')],
)
def test_native_empty_integer_dataset(sink_type, reader, suffix, tmp_path) -> None:
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={}, columns={'id': np.array([], dtype=np.uint64)})
    path = tmp_path / ('empty.' + suffix)
    dataset.write_to(sink_type(path))
    restored = reader(path)
    assert restored.column('id').shape == (0,)
    assert restored.column('id').dtype == np.dtype(np.uint64)
