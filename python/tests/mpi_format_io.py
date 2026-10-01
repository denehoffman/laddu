# ruff: noqa: S101, PLR2004, ARG001, EM101
"""Run with mpiexec -n 3 python mpi_format_io.py OUTPUT_DIRECTORY."""

import json
import sys
from pathlib import Path

import laddu as ld
import numpy as np
import pytest
from mpi4py import MPI

world = MPI.COMM_WORLD
directory = Path(sys.argv[1])
local = ld.Execution('cpu', mpi=False)
execution = ld.Execution('cpu', partitioning='rows')
plans = []
closed = []


def describe(source, *, options):
    return ld.io.SourceInfo(ld.io.Schema(scalars=['x'], weights=True), length=options.get('length'))


def read(source, *, options, plan):
    plans.append((plan.rank, plan.nranks, plan.partitioning))
    ids = np.arange(len(source))
    if plan.nranks > 1:
        if plan.partitioning == 'rows':
            ids = ids[ids % plan.nranks == plan.rank]
        elif plan.partitioning == 'file_groups':
            ids = ids[(ids // 2) % plan.nranks == plan.rank]
        else:
            ids = ids[len(source) * plan.rank // plan.nranks : len(source) * (plan.rank + 1) // plan.nranks]
    try:
        for start in range(0, len(ids), plan.chunk_size):
            if options.get('fail_rank') == plan.rank and plan.nranks > 1:
                raise OSError('injected reader failure')
            rows = ids[start : start + plan.chunk_size]
            yield ld.io.EventBatch(
                scalars={'x': np.array(source)[rows].astype(float)},
                weights=np.where(rows % 2 == 0, 1.0, -1.0),
                row_ids=rows.tolist() if plan.partitioning == 'file_groups' else None,
            )
    finally:
        closed.append(True)


def write(target, batches, *, options, schema, plan):
    if options.get('early'):
        return
    with target.open('w') as file:
        for batch in batches:
            if options.get('fail_rank') == plan.rank:
                raise OSError('injected writer failure')
            for x, weight in zip(batch.scalars['x'], batch.weights, strict=True):
                file.write(json.dumps([x.item(), weight.item()]) + '\n')


def fmt(*, native=False):
    return ld.io.FormatSpec(
        name='mpi-records',
        describe=describe,
        read_batches=read,
        write_batches=write,
        native_partitioning={'rows', 'contiguous', 'file_groups'} if native else (),
    )


def values(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


for native in [False, True]:
    for partition in ['rows', 'contiguous', 'file_groups']:
        execution = ld.Execution('cpu', partitioning=partition)
        if not native and partition == 'file_groups':
            data = fmt().read(list(range(7)), options={}, cache='streaming', execution=execution)
            with pytest.raises(ld.LadduError, match='requires declared native'):
                fmt().write(data, directory / 'unsupported.jsonl', options={}, execution=execution)
            continue
        for count in [0, 2, 7]:
            spec = fmt(native=native)
            plans.clear()
            options = {'length': count} if count == 2 else {}
            data = spec.read(list(range(count)), options=options, cache='streaming', execution=execution)
            data = data.bootstrap(seed=17).subsample(0.8, seed=9).select(ld.scalar('x') >= 0)
            serial = [
                (float(x), float(w))
                for b in data.batches(chunk_size=2, execution=local)
                for x, w in zip(b.scalars['x'], b.weights, strict=True)
            ]
            target = directory / f'{native}-{partition}-{count}.jsonl'
            result = spec.write(data, target, options={}, output='single', chunk_size=2, execution=execution)
            assert result.events == len(serial)
            if world.rank == 0:
                expected = []
                for rank in range(world.size):
                    for x, w in serial:
                        if partition == 'rows':
                            assigned = int(x) % world.size
                        elif partition == 'file_groups':
                            assigned = (int(x) // 2) % world.size
                        else:
                            assigned = next(
                                r
                                for r in range(world.size)
                                if count * r // world.size <= int(x) < count * (r + 1) // world.size
                            )
                        if assigned == rank:
                            expected.append([x, w])
                assert values(target) == expected
            if not native:
                assert all(rank == 0 and ranks == 1 for rank, ranks, _ in plans)
            result = spec.write(data, target, options={}, output='sharded', chunk_size=1, execution=execution)
            assert result.events == len(serial)
            assert len(result.paths) == world.size
            assert sum(result.counts) == len(serial)
            world.Barrier()

spec = fmt(native=True)
spec.check_partitioning(list(range(7)), options={}, nranks=world.size, chunk_size=2)
execution = ld.Execution('cpu', partitioning='rows')
target = directory / 'failure.jsonl'
for output in ['single', 'sharded']:
    for options in [{'early': True}, {'fail_rank': 0}, {'fail_rank': world.size - 1}]:
        if output == 'single' and options.get('fail_rank') not in [None, 0]:
            continue
        data = spec.read(list(range(7)), options={}, cache='streaming', execution=execution)
        with pytest.raises((RuntimeError, ld.LadduError), match=r'writer|consuming'):
            spec.write(data, target, options=options, output=output, chunk_size=1, execution=execution)
    data = spec.read(list(range(7)), options={'fail_rank': world.size - 1}, cache='streaming', execution=execution)
    with pytest.raises((RuntimeError, ld.LadduError), match='reader failure'):
        spec.write(data, target, options={}, output=output, chunk_size=1, execution=execution)

# A successful collective after failures detects stranded producers.
data = spec.read(list(range(7)), options={}, cache='streaming', execution=execution)
assert spec.write(data, target, options={}, output='single', chunk_size=1, execution=execution).events == 7
assert closed
for sink_type, reader in [(ld.ParquetSink, ld.read_parquet), (ld.RootSink, ld.read_root)]:
    target = directory / ('builtin.parquet' if sink_type == ld.ParquetSink else 'builtin.root')
    assert data.write_to(sink_type(target), output='single', execution=execution, chunk_size=2).events == 7
    if world.rank == 0:
        assert len(reader(target)) == 7
    world.Barrier()
if world.rank == 0:
    print('MPI format contracts passed')
