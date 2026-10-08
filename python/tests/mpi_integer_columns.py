# ruff: noqa: S101
"""Run with mpiexec -n 3 python mpi_integer_columns.py OUTPUT_DIRECTORY."""

import sys
from pathlib import Path

import laddu as ld
import numpy as np
import pytest
from mpi4py import MPI

world = MPI.COMM_WORLD
directory = Path(sys.argv[1])
execution = ld.Execution('cpu', partitioning='rows')
local = ld.Execution('cpu', mpi=False)
dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
columns = {name: np.array([np.iinfo(name).max - i for i in range(7)], dtype=name) for name in dtypes}
dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.arange(7, dtype=float)}, columns=columns)

for sink_type, reader, suffix in [(ld.ParquetSink, ld.read_parquet, 'parquet'), (ld.RootSink, ld.read_root, 'root')]:
    path = directory / ('integers.' + suffix)
    dataset.write_to(sink_type(path), output='single', execution=execution, chunk_size=2)
    if world.rank == 0:
        restored = reader(path)
        x = restored.column('x').astype(int)
        for name in dtypes:
            np.testing.assert_array_equal(restored.column(name), columns[name][x])
            assert restored.column(name).dtype == np.dtype(name)
    world.Barrier()

# Rank agreement must include both typed names and exact dtypes.
spec = ld.io.FormatSpec(
    name='dtype-agreement',
    describe=lambda *_args, **_kwargs: ld.io.SourceInfo(ld.Schema(columns={'id': 'u64' if world.rank == 0 else 'u32'})),
    read_batches=lambda *_args, **_kwargs: iter(()),
)
with pytest.raises(ValueError, match='disagree'):
    spec.read(None, execution=execution)

if world.rank == 0:
    print('MPI integer columns passed')
