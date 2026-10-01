"""Static contract for independently typed downstream format callbacks."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import assert_type

import numpy as np
from laddu import Dataset, MemoryBudget
from laddu.io import EventBatch, FormatSpec, ReadPlan, Schema, SourceInfo, WritePlan, WriteResult, convert
from numpy.typing import NDArray


@dataclass(frozen=True)
class Options:
    particle: str = 'beam'


def input_columns(vectors: NDArray[np.float64]) -> EventBatch:
    return EventBatch(p4s={'beam': vectors}, scalars={}, weights=[1.0])


def describe(source: Path, *, options: Options) -> SourceInfo:
    del source
    return SourceInfo(Schema(p4s=[options.particle]))


def read_batches(source: Path, *, options: Options, plan: ReadPlan) -> Iterable[EventBatch]:
    del source, options, plan
    return []


def write_batches(
    target: Path, batches: Iterable[EventBatch], *, options: Options, schema: Schema, plan: WritePlan
) -> None:
    del target, options, schema, plan
    for batch in batches:
        assert_type(batch, EventBatch)


def use_format(source: Path, target: Path) -> None:
    spec = FormatSpec(
        name='typed-records',
        describe=describe,
        read_batches=read_batches,
        write_batches=write_batches,
        native_partitioning={'rows', 'contiguous'},
    )
    options = Options()
    data = spec.read(source, options=options, memory=MemoryBudget('1 GiB'))
    assert_type(data, Dataset)
    assert_type(spec.write(data, target, options=options), WriteResult)
    assert_type(convert(source, target, reader=spec, writer=spec, read_options=options, memory=1024), WriteResult)
