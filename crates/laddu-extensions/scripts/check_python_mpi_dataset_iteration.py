#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import laddu as ld
from laddu import Dataset, Event, Vec3

P4_NAMES = ['beam']
PARQUET_P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']


def make_event(weight: float) -> Event:
    return Event(
        [Vec3(0.0, 0.0, weight).with_mass(0.0)],
        [],
        weight,
        p4_names=P4_NAMES,
        aux_names=[],
    )


def check_manual_dataset() -> None:
    rank = ld.mpi.get_rank() or 0
    size = ld.mpi.get_size() or 1
    events = [make_event(float(index + 1)) for index in range(size)]
    dataset = Dataset(events, p4_names=P4_NAMES, aux_names=[])

    default_weights = [event.weight for event in dataset]
    local_weights = [event.weight for event in dataset.iter_local()]
    global_weights = [event.weight for event in dataset.iter_global()]
    stored_weights = [event.weight for event in dataset.events]
    stored_local_weights = [event.weight for event in dataset.events_local]

    assert dataset.n_events == size
    assert dataset.n_events_local == 1
    assert dataset.n_events_weighted == sum(float(index + 1) for index in range(size))
    assert dataset.n_events_weighted_local == float(rank + 1)
    assert sorted(default_weights) == [float(index + 1) for index in range(size)]
    assert local_weights == [float(rank + 1)]
    assert sorted(global_weights) == [float(index + 1) for index in range(size)]
    assert sorted(stored_weights) == [float(index + 1) for index in range(size)]
    assert stored_local_weights == [float(rank + 1)]
    assert list(dataset.weights) == default_weights
    assert list(dataset.weights_local) == local_weights


def check_parquet_dataset() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / 'py-laddu' / 'tests' / 'data_files' / 'data_f32.parquet'
    dataset = ld.io.read_parquet(data_path, p4s=PARQUET_P4_NAMES)

    default_weights = [event.weight for event in dataset]
    local_weights = [event.weight for event in dataset.iter_local()]
    global_weights = [event.weight for event in dataset.iter_global()]
    stored_weights = [event.weight for event in dataset.events]
    stored_local_weights = [event.weight for event in dataset.events_local]

    assert default_weights == global_weights
    assert default_weights == stored_weights
    assert len(global_weights) == dataset.n_events
    assert len(default_weights) == len(stored_weights)
    assert len(local_weights) == dataset.n_events_local
    assert len(stored_local_weights) == dataset.n_events_local
    assert len(local_weights) <= len(global_weights)
    assert list(dataset.weights) == default_weights
    assert list(dataset.weights_local) == local_weights
    assert dataset.n_events_weighted == sum(default_weights)
    assert dataset.n_events_weighted_local == sum(local_weights)


def main() -> None:
    if not ld.mpi.is_mpi_available():
        msg = 'laddu MPI backend is not available'
        raise RuntimeError(msg)

    with ld.mpi.MPI(trigger=True):
        check_manual_dataset()
        check_parquet_dataset()

        if ld.mpi.is_root():
            print('python-mpi-dataset-iteration: ok')


if __name__ == '__main__':
    main()
