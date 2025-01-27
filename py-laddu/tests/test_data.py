from laddu import Dataset, Event, Mass, Vector3


def make_test_event() -> Event:
    return Event(
        [
            Vector3(0.0, 0.0, 8.747).with_mass(0.0),
            Vector3(0.119, 0.374, 0.222).with_mass(1.007),
            Vector3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vector3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        [Vector3(0.385, 0.022, 0.000)],
        0.48,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()])


def test_event_creation() -> None:
    event = make_test_event()
    assert len(event.p4s) == 4
    assert len(event.eps) == 1
    assert event.weight == 0.48


def test_event_p4_sum() -> None:
    event = make_test_event()
    sum = event.get_p4_sum([2, 3])
    assert sum[0] == event.p4s[2].px + event.p4s[3].px
    assert sum[1] == event.p4s[2].py + event.p4s[3].py
    assert sum[2] == event.p4s[2].pz + event.p4s[3].pz
    assert sum[3] == event.p4s[2].e + event.p4s[3].e


def test_dataset_size_check() -> None:
    dataset = Dataset([])
    assert len(dataset) == 0
    dataset = make_test_dataset()
    assert len(dataset) == 1


def test_dataset_weights() -> None:
    dataset = Dataset(
        [
            make_test_event(),
            Event(
                make_test_event().p4s,
                make_test_event().eps,
                0.52,
            ),
        ]
    )
    weights = dataset.weights
    assert len(weights) == 2
    assert weights[0] == 0.48
    assert weights[1] == 0.52
    assert dataset.weighted_len() == 1.0


def test_dataset_sum() -> None:
    dataset = make_test_dataset()
    dataset2 = Dataset([Event(make_test_event().p4s, make_test_event().eps, 0.52)])
    dataset_sum = dataset + dataset2
    assert dataset_sum[0].weight == dataset[0].weight
    assert dataset_sum[1].weight == dataset2[0].weight
    list_sum = sum([dataset, dataset2])
    assert list_sum != 0  # TODO: Pray the Python devs sort this out someday
    assert list_sum[0].weight == dataset_sum[0].weight
    assert list_sum[1].weight == dataset_sum[1].weight


# TODO: Dataset::filter requires free-threading or some other workaround (or maybe we make a non-parallel method)


def test_binned_dataset() -> None:
    dataset = Dataset(
        [
            Event(
                [Vector3(0.0, 0.0, 1.0).with_mass(1.0)],
                [],
                1.0,
            ),
            Event(
                [Vector3(0.0, 0.0, 2.0).with_mass(2.0)],
                [],
                2.0,
            ),
        ]
    )

    mass = Mass([0])
    binned = dataset.bin_by(mass, 2, (0.0, 3.0))

    assert binned.bins == 2
    assert len(binned.edges) == 3
    assert binned.edges[0] == 0.0
    assert binned.edges[2] == 3.0
    assert len(binned[0]) == 1
    assert binned[0].weighted_len() == 1.0
    assert len(binned[1]) == 1
    assert binned[1].weighted_len() == 2.0


def test_dataset_bootstrap() -> None:
    dataset = Dataset(
        [
            make_test_event(),
            Event(
                make_test_event().p4s,
                make_test_event().eps,
                1.0,
            ),
        ]
    )
    assert dataset[0].weight != dataset[1].weight

    bootstrapped = dataset.bootstrap(42)
    assert len(bootstrapped) == len(dataset)
    assert bootstrapped[0].weight == bootstrapped[1].weight

    empty_dataset = Dataset([])
    empty_bootstrap = empty_dataset.bootstrap(42)
    assert len(empty_bootstrap) == 0


def test_event_display() -> None:
    event = make_test_event()
    display_string = str(event)
    assert (
        display_string
        == 'Event:\n  p4s:\n    [e = 8.74700; p = (0.00000, 0.00000, 8.74700); m = 0.00000]\n    [e = 1.10334; p = (0.11900, 0.37400, 0.22200); m = 1.00700]\n    [e = 3.13671; p = (-0.11200, 0.29300, 3.08100); m = 0.49800]\n    [e = 5.50925; p = (-0.00700, -0.66700, 5.44600); m = 0.49800]\n  eps:\n    [0.385, 0.022, 0]\n  weight:\n    0.48\n'
    )
