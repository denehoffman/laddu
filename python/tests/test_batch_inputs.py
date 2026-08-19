# ruff: noqa: S101, PT027

import unittest

import laddu as ld
import numpy as np

EXPECTED_EVENTS = 3
MIN_TRAVERSALS = 2


class BatchInputTests(unittest.TestCase):
    def test_batch_factory_is_reusable_and_receives_read_plan(self) -> None:
        plans: list[dict[str, int | None]] = []

        def batches(**plan: int | None):
            plans.append(plan)
            yield {
                'p4s': {
                    'beam': np.array(
                        [[5.0, 0.0, 0.0, 5.0], [6.0, 0.0, 0.0, 6.0]],
                    ),
                },
                'scalars': {'run': np.array([1.0, 2.0])},
                'weights': np.array([0.5, 1.5]),
            }
            yield {
                'p4s': {'beam': np.array([[7.0, 0.0, 0.0, 7.0]])},
                'scalars': {'run': np.array([3.0])},
                'weights': np.array([2.0]),
            }

        dataset = ld.Dataset.from_batches(
            batches,
            schema={'p4s': ['beam'], 'scalars': ['run'], 'weights': True},
            length=3,
        )

        assert dataset.p4_names() == ['beam']
        assert dataset.scalar_names() == ['run']
        assert len(dataset) == EXPECTED_EVENTS
        np.testing.assert_allclose(dataset.weights(), [0.5, 1.5, 2.0])
        assert len(plans) >= MIN_TRAVERSALS
        for plan in plans:
            assert set(plan) == {'chunk_size', 'rank', 'nranks'}
            assert plan['rank'] == 0
            assert plan['nranks'] == 1

    def test_batch_schema_is_validated_during_traversal(self) -> None:
        def batches(**_plan: int | None):
            yield {
                'p4s': {},
                'scalars': {'unexpected': np.array([1.0])},
            }

        dataset = ld.Dataset.from_batches(
            batches,
            schema={'p4s': [], 'scalars': ['expected']},
        )

        with self.assertRaisesRegex(ld.LadduError, 'does not match declared schema'):
            dataset.weights()

    def test_batch_factory_must_be_callable(self) -> None:
        with self.assertRaisesRegex(TypeError, 'batch_factory must be callable'):
            ld.Dataset.from_batches(
                object(),
                schema={'p4s': [], 'scalars': []},
            )


if __name__ == '__main__':
    unittest.main()
