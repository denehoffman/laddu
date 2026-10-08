# ruff: noqa: S101

from collections import UserDict

import laddu as ld
import numpy as np
import pytest


def test_named_expressions_share_one_ordered_source_traversal() -> None:
    traversals = []

    def batches(**_plan):
        traversals.append(True)
        for values in ([1.0, 2.0], [4.0]):
            yield {'p4s': {}, 'scalars': {'x': np.array(values)}}

    dataset = ld.Dataset.from_batches(
        batches,
        schema={'p4s': [], 'scalars': ['x']},
        cache='streaming',
    )
    traversals.clear()
    result = dataset.evaluate({'x': ld.scalar('x'), 'twice': 2.0 * ld.scalar('x')})
    assert list(result) == ['x', 'twice']
    np.testing.assert_array_equal(result['x'], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(result['twice'], [2.0, 4.0, 8.0])
    assert len(traversals) == 1


@pytest.mark.parametrize('backend', ['cpu', 'jit'])
@pytest.mark.parametrize('precision', ['f32', 'f64'])
def test_mapping_matches_separate_real_and_complex_evaluation(backend, precision) -> None:
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.25, 2.0, -3.0]})
    execution = ld.Execution(backend, precision=precision)
    expressions = UserDict({'complex': ld.scalar('x') + 2j, 'real': ld.scalar('x') * ld.scalar('x')})
    result = dataset.evaluate(expressions, execution=execution)
    for name, expr in expressions.items():
        np.testing.assert_array_equal(result[name], dataset.evaluate(expr, execution=execution))
        assert result[name].dtype == np.dtype('complex128')
    real = dataset.evaluate({'square': expressions['real']}, execution=execution, real=True)
    np.testing.assert_array_equal(real['square'], [0.0625, 4.0, 9.0])
    assert real['square'].dtype == np.dtype('float64')


def test_empty_mapping_reads_nothing_and_empty_dataset_retains_output_shapes() -> None:
    def batches(**_plan):
        message = 'empty mapping must not read the source'
        raise AssertionError(message)

    dataset = ld.Dataset.from_batches(batches, schema={'p4s': [], 'scalars': ['x']})
    assert dataset.evaluate({}) == {}
    empty = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.array([], dtype=float)})
    assert empty.evaluate({'x': ld.scalar('x')})['x'].shape == (0,)
    assert empty.evaluate({'x': ld.scalar('x')}, real=True)['x'].dtype == np.dtype('float64')


def test_mapping_validation_rejects_invalid_input_and_complex_real_projection() -> None:
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': [1.0]})
    with pytest.raises(ld.LadduError, match='real-valued'):
        dataset.evaluate({'x': ld.scalar('x') + 1j}, real=True)
    with pytest.raises(TypeError):
        dataset.evaluate({'x': object()})  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        dataset.evaluate([ld.scalar('x')])  # ty: ignore[invalid-argument-type]
