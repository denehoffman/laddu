from __future__ import annotations

import sys
import sysconfig
import unittest
from concurrent.futures import ThreadPoolExecutor

import laddu as ld
import numpy as np


class FreeThreadingTests(unittest.TestCase):
    @staticmethod
    def model_fixture() -> tuple[ld.Model, ld.Dataset, np.ndarray]:
        values = np.linspace(-2.0, 2.0, 257)
        dataset = ld.Dataset.from_arrays(p4s={}, scalars={'x': values})
        scale = ld.parameter('scale', initial=2.0)
        return ld.Model(ld.scalar('x') * scale), dataset, values

    @unittest.skipUnless(
        sysconfig.get_config_var('Py_GIL_DISABLED'),
        'requires a free-threaded CPython build',
    )
    def test_import_does_not_enable_gil(self) -> None:
        is_gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)
        if is_gil_enabled():
            self.fail('importing laddu enabled the GIL')

    def test_shared_model_evaluates_concurrently(self) -> None:
        model, dataset, expected_gradient = self.model_fixture()
        expected_values = 2.0 * expected_gradient

        def evaluate() -> tuple[np.ndarray, np.ndarray]:
            values, gradients = model.value_and_gradient(dataset, real=True)
            values_array = np.asarray(values)
            gradients_array = np.asarray(gradients)
            return values_array, gradients_array[:, 0]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: evaluate(), range(24)))

        for values, gradients in results:
            np.testing.assert_allclose(values, expected_values)
            np.testing.assert_allclose(gradients, expected_gradient)


if __name__ == '__main__':
    unittest.main()
