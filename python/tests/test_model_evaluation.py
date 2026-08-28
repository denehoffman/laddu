# ruff: noqa: PT027, S101

import unittest

import laddu as ld
import numpy as np


class ModelEvaluationTests(unittest.TestCase):
    def test_parameter_only_complex_value(self) -> None:
        z = ld.complex(ld.parameter('x', initial=2.0), ld.parameter('y', initial=3.0))
        model = ld.Model(z * z + 1.0)

        value = model.evaluate()

        assert isinstance(value, complex)
        np.testing.assert_allclose(value, -4.0 + 12.0j, rtol=0, atol=1e-12)

    def test_parameter_only_complex_gradient(self) -> None:
        z = ld.complex(ld.parameter('x', initial=2.0), ld.parameter('y', initial=3.0))
        model = ld.Model(z * z + 1.0)

        value, gradient = model.value_and_gradient()

        assert isinstance(value, complex)
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert model.parameter_names == ['x', 'y']
        np.testing.assert_allclose(value, -4.0 + 12.0j, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradient, [4.0 + 6.0j, -6.0 + 4.0j], rtol=0, atol=1e-12)

    def test_explicit_none_and_real_output(self) -> None:
        z = ld.complex(ld.parameter('x', initial=2.0), ld.parameter('y', initial=3.0))
        model = ld.Model(z * z + 1.0)

        value = model.evaluate(None, real=True)
        gradient_value, gradient = model.value_and_gradient(None, real=True)

        assert isinstance(value, float)
        assert isinstance(gradient_value, float)
        assert gradient.dtype == np.float64
        assert gradient.shape == (2,)
        np.testing.assert_allclose([value, gradient_value], [-4.0, -4.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradient, [4.0, -6.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.evaluate(None), -4.0 + 12.0j, rtol=0, atol=1e-12)

    def test_constant_has_an_empty_gradient(self) -> None:
        model = ld.Model(ld.complex(2.0, 3.0) * ld.complex(2.0, 3.0) + 1.0)

        for real in (False, True):
            with self.subTest(real=real):
                value, gradient = model.value_and_gradient(real=real)
                assert isinstance(value, float if real else complex)
                assert gradient.shape == (0,)
                assert gradient.dtype == (np.float64 if real else np.complex128)
                np.testing.assert_allclose(value, -4.0 if real else -4.0 + 12.0j, rtol=0, atol=1e-12)

    def test_non_scalar_results_are_rejected(self) -> None:
        x = ld.parameter('x', initial=2.0)
        for expression in (ld.vector([x, 3.0]), ld.matrix([[x, 0.0], [0.0, 3.0]])):
            model = ld.Model(expression)
            for evaluate in (model.evaluate, model.value_and_gradient):
                with (
                    self.subTest(shape=expression.shape, method=evaluate.__name__),
                    self.assertRaisesRegex(ld.LadduError, 'scalar'),
                ):
                    evaluate()

    def test_event_inputs_require_a_dataset(self) -> None:
        channel = ld.Channel('test', edges=[ld.Edge('p', p4='p')], vertices=[])
        scale = ld.parameter('scale', initial=2.0)
        cases = (
            (ld.scalar('mass') * scale, 'mass'),
            (channel.s('p') * scale, 'p'),
        )
        for expression, name in cases:
            model = ld.Model(expression)
            for evaluate in (model.evaluate, model.value_and_gradient):
                with (
                    self.subTest(input=name, method=evaluate.__name__),
                    self.assertRaisesRegex(ld.LadduError, f'requires event input .*{name}.*provide a dataset'),
                ):
                    evaluate()

    def test_parameter_defaults_overrides_and_fixed_values(self) -> None:
        x = ld.parameter('x', initial=2.0)
        y = ld.parameter('y', initial=3.0)
        offset = ld.parameter('offset', fixed=5.0)
        model = ld.Model(x * x + ld.complex(offset, y))
        assert model.parameter_names == ['x', 'y']

        cases = (
            (None, 9.0 + 3.0j, [4.0, 1.0j]),
            ({'y': 4.0}, 9.0 + 4.0j, [4.0, 1.0j]),
            ({'y': 4.0, 'x': 3.0}, 14.0 + 4.0j, [6.0, 1.0j]),
            ([3.0, 4.0], 14.0 + 4.0j, [6.0, 1.0j]),
            ((3.0, 4.0), 14.0 + 4.0j, [6.0, 1.0j]),
            (np.array([3.0, 4.0], dtype=np.float32), 14.0 + 4.0j, [6.0, 1.0j]),
            (np.array([3.0, 4.0], dtype=np.float64), 14.0 + 4.0j, [6.0, 1.0j]),
        )
        for parameters, expected, expected_gradient in cases:
            with self.subTest(parameters=parameters):
                value, gradient = model.value_and_gradient(parameters=parameters)
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.evaluate(parameters=parameters), expected, rtol=0, atol=1e-12)

        fixed = model.fix('x', 3.0)
        assert fixed.parameter_names == ['y']
        value, gradient = fixed.value_and_gradient()
        np.testing.assert_allclose(value, 14.0 + 3.0j, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradient, [1.0j], rtol=0, atol=1e-12)

        freed = fixed.free('x')
        assert freed.parameter_names == ['x', 'y']
        value, gradient = freed.value_and_gradient(parameters={'x': 2.0})
        np.testing.assert_allclose(value, 9.0 + 3.0j, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradient, [4.0, 1.0j], rtol=0, atol=1e-12)

    def test_matrix_elements_and_solve_components(self) -> None:
        x = ld.parameter('x', initial=2.0)
        matrix = ld.matrix([[x, 0.0], [0.0, 3.0]])
        rhs = ld.vector([ld.complex(4.0, 2.0), 9.0])
        cases = (
            (matrix.at(0, 0), 2.0, [1.0]),
            ((matrix @ rhs)[0], 8.0 + 4.0j, [4.0 + 2.0j]),
            (ld.solve(matrix, rhs)[0], 2.0 + 1.0j, [-1.0 - 0.5j]),
        )
        for expression, expected, expected_gradient in cases:
            with self.subTest(expression=str(expression)):
                model = ld.Model(expression)
                value, gradient = model.value_and_gradient()
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.evaluate(), expected, rtol=0, atol=1e-12)

    def test_complex_gradient_matches_finite_differences(self) -> None:
        z = ld.complex(ld.parameter('x', initial=1.0), ld.parameter('y', initial=2.0))
        model = ld.Model(z * z * z)
        parameters = np.array(model.default_parameters)
        value, gradient = model.value_and_gradient(parameters=parameters)
        np.testing.assert_allclose(value, -11.0 - 2.0j, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradient, [-9.0 + 12.0j, -12.0 - 9.0j], rtol=0, atol=1e-12)

        step = 1e-6
        numerical = []
        for direction in np.eye(len(parameters)):
            plus = model.evaluate(parameters=parameters + step * direction)
            minus = model.evaluate(parameters=parameters - step * direction)
            numerical.append((plus - minus) / (2.0 * step))
        np.testing.assert_allclose(gradient, numerical, rtol=0, atol=1e-8)

    def test_dataset_evaluation_keeps_event_dimensions(self) -> None:
        z = ld.complex(ld.parameter('x', initial=2.0), ld.parameter('y', initial=3.0))
        model = ld.Model(z * z + 1.0)
        for size in (1, 3):
            dataset = ld.Dataset.from_arrays(p4s={}, scalars={}, weights=np.ones(size))
            for real in (False, True):
                with self.subTest(size=size, real=real):
                    value, gradient = model.value_and_gradient(real=real)
                    values, gradients = model.value_and_gradient(dataset, real=real)
                    evaluated = model.evaluate(dataset=dataset, real=real)
                    assert isinstance(values, np.ndarray)
                    assert isinstance(evaluated, np.ndarray)
                    assert values.shape == (size,)
                    assert evaluated.shape == (size,)
                    assert gradients.shape == (size, 2)
                    np.testing.assert_allclose(values, np.full(size, value), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(evaluated, values, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(gradients, np.tile(gradient, (size, 1)), rtol=0, atol=1e-12)

        event_model = ld.Model(ld.scalar('mass') * ld.parameter('scale', initial=2.0))
        dataset = ld.Dataset.from_arrays(p4s={}, scalars={'mass': np.array([1.0, 2.0, 4.0])})
        values, gradients = event_model.value_and_gradient(dataset, parameters={'scale': 3.0}, real=True)
        np.testing.assert_allclose(values, [3.0, 6.0, 12.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(gradients, [[1.0], [2.0], [4.0]], rtol=0, atol=1e-12)

    def test_invalid_parameters_match_dataset_validation(self) -> None:
        x = ld.parameter('x', initial=2.0)
        model = ld.Model(x * x)
        dataset = ld.Dataset.from_arrays(p4s={}, scalars={}, weights=np.ones(1))
        cases = (
            (object(), TypeError),
            ({'x': 'invalid'}, TypeError),
            ([], ld.LadduError),
            ([1.0, 2.0], ld.LadduError),
        )
        for parameters, error in cases:
            for events in (None, dataset):
                for evaluate in (model.evaluate, model.value_and_gradient):
                    with (
                        self.subTest(parameters=parameters, dataset=events is not None, method=evaluate.__name__),
                        self.assertRaises(error),
                    ):
                        evaluate(events, parameters=parameters)

    def test_cpu_precision_and_differentiation_settings(self) -> None:
        # 1 + 2**-24 rounds to 1 in f32; f64 retains the increment.
        x = ld.parameter('x', initial=1.0000000596046448)
        model = ld.Model(x * x)
        cases = (
            ('f32', 1.0, 2.0),
            ('f64', 1.000000119209293, 2.0000001192092896),
        )
        for precision, expected, expected_gradient in cases:
            for autodiff in ('auto', 'forward', 'reverse'):
                with self.subTest(precision=precision, autodiff=autodiff):
                    execution = ld.Execution('cpu', precision=precision, autodiff=autodiff, threads=1)
                    value, gradient = model.value_and_gradient(execution=execution, real=True)
                    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(gradient, [expected_gradient], rtol=0, atol=1e-15)
                    np.testing.assert_allclose(model.evaluate(execution=execution), expected, rtol=0, atol=1e-15)

    @unittest.skipUnless(ld.capabilities()['jit'], 'JIT support is not compiled into this installation')
    def test_explicit_jit_complex_evaluation(self) -> None:
        z = ld.complex(ld.parameter('x', initial=1.0), ld.parameter('y', initial=2.0))
        model = ld.Model(z * z * z)
        for autodiff in ('forward', 'reverse'):
            with self.subTest(autodiff=autodiff):
                execution = ld.Execution('jit', precision='f64', autodiff=autodiff)
                value, gradient = model.value_and_gradient(execution=execution)
                np.testing.assert_allclose(value, -11.0 - 2.0j, rtol=0, atol=1e-12)
                np.testing.assert_allclose(gradient, [-9.0 + 12.0j, -12.0 - 9.0j], rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.evaluate(execution=execution), value, rtol=0, atol=1e-12)

    def test_unavailable_gpu_request_does_not_fall_back(self) -> None:
        # Execution opens the adapter eagerly, so this exercises configuration
        # rejection on machines with or without GPU hardware.
        with self.assertRaisesRegex(ld.LadduError, 'GPU|WGPU|adapter'):
            ld.Execution('gpu', precision='f32', device='__laddu_test_missing_adapter__')

    def test_dataset_free_gpu_evaluation_is_rejected(self) -> None:
        if not ld.capabilities()['gpu'] or not ld.gpu.devices():
            self.skipTest('no GPU adapter available to construct a GPU execution')
        execution = ld.Execution('gpu', precision='f32')
        x = ld.parameter('x', initial=2.0)
        model = ld.Model(x * x)
        for evaluate in (model.evaluate, model.value_and_gradient):
            with (
                self.subTest(method=evaluate.__name__),
                self.assertRaisesRegex(ld.LadduError, 'without a dataset.*GPU'),
            ):
                evaluate(execution=execution)


if __name__ == '__main__':
    unittest.main()
