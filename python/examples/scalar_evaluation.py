"""Check scalar values and parameter gradients, with and without event inputs."""

# ruff: noqa: S101

import laddu as ld
import numpy as np

# Parameter-only expression: f(x, y) = (x + i*y)**2 + 1.
x = ld.parameter('x', initial=2.0)
y = ld.parameter('y', initial=3.0)
z = ld.complex(x, y)
model = ld.Model(z * z + 1.0)
parameters = {'x': 2.0, 'y': 3.0}
value = model.evaluate(parameters=parameters)
gradient_value, gradient = model.value_and_gradient(parameters=parameters)

# df/dx = 2*(x + i*y), df/dy = 2*i*(x + i*y).
assert isinstance(value, complex)
assert isinstance(gradient_value, complex)
assert model.parameter_names == ['x', 'y']
assert gradient.shape == (2,)
np.testing.assert_allclose([value, gradient_value], [-4.0 + 12.0j, -4.0 + 12.0j], rtol=0, atol=1e-12)
np.testing.assert_allclose(gradient, [4.0 + 6.0j, -6.0 + 4.0j], rtol=0, atol=1e-12)

# Explicit None, defaults, and ordered sequences use the same public methods.
np.testing.assert_allclose(model.evaluate(None), value, rtol=0, atol=1e-12)
real_value, real_gradient = model.value_and_gradient(None, parameters=[2.0, 3.0], real=True)
assert isinstance(real_value, float)
np.testing.assert_allclose(real_value, -4.0, rtol=0, atol=1e-12)
np.testing.assert_allclose(real_gradient, [4.0, -6.0], rtol=0, atol=1e-12)

# Select a scalar from vector/matrix operations before constructing a model.
matrix = ld.matrix([[x, 0.0], [0.0, 3.0]])
rhs = ld.vector([ld.complex(4.0, 2.0), 9.0])
component = ld.Model(ld.solve(matrix, rhs)[0])
component_value, component_gradient = component.value_and_gradient()
np.testing.assert_allclose(component_value, 2.0 + 1.0j, rtol=0, atol=1e-12)
np.testing.assert_allclose(component_gradient, [-1.0 - 0.5j], rtol=0, atol=1e-12)

# An event-dependent expression requires a dataset, even for one example event.
event_model = ld.Model(ld.scalar('mass') * ld.parameter('scale', initial=2.0))
dataset = ld.Dataset.from_arrays(p4s={}, scalars={'mass': np.array([1.5])})
values = event_model.evaluate(dataset, parameters={'scale': 3.0}, real=True)
gradient_values, gradients = event_model.value_and_gradient(dataset=dataset, parameters={'scale': 3.0}, real=True)
assert isinstance(values, np.ndarray)
assert isinstance(gradient_values, np.ndarray)
assert values.shape == (1,)
assert gradients.shape == (1, 1)
np.testing.assert_allclose(values, [4.5], rtol=0, atol=1e-12)
np.testing.assert_allclose(gradient_values, [4.5], rtol=0, atol=1e-12)
np.testing.assert_allclose(gradients, [[1.5]], rtol=0, atol=1e-12)
