from laddu import (
    Manager,
    Scalar,
    ComplexScalar,
    Event,
    Dataset,
    Vector3,
    constant,
    parameter,
)
import pytest


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


def test_constant_amplitude():
    manager = Manager()
    amp = Scalar('constant', constant(2.0))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([])
    assert result[0] == 2.0 + 0.0j


def test_parametric_amplitude():
    manager = Manager()
    amp = Scalar('parametric', parameter('test_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    evaluator = manager.load(aid, dataset)
    result = evaluator.evaluate([3.0])
    assert result[0] == 3.0 + 0.0j


def test_expression_operations():
    manager = Manager()
    amp1 = ComplexScalar('const1', constant(2.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(0.0), constant(1.0))
    amp3 = ComplexScalar('const3', constant(3.0), constant(4.0))
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    aid3 = manager.register(amp3)
    dataset = make_test_dataset()
    expr_add = aid1 + aid2
    eval_add = manager.load(expr_add, dataset)
    result_add = eval_add.evaluate([])
    assert result_add[0] == 2.0 + 1.0j

    expr_mul = aid1 * aid2
    eval_mul = manager.load(expr_mul, dataset)
    result_mul = eval_mul.evaluate([])
    assert result_mul[0] == 0.0 + 2.0j

    expr_add2 = expr_add + expr_mul
    eval_add2 = manager.load(expr_add2, dataset)
    result_add2 = eval_add2.evaluate([])
    assert result_add2[0] == 2.0 + 3.0j

    expr_mul2 = expr_add * expr_mul
    eval_mul2 = manager.load(expr_mul2, dataset)
    result_mul2 = eval_mul2.evaluate([])
    assert result_mul2[0] == -2.0 + 4.0j

    expr_real = aid3.real()
    eval_real = manager.load(expr_real, dataset)
    result_real = eval_real.evaluate([])
    assert result_real[0] == 3.0 + 0.0j

    expr_mul2_real = expr_mul2.real()
    eval_mul2_real = manager.load(expr_mul2_real, dataset)
    result_mul2_real = eval_mul2_real.evaluate([])
    assert result_mul2_real[0] == -2.0 + 0.0j

    expr_imag = aid3.imag()
    eval_imag = manager.load(expr_imag, dataset)
    result_imag = eval_imag.evaluate([])
    assert result_imag[0] == 4.0 + 0.0j

    expr_mul2_imag = expr_mul2.imag()
    eval_mul2_imag = manager.load(expr_mul2_imag, dataset)
    result_mul2_imag = eval_mul2_imag.evaluate([])
    assert result_mul2_imag[0] == 4.0 + 0.0j

    expr_norm = aid1.norm_sqr()
    eval_norm = manager.load(expr_norm, dataset)
    result_norm = eval_norm.evaluate([])
    assert result_norm[0] == 4.0 + 0.0j

    expr_mul2_norm = expr_mul2.norm_sqr()
    eval_mul2_norm = manager.load(expr_mul2_norm, dataset)
    result_mul2_norm = eval_mul2_norm.evaluate([])
    assert result_mul2_norm[0] == 20.0 + 0.0j


def test_amplitude_activation():
    manager = Manager()
    amp1 = ComplexScalar('const1', constant(1.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(2.0), constant(0.0))
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    dataset = make_test_dataset()

    expr = aid1 + aid2
    evaluator = manager.load(expr, dataset)
    result = evaluator.evaluate([])
    assert result[0] == 3.0 + 0.0j

    evaluator.deactivate('const1')
    result = evaluator.evaluate([])
    assert result[0] == 2.0 + 0.0j

    evaluator.isolate('const1')
    result = evaluator.evaluate([])
    assert result[0] == 1.0 + 0.0j

    evaluator.activate_all()
    result = evaluator.evaluate([])
    assert result[0] == 3.0 + 0.0j


def test_gradient():
    manager = Manager()
    amp = Scalar('parametric', parameter('test_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = aid.norm_sqr()
    evaluator = manager.load(expr, dataset)
    params = [2.0]
    gradient = evaluator.evaluate_gradient(params)
    # For |f(x)|^2 where f(x) = x, the derivative should be 2x
    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0


def test_parameter_registration():
    manager = Manager()
    amp = Scalar('parametric', parameter('test_param'))
    manager.register(amp)
    parameters = manager.parameters
    assert len(parameters) == 1
    assert parameters[0] == 'test_param'


def test_duplicate_amplitude_registration():
    manager = Manager()
    amp1 = ComplexScalar('same_name', constant(1.0), constant(0.0))
    amp2 = ComplexScalar('same_name', constant(2.0), constant(0.0))
    manager.register(amp1)
    with pytest.raises(ValueError):
        manager.register(amp2)