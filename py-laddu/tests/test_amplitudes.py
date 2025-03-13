import pytest

from laddu import (
    ComplexScalar,
    Dataset,
    Event,
    Manager,
    Scalar,
    Vector3,
    constant,
    parameter,
)
from laddu.amplitudes import AmplitudeOne, AmplitudeZero, amplitude_product, amplitude_sum


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


def test_constant_amplitude() -> None:
    manager = Manager()
    amp = Scalar('constant', constant(2.0))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([])
    assert result[0] == 2.0 + 0.0j


def test_parametric_amplitude() -> None:
    manager = Manager()
    amp = Scalar('parametric', parameter('test_param'))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    model = manager.model(aid)
    evaluator = model.load(dataset)
    result = evaluator.evaluate([3.0])
    assert result[0] == 3.0 + 0.0j


def test_expression_operations() -> None:
    manager = Manager()
    amp1 = ComplexScalar('const1', constant(2.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(0.0), constant(1.0))
    amp3 = ComplexScalar('const3', constant(3.0), constant(4.0))
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    aid3 = manager.register(amp3)
    dataset = make_test_dataset()
    expr_add = aid1 + aid2
    model_add = manager.model(expr_add)
    eval_add = model_add.load(dataset)
    result_add = eval_add.evaluate([])
    assert result_add[0] == 2.0 + 1.0j

    expr_mul = aid1 * aid2
    model_mul = manager.model(expr_mul)
    eval_mul = model_mul.load(dataset)
    result_mul = eval_mul.evaluate([])
    assert result_mul[0] == 0.0 + 2.0j

    expr_add2 = expr_add + expr_mul
    model_add2 = manager.model(expr_add2)
    eval_add2 = model_add2.load(dataset)
    result_add2 = eval_add2.evaluate([])
    assert result_add2[0] == 2.0 + 3.0j

    expr_mul2 = expr_add * expr_mul
    model_mul2 = manager.model(expr_mul2)
    eval_mul2 = model_mul2.load(dataset)
    result_mul2 = eval_mul2.evaluate([])
    assert result_mul2[0] == -2.0 + 4.0j

    expr_real = aid3.real()
    model_real = manager.model(expr_real)
    eval_real = model_real.load(dataset)
    result_real = eval_real.evaluate([])
    assert result_real[0] == 3.0 + 0.0j

    expr_mul2_real = expr_mul2.real()
    model_mul2_real = manager.model(expr_mul2_real)
    eval_mul2_real = model_mul2_real.load(dataset)
    result_mul2_real = eval_mul2_real.evaluate([])
    assert result_mul2_real[0] == -2.0 + 0.0j

    expr_imag = aid3.imag()
    model_imag = manager.model(expr_imag)
    eval_imag = model_imag.load(dataset)
    result_imag = eval_imag.evaluate([])
    assert result_imag[0] == 4.0 + 0.0j

    expr_mul2_imag = expr_mul2.imag()
    model_mul2_imag = manager.model(expr_mul2_imag)
    eval_mul2_imag = model_mul2_imag.load(dataset)
    result_mul2_imag = eval_mul2_imag.evaluate([])
    assert result_mul2_imag[0] == 4.0 + 0.0j

    expr_norm = aid1.norm_sqr()
    model_norm = manager.model(expr_norm)
    eval_norm = model_norm.load(dataset)
    result_norm = eval_norm.evaluate([])
    assert result_norm[0] == 4.0 + 0.0j

    expr_mul2_norm = expr_mul2.norm_sqr()
    model_mul2_norm = manager.model(expr_mul2_norm)
    eval_mul2_norm = model_mul2_norm.load(dataset)
    result_mul2_norm = eval_mul2_norm.evaluate([])
    assert result_mul2_norm[0] == 20.0 + 0.0j


def test_amplitude_activation() -> None:
    manager = Manager()
    amp1 = ComplexScalar('const1', constant(1.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(2.0), constant(0.0))
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    dataset = make_test_dataset()

    expr = aid1 + aid2
    model = manager.model(expr)
    evaluator = model.load(dataset)
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


def test_gradient() -> None:
    manager = Manager()
    amp1 = ComplexScalar(
        'parametric_1', parameter('test_param_re_1'), parameter('test_param_im_1')
    )
    amp2 = ComplexScalar(
        'parametric_2', parameter('test_param_re_2'), parameter('test_param_im_2')
    )
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    dataset = make_test_dataset()
    params = [2.0, 3.0, 4.0, 5.0]

    expr = aid1 * aid2
    model = manager.model(expr)
    evaluator = model.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 5.0
    assert gradient[0][1].real == -5.0
    assert gradient[0][1].imag == 4.0
    assert gradient[0][2].real == 2.0
    assert gradient[0][2].imag == 3.0
    assert gradient[0][3].real == -3.0
    assert gradient[0][3].imag == 2.0

    expr = (aid1 * aid2).real()
    model = manager.model(expr)
    evaluator = model.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == -5.0
    assert gradient[0][1].imag == 0.0
    assert gradient[0][2].real == 2.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == -3.0
    assert gradient[0][3].imag == 0.0

    expr = (aid1 * aid2).imag()
    model = manager.model(expr)
    evaluator = model.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 5.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 4.0
    assert gradient[0][1].imag == 0.0
    assert gradient[0][2].real == 3.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == 2.0
    assert gradient[0][3].imag == 0.0

    expr = (aid1 * aid2).norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 164.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 246.0
    assert gradient[0][1].imag == 0.0
    assert gradient[0][2].real == 104.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == 130.0
    assert gradient[0][3].imag == 0.0


def test_zeros_and_ones() -> None:
    manager = Manager()
    amp = ComplexScalar('parametric', parameter('test_param_re'), constant(2.0))
    aid = manager.register(amp)
    dataset = make_test_dataset()
    expr = (aid * AmplitudeOne() + AmplitudeZero()).norm_sqr()
    model = manager.model(expr)
    evaluator = model.load(dataset)

    params = [2.0]
    value = evaluator.evaluate(params)
    gradient = evaluator.evaluate_gradient(params)

    assert value[0].real == 8.0
    assert value[0].imag == 0.0

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0


def test_parameter_registration() -> None:
    manager = Manager()
    amp = Scalar('parametric', parameter('test_param'))
    aid = manager.register(amp)
    parameters = manager.parameters
    model = manager.model(aid)
    model_parameters = model.parameters
    assert len(parameters) == 1
    assert parameters[0] == 'test_param'
    assert len(model_parameters) == 1
    assert model_parameters[0] == 'test_param'


def test_duplicate_amplitude_registration() -> None:
    manager = Manager()
    amp1 = ComplexScalar('same_name', constant(1.0), constant(0.0))
    amp2 = ComplexScalar('same_name', constant(2.0), constant(0.0))
    _ = manager.register(amp1)
    with pytest.raises(ValueError):
        _ = manager.register(amp2)


def test_tree_printing() -> None:
    manager = Manager()
    amp1 = ComplexScalar(
        'parametric_1', parameter('test_param_re_1'), parameter('test_param_im_1')
    )
    amp2 = ComplexScalar(
        'parametric_2', parameter('test_param_re_2'), parameter('test_param_im_2')
    )
    aid1 = manager.register(amp1)
    aid2 = manager.register(amp2)
    expr = (
        aid1.real()
        + aid2.imag()
        + AmplitudeOne() * AmplitudeZero()
        + (aid1 * aid2).norm_sqr()
    )
    assert (
        str(expr)
        == """+
├─ +
│  ├─ +
│  │  ├─ Re
│  │  │  └─ parametric_1(id=0)
│  │  └─ Im
│  │     └─ parametric_2(id=1)
│  └─ *
│     ├─ 1
│     └─ 0
└─ NormSqr
   └─ *
      ├─ parametric_1(id=0)
      └─ parametric_2(id=1)
"""
    )


def test_amplitude_summation() -> None:
    manager = Manager()
    terms = [manager.register(Scalar(f'{i}', constant(i))) for i in range(1, 5)]
    dataset = make_test_dataset()
    params = []

    expr = amplitude_sum([])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 0.0 + 0.0j

    expr = amplitude_sum(terms)
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j

    expr = amplitude_sum([terms[0]])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = amplitude_sum([terms[0], amplitude_sum(terms[1:])])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j

    expr = amplitude_sum([amplitude_sum(terms[1:]), terms[0]])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j


def test_amplitude_product() -> None:
    manager = Manager()
    terms = [manager.register(Scalar(f'{i}', constant(i))) for i in range(1, 5)]
    dataset = make_test_dataset()
    params = []

    expr = amplitude_product([])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = amplitude_product(terms)
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j

    expr = amplitude_product([terms[0]])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = amplitude_product([terms[0], amplitude_product(terms[1:])])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j

    expr = amplitude_product([amplitude_product(terms[1:]), terms[0]])
    model = manager.model(expr)
    evaluator = model.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j
