from laddu import (
    ComplexScalar,
    Dataset,
    Event,
    Scalar,
    Vec3,
    constant,
    parameter,
)
from laddu.amplitudes import (
    One,
    TestAmplitude,
    Zero,
    expr_product,
    expr_sum,
)

P4_NAMES = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_NAMES = ['pol_magnitude', 'pol_angle']
AUX_VALUES = [0.38562805, 0.05708078]


def make_test_event() -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        AUX_VALUES.copy(),
        0.48,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def make_test_event_with_beam_energy(energy: float) -> Event:
    return Event(
        [
            Vec3(0.0, 0.0, energy).with_mass(0.0),
            Vec3(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        AUX_VALUES.copy(),
        0.48,
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )


def make_test_dataset() -> Dataset:
    return Dataset([make_test_event()], p4_names=P4_NAMES, aux_names=AUX_NAMES)


def test_constant_amplitude() -> None:
    amp = Scalar('constant', constant(2.0))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([])
    assert result[0] == 2.0 + 0.0j


def test_parametric_amplitude() -> None:
    amp = Scalar('parametric', parameter('test_param'))
    dataset = make_test_dataset()
    evaluator = amp.load(dataset)
    result = evaluator.evaluate([3.0])
    assert result[0] == 3.0 + 0.0j


def test_batch_evaluation() -> None:
    amp = TestAmplitude('test', parameter('real'), parameter('imag'))
    event1 = make_test_event_with_beam_energy(10.0)
    event2 = make_test_event_with_beam_energy(11.0)
    event3 = make_test_event_with_beam_energy(12.0)
    dataset = Dataset(
        [event1, event2, event3],
        p4_names=P4_NAMES,
        aux_names=AUX_NAMES,
    )
    evaluator = amp.load(dataset)
    result = evaluator.evaluate_batch([1.1, 2.2], [0, 2])
    assert len(result) == 2
    assert result[0] == (1.1 + 2.2j) * 10.0
    assert result[1] == (1.1 + 2.2j) * 12.0
    result_grad = evaluator.evaluate_gradient_batch([1.1, 2.2], [0, 2])
    assert len(result_grad) == 2
    assert result_grad[0][0].real == 10.0
    assert result_grad[0][0].imag == 0.0
    assert result_grad[0][1].real == 0.0
    assert result_grad[0][1].imag == 10.0
    assert result_grad[1][0].real == 12.0
    assert result_grad[1][0].imag == 0.0
    assert result_grad[1][1].real == 0.0
    assert result_grad[1][1].imag == 12.0


def test_expression_operations() -> None:
    amp1 = ComplexScalar('const1', constant(2.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(0.0), constant(1.0))
    amp3 = ComplexScalar('const3', constant(3.0), constant(4.0))
    dataset = make_test_dataset()

    expr_add = amp1 + amp2
    eval_add = expr_add.load(dataset)
    result_add = eval_add.evaluate([])
    assert result_add[0] == 2.0 + 1.0j

    expr_sub = amp1 - amp2
    eval_sub = expr_sub.load(dataset)
    result_sub = eval_sub.evaluate([])
    assert result_sub[0] == 2.0 - 1.0j

    expr_mul = amp1 * amp2
    eval_mul = expr_mul.load(dataset)
    result_mul = eval_mul.evaluate([])
    assert result_mul[0] == 0.0 + 2.0j

    expr_div = amp1 / amp3
    eval_div = expr_div.load(dataset)
    result_div = eval_div.evaluate([])
    assert result_div[0] == (6.0 / 25.0) - (8.0j / 25.0)

    expr_neg = -amp3
    eval_neg = expr_neg.load(dataset)
    result_neg = eval_neg.evaluate([])
    assert result_neg[0] == -3.0 - 4.0j

    expr_add2 = expr_add + expr_mul
    eval_add2 = expr_add2.load(dataset)
    result_add2 = eval_add2.evaluate([])
    assert result_add2[0] == 2.0 + 3.0j

    expr_sub2 = expr_add - expr_mul
    eval_sub2 = expr_sub2.load(dataset)
    result_sub2 = eval_sub2.evaluate([])
    assert result_sub2[0] == 2.0 - 1.0j

    expr_mul2 = expr_add * expr_mul
    eval_mul2 = expr_mul2.load(dataset)
    result_mul2 = eval_mul2.evaluate([])
    assert result_mul2[0] == -2.0 + 4.0j

    expr_div2 = expr_add / expr_add2
    eval_div2 = expr_div2.load(dataset)
    result_div2 = eval_div2.evaluate([])
    assert result_div2[0] == (7.0 / 13.0) - (4.0j / 13.0)

    expr_neg2 = -expr_mul2
    eval_neg2 = expr_neg2.load(dataset)
    result_neg2 = eval_neg2.evaluate([])
    assert result_neg2[0] == 2.0 - 4.0j

    expr_real = amp3.real()
    eval_real = expr_real.load(dataset)
    result_real = eval_real.evaluate([])
    assert result_real[0] == 3.0 + 0.0j

    expr_mul2_real = expr_mul2.real()
    eval_mul2_real = expr_mul2_real.load(dataset)
    result_mul2_real = eval_mul2_real.evaluate([])
    assert result_mul2_real[0] == -2.0 + 0.0j

    expr_imag = amp3.imag()
    eval_imag = expr_imag.load(dataset)
    result_imag = eval_imag.evaluate([])
    assert result_imag[0] == 4.0 + 0.0j

    expr_mul2_imag = expr_mul2.imag()
    eval_mul2_imag = expr_mul2_imag.load(dataset)
    result_mul2_imag = eval_mul2_imag.evaluate([])
    assert result_mul2_imag[0] == 4.0 + 0.0j

    expr_conj = amp3.conj()
    eval_conj = expr_conj.load(dataset)
    result_conj = eval_conj.evaluate([])
    assert result_conj[0] == 3.0 - 4.0j

    expr_mul2_conj = expr_mul2.conj()
    eval_mul2_conj = expr_mul2_conj.load(dataset)
    result_mul2_conj = eval_mul2_conj.evaluate([])
    assert result_mul2_conj[0] == -2.0 - 4.0j

    expr_norm = amp1.norm_sqr()
    eval_norm = expr_norm.load(dataset)
    result_norm = eval_norm.evaluate([])
    assert result_norm[0] == 4.0 + 0.0j

    expr_mul2_norm = expr_mul2.norm_sqr()
    eval_mul2_norm = expr_mul2_norm.load(dataset)
    result_mul2_norm = eval_mul2_norm.evaluate([])
    assert result_mul2_norm[0] == 20.0 + 0.0j


def test_amplitude_activation() -> None:
    amp1 = ComplexScalar('const1', constant(1.0), constant(0.0))
    amp2 = ComplexScalar('const2', constant(2.0), constant(0.0))
    dataset = make_test_dataset()

    expr = amp1 + amp2
    evaluator = expr.load(dataset)
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
    amp1 = ComplexScalar(
        'parametric_1', parameter('test_param_re_1'), parameter('test_param_im_1')
    )
    amp2 = ComplexScalar(
        'parametric_2', parameter('test_param_re_2'), parameter('test_param_im_2')
    )
    dataset = make_test_dataset()
    params = [2.0, 3.0, 4.0, 5.0]

    expr = amp1 + amp2
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 1.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 0.0
    assert gradient[0][1].imag == 1.0
    assert gradient[0][2].real == 1.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == 0.0
    assert gradient[0][3].imag == 1.0

    expr = amp1 - amp2
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 1.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 0.0
    assert gradient[0][1].imag == 1.0
    assert gradient[0][2].real == -1.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == 0.0
    assert gradient[0][3].imag == -1.0

    expr = amp1 * amp2
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 5.0
    assert gradient[0][1].real == -5.0
    assert gradient[0][1].imag == 4.0
    assert gradient[0][2].real == 2.0
    assert gradient[0][2].imag == 3.0
    assert gradient[0][3].real == -3.0
    assert gradient[0][3].imag == 2.0

    expr = amp1 / amp2
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0 / 41.0
    assert gradient[0][0].imag == -5.0 / 41.0
    assert gradient[0][1].real == 5.0 / 41.0
    assert gradient[0][1].imag == 4.0 / 41.0
    assert gradient[0][2].real == -102.0 / 1681.0
    assert gradient[0][2].imag == 107.0 / 1681.0
    assert gradient[0][3].real == -107.0 / 1681.0
    assert gradient[0][3].imag == -102.0 / 1681.0

    expr = -(amp1 * amp2)
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == -4.0
    assert gradient[0][0].imag == -5.0
    assert gradient[0][1].real == 5.0
    assert gradient[0][1].imag == -4.0
    assert gradient[0][2].real == -2.0
    assert gradient[0][2].imag == -3.0
    assert gradient[0][3].real == 3.0
    assert gradient[0][3].imag == -2.0

    expr = (amp1 * amp2).real()
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == -5.0
    assert gradient[0][1].imag == 0.0
    assert gradient[0][2].real == 2.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == -3.0
    assert gradient[0][3].imag == 0.0

    expr = (amp1 * amp2).imag()
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 5.0
    assert gradient[0][0].imag == 0.0
    assert gradient[0][1].real == 4.0
    assert gradient[0][1].imag == 0.0
    assert gradient[0][2].real == 3.0
    assert gradient[0][2].imag == 0.0
    assert gradient[0][3].real == 2.0
    assert gradient[0][3].imag == 0.0

    expr = (amp1 * amp2).conj()
    evaluator = expr.load(dataset)

    gradient = evaluator.evaluate_gradient(params)

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == -5.0
    assert gradient[0][1].real == -5.0
    assert gradient[0][1].imag == -4.0
    assert gradient[0][2].real == 2.0
    assert gradient[0][2].imag == -3.0
    assert gradient[0][3].real == -3.0
    assert gradient[0][3].imag == -2.0

    expr = (amp1 * amp2).norm_sqr()
    evaluator = expr.load(dataset)

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
    amp = ComplexScalar('parametric', parameter('test_param_re'), constant(2.0))
    dataset = make_test_dataset()
    expr = (amp * One() + Zero()).norm_sqr()
    evaluator = expr.load(dataset)

    params = [2.0]
    value = evaluator.evaluate(params)
    gradient = evaluator.evaluate_gradient(params)

    assert value[0].real == 8.0
    assert value[0].imag == 0.0

    assert gradient[0][0].real == 4.0
    assert gradient[0][0].imag == 0.0


def test_parameter_registration() -> None:
    amp = Scalar('parametric', parameter('test_param'))
    parameters = amp.parameters
    assert len(parameters) == 1
    assert parameters[0] == 'test_param'


# TODO: This panics rather than raises an exception.
# I'm not sure if it's possible to avoid this, since operations in Rust aren't fallible
# def test_duplicate_amplitude_registration() -> None:
#     amp1 = ComplexScalar('same_name', constant(1.0), constant(0.0))
#     amp2 = ComplexScalar('same_name', constant(2.0), constant(0.0))
#     with pytest.raises(ValueError):
#         amp1 + amp2
#


def test_tree_printing() -> None:
    amp1 = ComplexScalar(
        'parametric_1', parameter('test_param_re_1'), parameter('test_param_im_1')
    )
    amp2 = ComplexScalar(
        'parametric_2', parameter('test_param_re_2'), parameter('test_param_im_2')
    )
    expr = (
        amp1.real()
        + amp2.conj().imag()
        + One() * -Zero()
        - Zero() / One()
        + (amp1 * amp2).norm_sqr()
    )
    assert (
        str(expr)
        == """+
├─ -
│  ├─ +
│  │  ├─ +
│  │  │  ├─ Re
│  │  │  │  └─ parametric_1(id=0)
│  │  │  └─ Im
│  │  │     └─ *
│  │  │        └─ parametric_2(id=1)
│  │  └─ ×
│  │     ├─ 1
│  │     └─ -
│  │        └─ 0
│  └─ ÷
│     ├─ 0
│     └─ 1
└─ NormSqr
   └─ ×
      ├─ parametric_1(id=0)
      └─ parametric_2(id=1)
"""
    )


def test_amplitude_summation() -> None:
    terms = [Scalar(f'{i}', constant(i)) for i in range(1, 5)]
    dataset = make_test_dataset()
    params = []

    expr = expr_sum([])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 0.0 + 0.0j

    expr = expr_sum(terms)
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j

    expr = expr_sum([terms[0]])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = expr_sum([terms[0], expr_sum(terms[1:])])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j

    expr = expr_sum([expr_sum(terms[1:]), terms[0]])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 2.0 + 3.0 + 4.0 + 0.0j


def test_amplitude_product() -> None:
    terms = [Scalar(f'{i}', constant(i)) for i in range(1, 5)]
    dataset = make_test_dataset()
    params = []

    expr = expr_product([])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = expr_product(terms)
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j

    expr = expr_product([terms[0]])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 + 0.0j

    expr = expr_product([terms[0], expr_product(terms[1:])])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j

    expr = expr_product([expr_product(terms[1:]), terms[0]])
    evaluator = expr.load(dataset)

    value = evaluator.evaluate(params)
    assert value[0] == 1.0 * 2.0 * 3.0 * 4.0 + 0.0j
