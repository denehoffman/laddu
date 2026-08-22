use super::*;

#[test]
fn forward_gradients_match_scalar_complex_finite_differences() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 0.4));
    let y = laddu_expr::Expr::from(parameter!("y", initial: -0.2));
    let expression = complex(x.clone().sin(), y.clone().exp()).norm_sqr() + (x * y).cos();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let result = plan.evaluate_with_gradient(&params).unwrap();
    let ir_result = gradient_interpreter::GradientInterpreter::new(
        plan.scalar_kernel.as_ref().unwrap(),
        model.params().free_params(),
    )
    .unwrap()
    .evaluate(&params, None)
    .unwrap()
    .1;

    for (actual, expected) in ir_result.iter().zip(result.gradient()) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "{actual} != {expected}"
        );
    }

    for (parameter, derivative) in result.gradient().iter().enumerate() {
        let expected = finite_difference(&plan, &params, parameter);
        assert!((derivative - expected).norm() < 1.0e-8);
    }
}
#[test]
fn reverse_gradients_match_forward_for_scalar_complex_operations() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 0.8));
    let y = laddu_expr::Expr::from(parameter!("y", initial: -0.3));
    let z = complex(x.clone(), y.clone());
    let expression = x.clone().sqrt()
        + x.clone().log()
        + x.clone().powi(-2)
        + x.clone().sin()
        + x.clone().cos()
        + x.clone().exp()
        + z.clone().conj()
        + z.clone().real()
        + z.clone().imag()
        + z.norm_sqr()
        + atan2(y.clone(), x.clone())
        + x * y;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();

    let expected = forward.evaluate_with_gradient(&params).unwrap();
    let actual = reverse.evaluate_with_gradient(&params).unwrap();

    assert!((actual.value() - expected.value()).norm() < 1.0e-12);
    assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
}
#[test]
fn reverse_gradients_match_forward_for_structured_linear_algebra() {
    let a = laddu_expr::Expr::from(parameter!("a", initial: 0.7));
    let b = laddu_expr::Expr::from(parameter!("b", initial: -0.2));
    let c = laddu_expr::Expr::from(parameter!("c", initial: 1.1));
    let d = laddu_expr::Expr::from(parameter!("d", initial: 0.4));
    let x = laddu_expr::Expr::from(parameter!("x", initial: -0.3));
    let y = laddu_expr::Expr::from(parameter!("y", initial: 0.9));
    let left = matrix([
        [a.clone(), complex(b.clone(), 0.2)],
        [1.3.into(), c.clone()],
    ]);
    let right = matrix([[complex(0.5, -0.1), d.clone()], [b.clone(), 0.8.into()]]);
    let product = matmul(left, right);
    let input_vector = vector([x.clone(), complex(y.clone(), -0.4)]);
    let projected = matvec(product.clone(), input_vector);
    let expression = dot(projected.clone(), vector([complex(0.25, 0.3), c.clone()]))
        + product.matrix_element(1, 0)
        + projected.component(1);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();

    let expected = forward.evaluate_with_gradient(&params).unwrap();
    let actual = reverse.evaluate_with_gradient(&params).unwrap();

    assert!((actual.value() - expected.value()).norm() < 1.0e-12);
    assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
}
#[test]
fn reverse_gradients_match_forward_for_parameter_dependent_solve() {
    let a = laddu_expr::Expr::from(parameter!("a", initial: 2.0));
    let b = laddu_expr::Expr::from(parameter!("b", initial: 0.3));
    let r = laddu_expr::Expr::from(parameter!("r", initial: 1.2));
    let solution = solve(
        matrix([[a, complex(b.clone(), 0.1)], [b, 1.7.into()]]),
        vector([r, complex(0.5, -0.1)]),
    );
    let expression = dot(solution, vector([complex(1.0, 0.2), (-0.4).into()]));
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();

    let expected = forward.evaluate_with_gradient(&params).unwrap();
    let actual = reverse.evaluate_with_gradient(&params).unwrap();

    assert!((actual.value() - expected.value()).norm() < 1.0e-12);
    assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
}
#[test]
fn reverse_cached_event_gradients_match_forward() {
    let x = event_scalar("x");
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.4));
    let phase = laddu_expr::Expr::from(parameter!("phase", initial: -0.2));
    let expression =
        complex((x.clone() * &scale).sin(), (x.clone() + phase).cos()).norm_sqr() + x * scale;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.25]),
            OwnedEvent::new(vec![], vec![0.75]),
            OwnedEvent::new(vec![], vec![1.25]),
        ],
    )
    .unwrap();

    let expected = forward
        .evaluate_cache_with_gradient(&params, &forward.cache_event_batch(&batch).unwrap())
        .unwrap();
    let actual = reverse
        .evaluate_cache_with_gradient(&params, &reverse.cache_event_batch(&batch).unwrap())
        .unwrap();

    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual.value() - expected.value()).norm() < 1.0e-12);
        assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
    }
}
#[test]
fn reverse_cached_event_materialization_is_a_leaf() {
    let event_sum = event_scalar("x") + event_scalar("y").sin();
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
    let expression = scale * event_sum;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.5, 0.25])],
    )
    .unwrap();
    assert!(
        reverse
            .cache_slots
            .iter()
            .enumerate()
            .any(|(index, slot)| slot.is_some() && reverse.cached_value_slots[index].is_some())
    );

    let expected = forward
        .evaluate_cache_row_with_gradient(&params, &forward.cache_event_batch(&batch).unwrap(), 0)
        .unwrap();
    let actual = reverse
        .evaluate_cache_row_with_gradient(&params, &reverse.cache_event_batch(&batch).unwrap(), 0)
        .unwrap();

    assert_eq!(actual.value(), expected.value());
    assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
}
#[test]
fn reverse_f32_gradients_match_forward() {
    let expression = laddu_expr::Expr::from(parameter!("x", initial: 0.4)).sin();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let reverse = CpuBackend
        .prepare_with_modes_precision(
            &model,
            AutodiffMode::Reverse,
            CpuExecutionMode::Interpreter,
            Precision::F32,
        )
        .unwrap();
    let forward = CpuBackend
        .prepare_with_modes_precision(
            &model,
            AutodiffMode::Forward,
            CpuExecutionMode::Interpreter,
            Precision::F32,
        )
        .unwrap();

    assert_eq!(
        reverse.evaluate_with_gradient(&params).unwrap(),
        forward.evaluate_with_gradient(&params).unwrap()
    );
}
#[test]
fn forward_gradients_cover_unary_atan2_and_zero_products() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 0.8));
    let y = laddu_expr::Expr::from(parameter!("y", initial: 0.0));
    let z = complex(x.clone(), y.clone());
    let expression = x.clone().sqrt()
        + x.clone().log()
        + x.clone().powi(-2)
        + x.clone().sin()
        + x.clone().cos()
        + x.clone().exp()
        + z.clone().conj().real()
        + z.clone().imag()
        + z.norm_sqr()
        + atan2(y.clone(), x.clone())
        + y * x;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let result = plan.evaluate_with_gradient(&params).unwrap();

    for (parameter, derivative) in result.gradient().iter().enumerate() {
        let expected = finite_difference(&plan, &params, parameter);
        assert!((derivative - expected).norm() < 1.0e-7);
    }
}
#[test]
fn forward_gradients_cover_matrix_vector_and_dot_operations() {
    let a = laddu_expr::Expr::from(parameter!("a", initial: 0.7));
    let b = laddu_expr::Expr::from(parameter!("b", initial: -0.2));
    let c = laddu_expr::Expr::from(parameter!("c", initial: 1.1));
    let d = laddu_expr::Expr::from(parameter!("d", initial: 0.4));
    let x = laddu_expr::Expr::from(parameter!("x", initial: -0.3));
    let y = laddu_expr::Expr::from(parameter!("y", initial: 0.9));
    let left = matrix([
        [a.clone(), complex(b.clone(), 0.2)],
        [1.3.into(), c.clone()],
    ]);
    let right = matrix([[complex(0.5, -0.1), d.clone()], [b.clone(), 0.8.into()]]);
    let product = matmul(left, right);
    let input_vector = vector([x.clone(), complex(y.clone(), -0.4)]);
    let projected = matvec(product.clone(), input_vector);
    let expression = dot(projected.clone(), vector([complex(0.25, 0.3), c.clone()]))
        + product.matrix_element(1, 0)
        + projected.component(1);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let result = plan.evaluate_with_gradient(&params).unwrap();

    for (parameter, derivative) in result.gradient().iter().enumerate() {
        let expected = finite_difference(&plan, &params, parameter);
        assert!(
            (derivative - expected).norm() < 1.0e-7,
            "{derivative} != {expected}"
        );
    }
}
#[test]
fn solve_gradients_match_finite_differences_for_matrix_and_rhs_parameters() {
    let a = laddu_expr::Expr::from(parameter!("a", initial: 2.0));
    let b = laddu_expr::Expr::from(parameter!("b", initial: 0.3));
    let r = laddu_expr::Expr::from(parameter!("r", initial: 1.2));
    let solution = solve(
        matrix([[a, b], [0.2.into(), 1.7.into()]]),
        vector([r, complex(0.5, -0.1)]),
    );
    let expression = dot(solution, vector([complex(1.0, 0.2), (-0.4).into()]));
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let result = plan.evaluate_with_gradient(&params).unwrap();

    for (parameter, derivative) in result.gradient().iter().enumerate() {
        let expected = finite_difference(&plan, &params, parameter);
        assert!((derivative - expected).norm() < 1.0e-8);
    }
}
#[test]
fn cpu_f32_direct_event_gradient_matches_cached_event_gradient() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.5));
    let offset = laddu_expr::Expr::from(parameter!("offset", initial: -0.25));
    let x = event_scalar("x");
    let expression =
        (x.clone().sin() * scale.clone() + offset.clone()).exp() + complex(scale, x).norm_sqr();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions::default()),
        precision: Precision::F32,
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let params = model.params().default_values();
    let event = HashMap::from([("x".to_owned(), 0.75)]);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.75])],
    )
    .unwrap();
    let cache = plan.cache_event_batch(&batch).unwrap();

    let direct_value = plan.evaluate_with_event(&params, &event).unwrap();
    let direct = plan
        .evaluate_with_event_and_gradient(&params, &event)
        .unwrap();
    let cached_value = plan.evaluate_cache_row(&params, &cache, 0).unwrap();
    let cached = plan
        .evaluate_cache_row_with_gradient(&params, &cache, 0)
        .unwrap();

    assert_eq!(direct_value, cached_value);
    assert_eq!(direct.value(), cached.value());
    assert_eq!(direct.gradient(), cached.gradient());
}
