use super::*;

#[cfg(feature = "jit")]
#[test]
fn forward_and_reverse_gradients_match_interpreter_and_jit_in_both_precisions() {
    let event = event_scalar("x");
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.4));
    let phase = laddu_expr::Expr::from(parameter!("phase", initial: -0.2));
    let expression = complex(
        (event.clone() * scale.clone()).sin(),
        (event.clone() + phase).cos(),
    )
    .norm_sqr()
        + event * scale;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let batch = EventBatch::from_events(
        schema,
        [
            OwnedEvent::weighted(vec![], vec![0.25], 0.5),
            OwnedEvent::weighted(vec![], vec![0.75], 1.5),
        ],
    )
    .unwrap();

    for (precision, tolerance) in [(Precision::F32, 1.0e-6), (Precision::F64, 1.0e-12)] {
        let baseline = CpuBackend
            .prepare_with_modes_precision(
                &model,
                AutodiffMode::Forward,
                CpuExecutionMode::Interpreter,
                precision,
            )
            .unwrap();
        let expected = baseline
            .evaluate_cache_with_gradient(&params, &baseline.cache_event_batch(&batch).unwrap())
            .unwrap();

        for (autodiff, execution) in [
            (AutodiffMode::Reverse, CpuExecutionMode::Interpreter),
            (AutodiffMode::Forward, CpuExecutionMode::Auto),
            (AutodiffMode::Reverse, CpuExecutionMode::Auto),
        ] {
            let plan = CpuBackend
                .prepare_with_modes_precision(&model, autodiff, execution, precision)
                .unwrap();
            if execution == CpuExecutionMode::Auto {
                assert!(matches!(plan.gradient_executor, GradientExecutor::Jit(_)));
            }
            let actual = plan
                .evaluate_cache_with_gradient(&params, &plan.cache_event_batch(&batch).unwrap())
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual.value() - expected.value()).norm() < tolerance);
                assert_gradient_close(actual.gradient(), expected.gradient(), tolerance);
            }
        }
    }
}
#[cfg(feature = "jit")]
#[test]
fn jit_gradient_reduction_matches_interpreter() {
    let x = event_scalar("x").real();
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.4));
    let phase = laddu_expr::Expr::from(parameter!("phase", initial: -0.2));
    let intensity = complex((x.clone() * &scale).sin(), (x.clone() + phase).cos()).norm_sqr() + 0.5;
    let expression = complex(intensity, x * scale);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));

    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let batch = EventBatch::from_events(
        schema,
        [
            OwnedEvent::weighted(vec![], vec![0.25], 0.5),
            OwnedEvent::weighted(vec![], vec![0.75], 1.5),
            OwnedEvent::weighted(vec![], vec![1.25], 2.0),
        ],
    )
    .unwrap();
    let actual = automatic
        .evaluate_cache_with_gradient(&params, &automatic.cache_event_batch(&batch).unwrap())
        .unwrap();
    let expected = interpreted
        .evaluate_cache_with_gradient(&params, &interpreted.cache_event_batch(&batch).unwrap())
        .unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(
            (actual.value() - expected.value()).norm() < 1.0e-12,
            "{} != {}",
            actual.value(),
            expected.value()
        );
        for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
            assert!((actual - expected).norm() < 1.0e-12);
        }
    }
    let dataset = Dataset::from_batch(batch);
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            threads: crate::ThreadPolicy::Serial,
            ..crate::CpuOptions::default()
        }),
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let automatic_data = automatic.prepare_dataset(&execution, &dataset).unwrap();
    let interpreted_data = interpreted.prepare_dataset(&execution, &dataset).unwrap();
    let automatic_result = automatic
        .reduce_with_gradient(
            &execution,
            &params,
            &automatic_data,
            ReductionPlan::weighted_log_positive_real(),
        )
        .unwrap();
    let interpreted_result = interpreted
        .reduce_with_gradient(
            &execution,
            &params,
            &interpreted_data,
            ReductionPlan::weighted_log_positive_real(),
        )
        .unwrap();

    assert!((automatic_result.value() - interpreted_result.value()).abs() < 1.0e-12);
    for (actual, expected) in automatic_result
        .gradient()
        .iter()
        .zip(interpreted_result.gradient())
    {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f32_auto_jit_matches_f32_interpreter_for_parameter_arithmetic() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 16_777_216.0));
    let y = laddu_expr::Expr::from(parameter!("y", initial: 1.0));
    let model = CompiledModel::from_expr(&(x + y)).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f32_jit_and_interpreter(&model);
    let GradientExecutor::Jit(kernel) = &automatic.gradient_executor else {
        unreachable!("f32_jit_and_interpreter already requires a gradient JIT")
    };
    assert_eq!(kernel.compiled_component_count(), 1);

    assert_eq!(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap()
    );
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f32_auto_jit_matches_f32_interpreter_for_unary_and_binary_ops() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 0.8));
    let y = laddu_expr::Expr::from(parameter!("y", initial: -0.35));
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
        + complex(x.clone(), y.clone()) / complex(1.25, -0.5);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f32_jit_and_interpreter(&model);

    assert_complex_close(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap(),
    );
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f32_auto_jit_matches_f32_interpreter_for_cached_linear_algebra() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
    let matrix = matrix([
        [event_scalar("x") + 2.0, complex(0.25, -0.5)],
        [0.5.into(), scale.clone() + 3.0],
    ]);
    let rhs = vector([1.0.into(), scale]);
    let model = CompiledModel::from_expr(&dot(
        vector([1.0.into(), complex(0.0, 1.0)]),
        solve(matrix, rhs),
    ))
    .unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f32_jit_and_interpreter(&model);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.75])],
    )
    .unwrap();
    let automatic_cache = automatic.cache_event_batch(&batch).unwrap();
    let interpreted_cache = interpreted.cache_event_batch(&batch).unwrap();

    let actual = automatic.evaluate_cache(&params, &automatic_cache).unwrap();
    let expected = interpreted
        .evaluate_cache(&params, &interpreted_cache)
        .unwrap();
    assert_complex_slices_close(&actual, &expected);
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f32_auto_jit_matches_f32_interpreter_for_event_cache_ops() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.75));
    let x = event_scalar("x");
    let y = event_scalar("y");
    let expression = ((x.clone() + scale.clone()).sin()
        + (y.clone() - 0.25).cos()
        + (x.clone() * y.clone()).exp()
        + atan2(y.clone(), x.clone() + 1.0))
        / complex(scale, -0.5);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f32_jit_and_interpreter(&model);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.25, -0.5]),
            OwnedEvent::new(vec![], vec![0.75, 0.125]),
            OwnedEvent::new(vec![], vec![1.5, 0.5]),
        ],
    )
    .unwrap();
    let automatic_cache = automatic.cache_event_batch(&batch).unwrap();
    let interpreted_cache = interpreted.cache_event_batch(&batch).unwrap();

    assert_complex_slices_close(
        &automatic.evaluate_cache(&params, &automatic_cache).unwrap(),
        &interpreted
            .evaluate_cache(&params, &interpreted_cache)
            .unwrap(),
    );
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f32_auto_jit_matches_f32_interpreter_for_reductions_and_gradients() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
    let offset = laddu_expr::Expr::from(parameter!("offset", initial: 0.5));
    let x = event_scalar("x");
    let expression = (x.clone() * scale.clone() + offset.clone()).sin()
        + complex(scale, offset).norm_sqr()
        + 2.0;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f32_jit_and_interpreter(&model);
    let auto_execution = f32_execution(JitPolicy::Auto);
    let interpreter_execution = f32_execution(JitPolicy::Disabled);
    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap()),
            [
                OwnedEvent::weighted(vec![], vec![0.25], 0.5),
                OwnedEvent::weighted(vec![], vec![0.75], 1.5),
                OwnedEvent::weighted(vec![], vec![1.25], 2.0),
            ],
        )
        .unwrap(),
    );
    let automatic_data = automatic
        .prepare_dataset(&auto_execution, &dataset)
        .unwrap();
    let interpreted_data = interpreted
        .prepare_dataset(&interpreter_execution, &dataset)
        .unwrap();

    let actual = automatic
        .reduce_with_gradient(
            &auto_execution,
            &params,
            &automatic_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    let expected = interpreted
        .reduce_with_gradient(
            &interpreter_execution,
            &params,
            &interpreted_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    assert!((actual.value() - expected.value()).abs() < 1.0e-6);
    assert_eq!(actual.gradient().len(), expected.gradient().len());
    for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
}
#[cfg(feature = "jit")]
#[test]
fn auto_jit_matches_interpreter_for_supported_real_arithmetic() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 2.0));
    let y = laddu_expr::Expr::from(parameter!("y", initial: -0.5));
    let expr = (x * 3.0 + y) / 2.0;
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

    assert!(matches!(
        automatic.scalar_executor,
        Some(ScalarExecutor::Jit(_))
    ));
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));
    assert_eq!(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap()
    );
    assert_eq!(
        automatic.evaluate_with_gradient(&params).unwrap(),
        interpreted.evaluate_with_gradient(&params).unwrap()
    );
}
#[cfg(feature = "jit")]
#[test]
fn auto_jit_supports_complex_transcendentals() {
    let expr = laddu_expr::Expr::from(parameter!("x", initial: 2.0)).exp();
    let model = CompiledModel::from_expr(&expr).unwrap();
    let plan = CpuBackend.prepare(&model);

    assert!(matches!(plan.scalar_executor, Some(ScalarExecutor::Jit(_))));
    let params = Arc::new(model.params().clone()).default_values();
    assert_eq!(plan.evaluate(&params).unwrap(), Complex64::from(2.0).exp());
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f64_auto_jit_matches_interpreter_for_unary_binary_ops_and_gradients() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 0.8));
    let y = laddu_expr::Expr::from(parameter!("y", initial: -0.35));
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
        + complex(x.clone(), y.clone()) / complex(1.25, -0.5);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f64_jit_and_interpreter(&model);

    assert_complex_close_f64(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap(),
    );
    let actual_gradient = automatic.evaluate_with_gradient(&params).unwrap();
    let expected_gradient = interpreted.evaluate_with_gradient(&params).unwrap();
    assert_complex_close_f64(actual_gradient.value(), expected_gradient.value());
    assert_complex_slices_close_f64(actual_gradient.gradient(), expected_gradient.gradient());
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f64_auto_jit_matches_interpreter_for_event_cache_ops_and_gradients() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.75));
    let x = event_scalar("x");
    let y = event_scalar("y");
    let expression = ((x.clone() + scale.clone()).sin()
        + (y.clone() - 0.25).cos()
        + (x.clone() * y.clone()).exp()
        + atan2(y.clone(), x.clone() + 1.0))
        / complex(scale, -0.5);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f64_jit_and_interpreter(&model);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.25, -0.5]),
            OwnedEvent::new(vec![], vec![0.75, 0.125]),
            OwnedEvent::new(vec![], vec![1.5, 0.5]),
        ],
    )
    .unwrap();
    let automatic_cache = automatic.cache_event_batch(&batch).unwrap();
    let interpreted_cache = interpreted.cache_event_batch(&batch).unwrap();

    assert_complex_slices_close_f64(
        &automatic.evaluate_cache(&params, &automatic_cache).unwrap(),
        &interpreted
            .evaluate_cache(&params, &interpreted_cache)
            .unwrap(),
    );
    for (actual, expected) in automatic
        .evaluate_cache_with_gradient(&params, &automatic_cache)
        .unwrap()
        .iter()
        .zip(
            interpreted
                .evaluate_cache_with_gradient(&params, &interpreted_cache)
                .unwrap(),
        )
    {
        assert_complex_close_f64(actual.value(), expected.value());
        assert_complex_slices_close_f64(actual.gradient(), expected.gradient());
    }
}
#[cfg(feature = "jit")]
#[test]
fn cpu_f64_auto_jit_matches_interpreter_for_reductions_and_gradients() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
    let offset = laddu_expr::Expr::from(parameter!("offset", initial: 0.5));
    let x = event_scalar("x");
    let expression = (x.clone() * scale.clone() + offset.clone()).sin()
        + complex(scale, offset).norm_sqr()
        + 2.0;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let (automatic, interpreted) = f64_jit_and_interpreter(&model);
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));
    let execution = Execution::default();
    let interpreter_execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            jit: JitPolicy::Disabled,
            ..crate::CpuOptions::default()
        }),
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap()),
            [
                OwnedEvent::weighted(vec![], vec![0.25], 0.5),
                OwnedEvent::weighted(vec![], vec![0.75], 1.5),
                OwnedEvent::weighted(vec![], vec![1.25], 2.0),
            ],
        )
        .unwrap(),
    );
    let automatic_data = automatic.prepare_dataset(&execution, &dataset).unwrap();
    let interpreted_data = interpreted
        .prepare_dataset(&interpreter_execution, &dataset)
        .unwrap();

    let actual = automatic
        .reduce_with_gradient(
            &execution,
            &params,
            &automatic_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    let expected = interpreted
        .reduce_with_gradient(
            &interpreter_execution,
            &params,
            &interpreted_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    assert!((actual.value() - expected.value()).abs() < 1.0e-10);
    assert_eq!(actual.gradient().len(), expected.gradient().len());
    for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
        assert!((actual - expected).abs() < 1.0e-10);
    }
}
#[cfg(feature = "jit")]
#[test]
fn jit_gradient_reduction_handles_cached_solve_rows() {
    let expression = solve(
        matrix([
            [event_scalar("x") + 2.0, Complex64::I.into()],
            [Complex64::new(2.0, -1.0).into(), 3.0.into()],
        ]),
        vector([
            parameter!("p", initial: 1.5),
            parameter!("q", initial: -0.25),
        ]),
    )
    .component(1);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));

    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [
                OwnedEvent::new(vec![], vec![0.25]),
                OwnedEvent::new(vec![], vec![0.75]),
                OwnedEvent::new(vec![], vec![1.25]),
            ],
        )
        .unwrap(),
    );
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            threads: crate::ThreadPolicy::Fixed(2),
            ..crate::CpuOptions::default()
        }),
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let automatic_data = automatic.prepare_dataset(&execution, &dataset).unwrap();
    let interpreted_data = interpreted.prepare_dataset(&execution, &dataset).unwrap();
    let actual = automatic
        .reduce_with_gradient(
            &execution,
            &params,
            &automatic_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    let expected = interpreted
        .reduce_with_gradient(
            &execution,
            &params,
            &interpreted_data,
            ReductionPlan::weighted_real(),
        )
        .unwrap();

    assert!((actual.value() - expected.value()).abs() < 1.0e-12);
    for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}
#[cfg(feature = "jit")]
#[test]
fn auto_jit_matches_interpreter_for_complex_linear_algebra() {
    let diagonal = laddu_expr::Expr::from(parameter!("diagonal", initial: 4.0));
    let matrix = matrix([
        [complex(2.0, 0.5), complex(0.25, -0.1)],
        [complex(-0.2, 0.3), diagonal],
    ]);
    let rhs = vector([complex(8.0, 1.0), complex(12.0, -0.5)]);
    let solution = solve(matrix, rhs);
    let expression = dot(solution, vector([complex(1.0, -0.2), complex(0.5, 0.3)])).exp();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

    assert!(matches!(
        automatic.scalar_executor,
        Some(ScalarExecutor::Jit(_))
    ));
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));
    let actual = automatic.evaluate(&params).unwrap();
    let expected = interpreted.evaluate(&params).unwrap();
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "{actual} != {expected}"
    );
    let actual = automatic.evaluate_with_gradient(&params).unwrap();
    let expected = interpreted.evaluate_with_gradient(&params).unwrap();
    assert!((actual.value() - expected.value()).norm() < 1.0e-12);
    for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}
#[cfg(feature = "jit")]
#[test]
fn auto_jit_gradients_support_parameter_dependent_solve() {
    let x = event_scalar("x");
    let coupling = laddu_expr::Expr::from(parameter!("coupling", initial: 0.2));
    let drive = laddu_expr::Expr::from(parameter!("drive", initial: -0.4));
    let matrix = matrix([
        [x.clone() + 2.0, complex(coupling.clone(), 0.1)],
        [complex(-0.3, coupling), 3.0.into()],
    ]);
    let expression = solve(
        matrix,
        vector([x.clone().sin() + drive, complex(x.cos(), 0.5)]),
    )
    .component(1)
    .norm_sqr();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.25]),
            OwnedEvent::new(vec![], vec![0.75]),
            OwnedEvent::new(vec![], vec![1.25]),
        ],
    )
    .unwrap();
    let automatic_cache = automatic.cache_event_batch(&batch).unwrap();
    let interpreted_cache = interpreted.cache_event_batch(&batch).unwrap();

    assert!(matches!(
        automatic.scalar_executor,
        Some(ScalarExecutor::Jit(_))
    ));
    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));
    let actual = automatic.evaluate_cache(&params, &automatic_cache).unwrap();
    let expected = interpreted
        .evaluate_cache(&params, &interpreted_cache)
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - expected).norm() < 1.0e-12,
            "{actual} != {expected}"
        );
    }
    let actual = automatic
        .evaluate_cache_with_gradient(&params, &automatic_cache)
        .unwrap();
    let expected = interpreted
        .evaluate_cache_with_gradient(&params, &interpreted_cache)
        .unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual.value() - expected.value()).norm() < 1.0e-12);
        for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
            assert!((actual - expected).norm() < 1.0e-12);
        }
    }
    let ir_interpreter = gradient_interpreter::GradientInterpreter::new(
        automatic.scalar_kernel.as_ref().unwrap(),
        model.params().free_params(),
    )
    .unwrap();
    for (row, expected) in expected.iter().enumerate() {
        let actual = ir_interpreter
            .evaluate(&params, Some((&automatic_cache, row)))
            .unwrap()
            .1;
        for (actual, expected) in actual.iter().zip(expected.gradient()) {
            assert!((actual - expected).norm() < 1.0e-12);
        }
    }
    for (row, actual) in actual.iter().enumerate() {
        for parameter in 0..params.layout().n_free() {
            let h = 1.0e-6;
            let id = params.layout().free_params()[parameter];
            let free_id = params.layout().free_id(id).unwrap().unwrap();
            let value = params.get(id).unwrap();
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.set_free(free_id, value + h).unwrap();
            minus.set_free(free_id, value - h).unwrap();
            let expected = (automatic.evaluate_cache(&plus, &automatic_cache).unwrap()[row]
                - automatic.evaluate_cache(&minus, &automatic_cache).unwrap()[row])
                / (2.0 * h);
            assert!((actual.gradient()[parameter] - expected).norm() < 1.0e-8);
        }
    }
}
#[cfg(feature = "jit")]
#[test]
fn jit_and_interpreter_reject_singular_parameter_dependent_solve() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.0));
    let expression = solve(
        matrix([[scale.clone(), 2.0.into()], [scale * 2.0, 4.0.into()]]),
        vector([1.0, 2.0]),
    )
    .component(0);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

    assert!(matches!(
        automatic.gradient_executor,
        GradientExecutor::Jit(_)
    ));
    assert!(automatic.evaluate_with_gradient(&params).is_err());
    assert!(interpreted.evaluate_with_gradient(&params).is_err());
}
