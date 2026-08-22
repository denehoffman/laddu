use super::*;
use laddu_kernel::ir::{KernelInstruction, KernelValueKind};

#[test]
fn evaluates_scalar_expression_with_parameters() {
    let expr = (2.0 * parameter!("x", initial: 3.0)
        + complex(
            parameter!("re", initial: 1.0),
            parameter!("im", initial: 2.0),
        ))
    .norm_sqr();

    assert_eq!(evaluate(&expr), Complex64::from(53.0));
}
#[test]
fn evaluates_event_scalars() {
    let expr = laddu_expr::event_scalar("x") * 2.0;
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let event = HashMap::from([("x".to_owned(), 3.0)]);

    assert_eq!(
        plan.evaluate_with_event(&params, &event).unwrap(),
        Complex64::from(6.0)
    );
}
#[test]
fn scalar_kernel_ir_preserves_typed_dependency_classes() {
    let coefficient = complex(parameter!("re", initial: 2.0), 1.0);
    let expr = coefficient * event_scalar("x");
    let model = CompiledModel::from_expr(&expr).unwrap();
    let plan = CpuBackend.prepare(&model);
    let kernel = plan.scalar_kernel.as_ref().unwrap();

    assert!(
        kernel
            .values()
            .iter()
            .any(|value| value.class == KernelValueClass::Invariant)
    );
    assert!(
        kernel
            .values()
            .iter()
            .any(|value| value.class == KernelValueClass::Event)
    );
    let root = &kernel.values()[kernel.root().index()];
    assert_eq!(root.kind, KernelValueKind::Complex);
    assert_eq!(root.class, KernelValueClass::Event);
    assert!(matches!(root.instruction, KernelInstruction::Mul(_)));
}
#[test]
fn cpu_execution_mode_selects_retained_interpreter() {
    let expr = laddu_expr::Expr::from(parameter!("x", initial: 2.0)).exp() + 1.0;
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            jit: crate::JitPolicy::Disabled,
            ..crate::CpuOptions::default()
        }),
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let configured = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();

    assert!(matches!(
        interpreted.scalar_executor,
        Some(ScalarExecutor::Interpreter(_))
    ));
    assert!(matches!(
        configured.scalar_executor,
        Some(ScalarExecutor::Interpreter(_))
    ));
    assert!(matches!(
        configured.gradient_executor,
        GradientExecutor::Interpreter(_)
    ));
    assert_eq!(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap()
    );

    let empty_model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
    let wrong_params = empty_model.params().default_values();
    assert!(matches!(
        automatic.evaluate(&wrong_params),
        Err(RuntimeError::Parameter(_))
    ));
}
#[test]
fn cpu_plan_executes_parameter_only_scalar_kernel_in_f32() {
    let x = laddu_expr::Expr::from(parameter!("x", initial: 16_777_216.0));
    let y = laddu_expr::Expr::from(parameter!("y", initial: 1.0));
    let model = CompiledModel::from_expr(&(x + y)).unwrap();
    let params = model.params().default_values();
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions::default()),
        precision: Precision::F32,
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let f32_plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let f64_plan = CpuBackend.prepare(&model);

    assert_eq!(f32_plan.evaluate(&params).unwrap().re, 16_777_216.0);
    assert_eq!(f64_plan.evaluate(&params).unwrap().re, 16_777_217.0);
    let gradient = f32_plan.evaluate_with_gradient(&params).unwrap();
    assert_eq!(gradient.value().re, 16_777_216.0);
    assert_eq!(gradient.gradient(), &[Complex64::ONE, Complex64::ONE]);
}
#[test]
fn cpu_f32_evaluates_complex_linear_algebra() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
    let lhs = matrix([
        [scale.clone() + 2.0, complex(0.25, -0.5)],
        [0.5.into(), scale.clone() + 3.0],
    ]);
    let rhs = vector([1.0.into(), scale]);
    let model = CompiledModel::from_expr(&dot(
        vector([1.0.into(), complex(0.0, 1.0)]),
        solve(lhs, rhs),
    ))
    .unwrap();
    let params = model.params().default_values();
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions::default()),
        precision: Precision::F32,
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let f32_plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let f64_plan = CpuBackend.prepare(&model);

    let actual = f32_plan.evaluate(&params).unwrap();
    let expected = f64_plan.evaluate(&params).unwrap();
    assert!((actual.re - expected.re).abs() < 1.0e-6);
    assert!((actual.im - expected.im).abs() < 1.0e-6);
}
#[test]
fn evaluates_p4_schema_components_and_atan2() {
    let expr = event_p4_component("ks1", P4Component::E)
        + event_p4_component("ks1", P4Component::Px)
        + atan2(
            event_p4_component("ks1", P4Component::Py),
            event_p4_component("ks1", P4Component::Px),
        );
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(["ks1"], std::iter::empty::<&str>(), false).unwrap()),
        [OwnedEvent::new(
            vec![RealVec4::new(10.0, 3.0, 4.0, 5.0)],
            vec![],
        )],
    )
    .unwrap();

    assert_eq!(
        plan.evaluate_batch(&params, &batch).unwrap()[0],
        Complex64::from(13.0 + 4.0_f64.atan2(3.0))
    );
}
#[test]
fn evaluates_linear_algebra_nodes() {
    let a = matrix([[2.0, 0.0], [0.0, 4.0]]);
    let b = vector([8.0, 12.0]);
    let x = solve(a, b);
    let expr = dot(&x, vector([1.0, 1.0]));
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);

    assert_eq!(plan.evaluate(&params).unwrap(), Complex64::from(7.0));
    assert_eq!(plan.constant_factors.len(), 1);
    assert!(plan.constant_factors[0].get().is_some());
}
#[test]
fn optimized_and_unoptimized_plans_evaluate_the_same_expression() {
    let solved = solve(matrix([[2.0, 0.0], [0.0, 4.0]]), vector([8.0, 12.0]));
    let complex_offset = complex(
        parameter!("offset_re", initial: 1.5),
        parameter!("offset_im", initial: -0.5),
    );
    let polar_product = polar_complex(
        parameter!("mag1", initial: 2.0),
        parameter!("phase1", initial: 0.25),
    ) * polar_complex(
        parameter!("mag2", initial: 3.0),
        parameter!("phase2", initial: -0.5),
    );
    let expr = ((laddu_expr::event_scalar("mass") + 0.0) * 1.0
        + dot(solved, vector([1.0, 1.0]))
        + complex_offset.conj().real()
        + polar_product.real()
        + parameter!("unused", initial: 3.0) * 0.0)
        .norm_sqr();
    let no_optimization = CompileOptions::without_optimizations();
    let optimized = CompiledModel::from_expr(&expr).unwrap();
    let unoptimized = CompiledModel::from_expr_with_options(&expr, &no_optimization).unwrap();
    let optimized_params = Arc::new(optimized.params().clone()).default_values();
    let unoptimized_params = Arc::new(unoptimized.params().clone()).default_values();
    let event = HashMap::from([("mass".to_owned(), 2.0)]);

    let optimized = CpuBackend
        .prepare(&optimized)
        .evaluate_with_event_and_gradient(&optimized_params, &event)
        .unwrap();
    let unoptimized = CpuBackend
        .prepare(&unoptimized)
        .evaluate_with_event_and_gradient(&unoptimized_params, &event)
        .unwrap();
    assert_eq!(optimized.value(), unoptimized.value());
    for (optimized, unoptimized) in optimized.gradient().iter().zip(unoptimized.gradient()) {
        assert!((optimized - unoptimized).norm() < 1.0e-12);
    }
}
