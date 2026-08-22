use super::*;

#[test]
fn cpu_f32_evaluates_computed_event_cache_entries() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 2.0));
    let model =
        CompiledModel::from_expr(&(event_scalar("x").sin() * scale + 16_777_216.0)).unwrap();
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
    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![1.0])],
        )
        .unwrap(),
    );
    let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();

    let reduction = plan
        .reduce_with_gradient(
            &execution,
            &params,
            &prepared,
            ReductionPlan::weighted_real(),
        )
        .unwrap();
    assert_eq!(reduction.value(), 16_777_218.0);
    assert_eq!(reduction.gradient(), &[(1.0_f32.sin() as f64)]);
}
#[test]
fn batch_cache_evaluates_without_original_event_batch() {
    let expr = event_scalar("x").real().sin() * parameter!("scale", initial: 2.0);
    let model = CompiledModel::from_expr(&expr).unwrap();
    let layout = Arc::new(model.params().clone());
    let mut params = layout.default_values();
    let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.5]),
            OwnedEvent::new(vec![], vec![1.0]),
        ],
    )
    .unwrap();
    let cache = plan.cache_event_batch(&batch).unwrap();

    assert_eq!(cache.weights(), &[1.0, 1.0]);
    assert!(matches!(&cache.slots[0], CachedSlot::Real(values) if values.len() == 2));
    assert_eq!(cache.slots[0].resident_bytes(), 2 * size_of::<f64>());
    assert_eq!(
        plan.evaluate_cache(&params, &cache).unwrap(),
        vec![
            Complex64::from(2.0 * 0.5_f64.sin()),
            Complex64::from(2.0 * 1.0_f64.sin())
        ]
    );

    let scale = layout
        .free_id(layout.id("scale").unwrap())
        .unwrap()
        .unwrap();
    params.set_free(scale, 3.0).unwrap();
    assert_eq!(
        plan.evaluate_cache(&params, &cache).unwrap(),
        vec![
            Complex64::from(3.0 * 0.5_f64.sin()),
            Complex64::from(3.0 * 1.0_f64.sin())
        ]
    );
}
#[test]
fn real_cache_slots_use_half_the_scalar_payload_of_complex_slots() {
    let real_model =
        CompiledModel::from_expr(&(parameter!("scale") * event_scalar("x").real().sin())).unwrap();
    let x = event_scalar("x");
    let complex_model =
        CompiledModel::from_expr(&(parameter!("scale") * complex(x.clone().sin(), x.cos())))
            .unwrap();
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.5]),
            OwnedEvent::new(vec![], vec![1.0]),
        ],
    )
    .unwrap();
    let real_cache = CpuBackend
        .prepare(&real_model)
        .cache_event_batch(&batch)
        .unwrap();
    let complex_cache = CpuBackend
        .prepare(&complex_model)
        .cache_event_batch(&batch)
        .unwrap();

    assert!(matches!(&real_cache.slots[0], CachedSlot::Real(_)));
    assert!(matches!(&complex_cache.slots[0], CachedSlot::Complex(_)));
    assert_eq!(
        complex_cache.slots[0].resident_bytes(),
        2 * real_cache.slots[0].resident_bytes()
    );
}
#[test]
fn selected_event_only_solve_components_cache_inverse_rows() {
    let expression = solve(
        matrix([[event_scalar("x") + 2.0]]),
        vector([parameter!("rhs", initial: 3.0)]),
    )
    .component(0);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let scalar_plan = plan.scalar_interpreter_plan().unwrap();
    assert!(!scalar_plan.invariant_instructions.is_empty());
    assert!(!scalar_plan.event_instructions.is_empty());
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [
            OwnedEvent::new(vec![], vec![0.0]),
            OwnedEvent::new(vec![], vec![1.0]),
        ],
    )
    .unwrap();
    let cache = plan.cache_event_batch(&batch).unwrap();

    assert!(cache.factor_slots.is_empty());
    assert_eq!(cache.solve_row_slots.len(), 1);
    assert_eq!(cache.solve_row_slots[0].values.len(), 2);
    assert!(cache.resident_bytes() > 0);
    let first = plan
        .evaluate_cache_row_with_gradient(&params, &cache, 0)
        .unwrap();
    let second = plan
        .evaluate_cache_row_with_gradient(&params, &cache, 1)
        .unwrap();
    assert_eq!(first.value(), Complex64::from(1.5));
    assert_eq!(first.gradient(), &[Complex64::from(0.5)]);
    assert_eq!(second.value(), Complex64::from(1.0));
    assert_eq!(second.gradient(), &[Complex64::from(1.0 / 3.0)]);
}
#[test]
fn cached_solve_component_matches_general_complex_nonsymmetric_solve() {
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
    let plan = CpuBackend.prepare(&model);
    assert!(plan.solve_components.iter().any(Option::is_some));

    let event = HashMap::from([("x".to_owned(), 0.75)]);
    let direct = plan
        .evaluate_with_event_and_gradient(&params, &event)
        .unwrap();
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.75])],
    )
    .unwrap();
    let cache = plan.cache_event_batch(&batch).unwrap();
    let cached = plan
        .evaluate_cache_row_with_gradient(&params, &cache, 0)
        .unwrap();
    let ir_gradient = gradient_interpreter::GradientInterpreter::new(
        plan.scalar_kernel.as_ref().unwrap(),
        model.params().free_params(),
    )
    .unwrap()
    .evaluate(&params, Some((&cache, 0)))
    .unwrap()
    .1;

    assert!((cached.value() - direct.value()).norm() < 1.0e-12);
    for (cached, direct) in cached.gradient().iter().zip(direct.gradient()) {
        assert!((cached - direct).norm() < 1.0e-12);
    }
    for (actual, expected) in ir_gradient.iter().zip(cached.gradient()) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}
#[test]
fn reverse_cached_solve_component_matches_forward_nonsymmetric_solve() {
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
    let forward = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    let reverse = CpuBackend
        .prepare_with_autodiff_mode(&model, AutodiffMode::Reverse)
        .unwrap();
    assert!(reverse.solve_components.iter().any(Option::is_some));

    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.75])],
    )
    .unwrap();
    let expected = forward
        .evaluate_cache_row_with_gradient(&params, &forward.cache_event_batch(&batch).unwrap(), 0)
        .unwrap();
    let actual = reverse
        .evaluate_cache_row_with_gradient(&params, &reverse.cache_event_batch(&batch).unwrap(), 0)
        .unwrap();

    assert!((actual.value() - expected.value()).norm() < 1.0e-12);
    assert_gradient_close(actual.gradient(), expected.gradient(), 1.0e-12);
}
#[test]
fn batch_cache_reports_missing_event_columns() {
    let expr = event_scalar("missing");
    let model = CompiledModel::from_expr(&expr).unwrap();
    let plan = CpuBackend.prepare(&model);
    let batch = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
        [OwnedEvent::new(vec![], vec![0.5])],
    )
    .unwrap();

    assert!(matches!(
        plan.cache_event_batch(&batch),
        Err(RuntimeError::MissingEventColumn(name)) if name == "missing"
    ));
}
#[test]
fn cached_dataset_preserves_transformed_batches_and_weights() {
    let expr = event_scalar("x") * parameter!("scale", initial: 2.0);
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let batch = EventBatch::from_events(
        schema,
        [
            OwnedEvent::weighted(vec![], vec![0.5], 2.0),
            OwnedEvent::weighted(vec![], vec![1.0], 3.0),
        ],
    )
    .unwrap();
    let dataset = Dataset::from_batch(batch).filter(|event| event.scalar(0) > 0.75);
    let cached = plan.cache_dataset(&dataset).unwrap();

    assert_eq!(cached.len(), 1);
    assert_eq!(cached.batches()[0].weights(), &[3.0]);
    assert_eq!(cached.batches()[0].sum_weights(), 3.0);
    assert_eq!(
        plan.evaluate_cached_dataset(&params, &cached).unwrap(),
        vec![Complex64::from(2.0)]
    );
}
#[test]
fn cached_dataset_weighted_reductions_match_dataset_path() {
    let expr = event_scalar("x") * parameter!("scale", initial: 2.0);
    let model = CompiledModel::from_expr(&expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let first = EventBatch::from_events(
        Arc::clone(&schema),
        [
            OwnedEvent::weighted(vec![], vec![1.0], 2.0),
            OwnedEvent::weighted(vec![], vec![2.0], 3.0),
        ],
    )
    .unwrap();
    let second =
        EventBatch::from_events(schema, [OwnedEvent::weighted(vec![], vec![3.0], 4.0)]).unwrap();
    let dataset = Dataset::from_batches(vec![first, second]).unwrap();
    let cached = plan.cache_dataset(&dataset).unwrap();

    let expected = dataset.weighted_sum(|event| 2.0 * event.scalar(0)).unwrap();
    assert_eq!(cached.sum_weights(), dataset.sum_weights().unwrap());
    assert_eq!(
        plan.weighted_sum_cached(&params, &cached, |value| value.re)
            .unwrap(),
        expected
    );
    assert_eq!(
        plan.weighted_complex_sum_cached(&params, &cached, |value| value * Complex64::I)
            .unwrap(),
        Complex64::I * expected
    );
    assert_eq!(
        plan.par_weighted_sum_cached(&params, &cached, |value| value.re)
            .unwrap(),
        expected
    );
    assert_eq!(
        plan.par_weighted_complex_sum_cached(&params, &cached, |value| value * Complex64::I)
            .unwrap(),
        Complex64::I * expected
    );
    let serial_gradient = plan
        .try_weighted_real_sum_with_gradient_cached(&params, &cached, |value| {
            Ok::<_, RuntimeError>((value.re.powi(2), 2.0 * value.re))
        })
        .unwrap();
    let parallel_gradient = plan
        .par_try_weighted_real_sum_with_gradient_cached(&params, &cached, |value| {
            Ok::<_, RuntimeError>((value.re.powi(2), 2.0 * value.re))
        })
        .unwrap();
    assert_eq!(serial_gradient, parallel_gradient);
}
