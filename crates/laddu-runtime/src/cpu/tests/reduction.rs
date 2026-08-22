use super::*;

#[test]
fn resident_cache_plan_accounts_for_batch_overhead_and_source_prefetch() {
    // 100 events require 1,000 cache bytes. With 100 bytes of fixed
    // overhead per cached batch and 20 source bytes per event, a
    // 2,000-byte budget first suggests 45 events, then accounts for three
    // fixed-overhead batches and converges to a 35-event chunk.
    let (resident, chunk) = resident_cache_plan(100, 10, 20, 100, 2_000).unwrap();
    let batches = 100_usize.div_ceil(chunk);
    assert_eq!(resident, 1_000 + 100 * batches);
    assert!(resident + 20 * chunk <= 2_000);
    assert!(resident_cache_plan(100, 10, 20, 100, 1_119).is_none());
}
#[test]
fn cpu_f32_reduces_event_scalar_arithmetic_with_f64_accumulation() {
    let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.0));
    let model = CompiledModel::from_expr(&(event_scalar("x") + scale)).unwrap();
    let params = model.params().default_values();
    let execution = Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            threads: crate::ThreadPolicy::Serial,
            ..crate::CpuOptions::default()
        }),
        precision: Precision::F32,
        ..crate::ExecutionOptions::default()
    })
    .unwrap();
    let plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![16_777_216.0])],
        )
        .unwrap(),
    );
    let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();

    assert_eq!(
        plan.reduce(
            &execution,
            &params,
            &prepared,
            ReductionPlan::weighted_real(),
        )
        .unwrap(),
        16_777_216.0
    );
}
#[test]
fn reduction_plans_match_across_storage_and_thread_policies() {
    let expr = event_scalar("x") * parameter!("scale", initial: 2.0) + 1.0;
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
    let resident = Dataset::from_batches(vec![first, second]).unwrap();
    let datasets = [resident.clone(), resident.streaming()];
    let executions = [
        Execution::local(crate::ExecutionOptions {
            device: crate::Device::Cpu(crate::CpuOptions {
                threads: crate::ThreadPolicy::Serial,
                ..crate::CpuOptions::default()
            }),
            ..crate::ExecutionOptions::default()
        })
        .unwrap(),
        Execution::local(crate::ExecutionOptions {
            device: crate::Device::Cpu(crate::CpuOptions {
                threads: crate::ThreadPolicy::Fixed(2),
                ..crate::CpuOptions::default()
            }),
            ..crate::ExecutionOptions::default()
        })
        .unwrap(),
    ];
    let expected_real = 49.0;
    let expected_log = 2.0 * 3.0_f64.ln() + 3.0 * 5.0_f64.ln() + 4.0 * 7.0_f64.ln();
    let expected_log_gradient = 2.0 / 3.0 + 6.0 / 5.0 + 12.0 / 7.0;

    for execution in &executions {
        for dataset in &datasets {
            let prepared = plan.prepare_dataset(execution, dataset).unwrap();
            assert_eq!(
                plan.reduce(
                    execution,
                    &params,
                    &prepared,
                    ReductionPlan::weighted_real(),
                )
                .unwrap(),
                expected_real
            );
            assert_eq!(
                plan.reduce(
                    execution,
                    &params,
                    &prepared,
                    ReductionPlan::weighted_positive_real(),
                )
                .unwrap(),
                expected_real
            );
            let evaluation = plan
                .reduce_with_gradient(
                    execution,
                    &params,
                    &prepared,
                    ReductionPlan::weighted_log_positive_real(),
                )
                .unwrap();
            assert!((evaluation.value() - expected_log).abs() < 1.0e-12);
            assert!((evaluation.gradient()[0] - expected_log_gradient).abs() < 1.0e-12);
        }
    }
}
#[test]
fn positive_reduction_reports_the_invalid_value() {
    let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    let plan = CpuBackend.prepare(&model);
    let dataset = Dataset::from_batch(
        EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![-2.0])],
        )
        .unwrap(),
    );
    let execution = Execution::default();
    let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();

    assert!(matches!(
        plan.reduce(
            &execution,
            &params,
            &prepared,
            ReductionPlan::weighted_log_positive_real(),
        ),
        Err(RuntimeError::Reduction(
            laddu_compile::ReductionError::NonPositiveValue {
                transform: laddu_compile::ReductionTransform::LogPositiveReal,
                value: -2.0,
            }
        ))
    ));
}
