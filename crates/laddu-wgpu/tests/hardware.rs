//! Physical-adapter parity and cache integration tests.
//!
//! Every test in this target is deliberately ignored in ordinary CI. Run the
//! full adapter/device suite only on a runner with a usable WGPU adapter:
//!
//! ```text
//! cargo test -p laddu-wgpu --test hardware -- --ignored
//! ```
#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::{
    data::{Dataset, EventBatch, OwnedEvent},
    schema::Schema,
};
use laddu_expr::{
    Expr, atan2, complex, dot, event_scalar, matmul, matrix, matvec, parameter, solve, vector,
};
use laddu_runtime::{CpuBackend, CpuOptions, Device, Execution, ExecutionOptions, Precision};
use laddu_wgpu::{WgpuBackend, WgpuOptions, WgpuPrecision, WgpuScalarKernel};

const F32_PARITY_TOLERANCE: f64 = 1.0e-5;

fn cpu_execution(precision: Precision) -> Execution {
    Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision,
        ..ExecutionOptions::default()
    })
    .unwrap()
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_scalar_kernel_matches_f32_cpu() {
    let x = Expr::from(parameter!("x", initial: 1.25));
    let y = Expr::from(parameter!("y", initial: -0.4));
    let expression = (complex(x.clone() * y.clone() + 2.0, y) * complex(x, -1.0)).norm_sqr();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let context = WgpuBackend::default()
        .open(&WgpuOptions::default(), WgpuPrecision::F32)
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let gpu = kernel.evaluate(&context, &params).unwrap();
    let execution = cpu_execution(Precision::F32);
    let cpu = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap()
        .evaluate(&params)
        .unwrap();

    assert_eq!(gpu, (cpu.re, cpu.im));
}

#[test]
#[ignore = "requires a WGPU adapter with shader f64 support"]
fn gpu_event_batch_and_reduction_match_f64_cpu() {
    let expression =
        (event_scalar("x") * parameter!("scale", initial: 1.25) + event_scalar("y") - 0.5).exp();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        (0..70).map(|index| {
            OwnedEvent::weighted(
                vec![],
                vec![index as f64 * 0.0125, 0.5 - index as f64 * 0.005],
                1.0 + index as f64 * 0.01,
            )
        }),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(&WgpuOptions::default(), WgpuPrecision::F64)
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F64,
        ..ExecutionOptions::default()
    })
    .unwrap();
    let plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let cpu = plan.evaluate_batch(&params, &batch).unwrap();

    for (gpu, cpu) in gpu.iter().zip(cpu) {
        assert!((gpu.0 - cpu.re).abs() <= 1.0e-12 * cpu.re.abs().max(1.0));
        assert!((gpu.1 - cpu.im).abs() <= 1.0e-12 * cpu.im.abs().max(1.0));
    }

    let reduction = ReductionPlan::weighted_log_positive_real();
    let gpu = kernel
        .reduce_batch(&context, &params, &batch, reduction)
        .unwrap();
    let dataset = Dataset::from_batch(batch);
    let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();
    let cpu = plan
        .reduce(&execution, &params, &prepared, reduction)
        .unwrap();
    assert!((gpu - cpu).abs() <= 1.0e-12 * cpu.abs().max(1.0));
}

#[test]
#[ignore = "requires a WGPU adapter with shader f64 support"]
fn gpu_f64_approximations_match_cpu() {
    let x = event_scalar("x");
    let y = event_scalar("y");
    let z = complex(x.clone(), y.clone());
    let expressions = [
        ("sin", x.clone().sin()),
        ("cos", x.clone().cos()),
        ("exp", x.clone().exp()),
        ("log", (x.clone() * x.clone() + 0.125).log()),
        ("atan2", atan2(y.clone(), x.clone())),
        ("complex sin", z.clone().sin()),
        ("complex cos", z.clone().cos()),
        ("complex exp", z.clone().exp()),
        ("complex log", z.log()),
    ];
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        (0..49).map(|index| {
            let x = [0.0, -20.0, -5.125, -0.3, 1.875, 10.0, 20.0][index % 7];
            let y = [-10.0, -3.0, -0.5, 0.25, 1.5, 5.0, 10.0][index / 7];
            OwnedEvent::new(vec![], vec![x, y])
        }),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(&WgpuOptions::default(), WgpuPrecision::F64)
        .unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F64,
        ..ExecutionOptions::default()
    })
    .unwrap();

    for (name, expression) in expressions {
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = model.params().default_values();
        let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
        let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
        let plan = CpuBackend
            .prepare_for_execution(&model, &execution)
            .unwrap();
        let cpu = plan.evaluate_batch(&params, &batch).unwrap();
        for (index, (gpu, cpu)) in gpu.iter().zip(cpu).enumerate() {
            let re_tolerance = 2.0e-12 * cpu.re.abs().max(1.0);
            let im_tolerance = 2.0e-12 * cpu.im.abs().max(1.0);
            assert!(
                gpu.0 == cpu.re || (gpu.0 - cpu.re).abs() <= re_tolerance,
                "{name} real mismatch at event {index}: GPU ({}, {}), CPU ({}, {})",
                gpu.0,
                gpu.1,
                cpu.re,
                cpu.im
            );
            assert!(
                gpu.1 == cpu.im || (gpu.1 - cpu.im).abs() <= im_tolerance,
                "{name} imaginary mismatch at event {index}: GPU {}, CPU {}",
                gpu.1,
                cpu.im
            );
        }
    }
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_event_batch_matches_f32_cpu_across_partial_workgroup() {
    let scale = Expr::from(parameter!("scale", initial: 1.25));
    let expression = event_scalar("x") * scale + event_scalar("y") - 0.5;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        (0..70).map(|index| {
            OwnedEvent::weighted(
                vec![],
                vec![index as f64 * 0.125, 2.0 - index as f64 * 0.01],
                1.0 + index as f64 * 0.01,
            )
        }),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(
            &WgpuOptions {
                memory_budget: Some(256),
                ..WgpuOptions::default()
            },
            WgpuPrecision::F32,
        )
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap();
    let plan = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap();
    let cpu = plan.evaluate_batch(&params, &batch).unwrap();

    assert_eq!(gpu.len(), cpu.len());
    for (gpu, cpu) in gpu.iter().zip(cpu) {
        assert_eq!(*gpu, (cpu.re, cpu.im));
    }
    let dataset = Dataset::from_batch(batch.clone());
    let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();
    for reduction in [
        ReductionPlan::weighted_real(),
        ReductionPlan::weighted_positive_real(),
        ReductionPlan::weighted_log_positive_real(),
    ] {
        let gpu = kernel
            .reduce_batch(&context, &params, &batch, reduction)
            .unwrap();
        let cpu = plan
            .reduce(&execution, &params, &prepared, reduction)
            .unwrap();
        assert!((gpu - cpu).abs() <= 1.0e-4 * cpu.abs().max(1.0));
    }

    let invalid = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap()),
        (0..12).map(|index| {
            if index == 11 {
                OwnedEvent::new(vec![], vec![0.0, 0.0])
            } else {
                OwnedEvent::new(vec![], vec![1.0, 2.0])
            }
        }),
    )
    .unwrap();
    assert!(matches!(
        kernel.reduce_batch(
            &context,
            &params,
            &invalid,
            ReductionPlan::weighted_positive_real()
        ),
        Err(laddu_wgpu::WgpuError::NonPositiveEvent(11))
    ));
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_refresh_rejects_late_chunk_mismatch_without_mutating_prepared_batch() {
    let expression = event_scalar("x") * parameter!("scale", initial: 1.25) + 2.0;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap());
    let make_batch = |len| {
        EventBatch::from_events(
            Arc::clone(&schema),
            (0..len)
                .map(|index| OwnedEvent::weighted(vec![], vec![1.0 + index as f64 * 0.01], 1.0)),
        )
        .unwrap()
    };
    let original = make_batch(129);
    let incoming = make_batch(130);
    let context = WgpuBackend::default()
        .open(
            &WgpuOptions {
                memory_budget: Some(256),
                ..WgpuOptions::default()
            },
            WgpuPrecision::F32,
        )
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let chunk_len = 64;
    assert_eq!(129_usize.div_ceil(chunk_len), 130_usize.div_ceil(chunk_len));
    assert_ne!(129 % chunk_len, 0);

    let mut prepared = kernel.prepare_batch(&context, &params, &original).unwrap();
    let prepared_len = prepared.len();
    let prepared_resident_bytes = prepared.resident_bytes();
    let reduction = ReductionPlan::weighted_real();
    let original_values = kernel.evaluate_batch(&context, &params, &original).unwrap();
    let original_reduction = kernel
        .reduce_prepared_batch(&context, &params, &prepared, reduction)
        .unwrap();

    assert!(
        !kernel
            .refresh_batch(&context, &params, &incoming, &mut prepared)
            .unwrap()
    );
    assert_eq!(prepared.len(), prepared_len);
    assert_eq!(prepared.resident_bytes(), prepared_resident_bytes);

    let refreshed_values = kernel.evaluate_batch(&context, &params, &original).unwrap();
    let refreshed_reduction = kernel
        .reduce_prepared_batch(&context, &params, &prepared, reduction)
        .unwrap();
    assert_eq!(refreshed_values, original_values);
    assert_eq!(refreshed_reduction, original_reduction);
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_refresh_packing_error_leaves_prepared_batch_unchanged() {
    let expression = event_scalar("x") * parameter!("scale", initial: 1.25) + 2.0;
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let original_schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap());
    let original = EventBatch::from_events(
        Arc::clone(&original_schema),
        (0..129).map(|index| OwnedEvent::weighted(vec![], vec![1.0 + index as f64 * 0.01], 1.0)),
    )
    .unwrap();
    let incoming = EventBatch::from_events(
        Arc::new(Schema::new(std::iter::empty::<&str>(), ["y"], false).unwrap()),
        (0..130).map(|index| OwnedEvent::weighted(vec![], vec![1.0 + index as f64 * 0.01], 1.0)),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(
            &WgpuOptions {
                memory_budget: Some(256),
                ..WgpuOptions::default()
            },
            WgpuPrecision::F32,
        )
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let mut prepared = kernel.prepare_batch(&context, &params, &original).unwrap();
    let reduction = ReductionPlan::weighted_real();
    let before = kernel
        .reduce_prepared_batch(&context, &params, &prepared, reduction)
        .unwrap();

    assert!(matches!(
        kernel.refresh_batch(&context, &params, &incoming, &mut prepared),
        Err(laddu_wgpu::WgpuError::MissingEventColumn(name)) if name == "x"
    ));
    let after = kernel
        .reduce_prepared_batch(&context, &params, &prepared, reduction)
        .unwrap();
    assert_eq!(after, before);
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_refresh_rebases_singular_cache_error_to_global_event_index() {
    let expression = solve(
        matrix([[event_scalar("x"), 0.0.into()], [0.0.into(), 1.0.into()]]),
        vector([1.0, 1.0]),
    )
    .component(0);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap());
    let chunk_values = |singular_at: Option<usize>| {
        (0..129).map(move |index| {
            let value = if singular_at == Some(index) { 0.0 } else { 1.0 };
            OwnedEvent::new(vec![], vec![value])
        })
    };
    let original = EventBatch::from_events(Arc::clone(&schema), chunk_values(None)).unwrap();
    let context = WgpuBackend::default()
        .open(
            &WgpuOptions {
                memory_budget: Some(256),
                ..WgpuOptions::default()
            },
            WgpuPrecision::F32,
        )
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let chunk_len = 64;
    let singular_index = chunk_len + 1;
    assert!(singular_index < 129);
    let incoming =
        EventBatch::from_events(Arc::clone(&schema), chunk_values(Some(singular_index))).unwrap();
    let mut prepared = kernel.prepare_batch(&context, &params, &original).unwrap();

    assert!(matches!(
        kernel.refresh_batch(&context, &params, &incoming, &mut prepared),
        Err(laddu_wgpu::WgpuError::SingularMatrixEvent(index)) if index == singular_index
    ));
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_materializes_computed_event_cache() {
    let expression = (event_scalar("x").sin() + event_scalar("y").cos()).exp();
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        (0..70).map(|index| {
            OwnedEvent::new(
                vec![],
                vec![index as f64 * 0.03125, 1.0 - index as f64 * 0.0125],
            )
        }),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(&WgpuOptions::default(), WgpuPrecision::F32)
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();

    let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap();
    let cpu = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap()
        .evaluate_batch(&params, &batch)
        .unwrap();

    for (gpu, cpu) in gpu.iter().zip(cpu) {
        assert!((gpu.0 - cpu.re).abs() <= F32_PARITY_TOLERANCE * cpu.re.abs().max(1.0));
        assert!((gpu.1 - cpu.im).abs() <= F32_PARITY_TOLERANCE * cpu.im.abs().max(1.0));
    }
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_aggregate_algebra_matches_f32_cpu_with_cached_rectangular_matrices() {
    let cached_matrix = matrix([
        [
            event_scalar("x"),
            event_scalar("y"),
            event_scalar("x") + 1.0,
        ],
        [
            event_scalar("y") - 0.5,
            event_scalar("x") * 2.0,
            event_scalar("y") + 2.0,
        ],
    ]);
    let projected = matvec(
        cached_matrix.clone(),
        vector([
            parameter!("a", initial: 0.5),
            parameter!("b", initial: -0.25),
            parameter!("c", initial: 1.5),
        ]),
    );
    let remixed = matmul(
        matrix([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]),
        cached_matrix,
    )
    .matrix_element(2, 1);
    let expression = dot(projected, vector([1.0, -2.0])) + remixed;
    let model = CompiledModel::from_expr_with_options(
        &expression,
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        (0..70).map(|index| {
            OwnedEvent::new(
                vec![],
                vec![index as f64 * 0.03125, 1.0 - index as f64 * 0.0125],
            )
        }),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(
            &WgpuOptions {
                memory_budget: Some(512),
                ..WgpuOptions::default()
            },
            WgpuPrecision::F32,
        )
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap();
    let cpu = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap()
        .evaluate_batch(&params, &batch)
        .unwrap();

    for (gpu, cpu) in gpu.iter().zip(cpu) {
        assert!((gpu.0 - cpu.re).abs() <= F32_PARITY_TOLERANCE * cpu.re.abs().max(1.0));
        assert!((gpu.1 - cpu.im).abs() <= F32_PARITY_TOLERANCE * cpu.im.abs().max(1.0));
    }
}

#[test]
#[ignore = "requires a WGPU-compatible hardware adapter"]
fn gpu_fused_solve_matches_cpu() {
    let expression = solve(
        matrix([
            [event_scalar("x"), 1.0.into()],
            [complex(0.5, 0.25), event_scalar("x") + 2.0],
        ]),
        vector([
            Expr::from(parameter!("a", initial: 1.25)),
            complex(parameter!("b", initial: -0.4), 0.5),
        ]),
    )
    .component(1);
    let model = CompiledModel::from_expr(&expression).unwrap();
    let params = model.params().default_values();
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], false).unwrap());
    let batch = EventBatch::from_events(
        schema,
        [0.0, 0.5, 1.25]
            .into_iter()
            .map(|x| OwnedEvent::new(vec![], vec![x])),
    )
    .unwrap();
    let context = WgpuBackend::default()
        .open(&WgpuOptions::default(), WgpuPrecision::F32)
        .unwrap();
    let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
    let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
    let execution = Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions::default()),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap();
    let cpu = CpuBackend
        .prepare_for_execution(&model, &execution)
        .unwrap()
        .evaluate_batch(&params, &batch)
        .unwrap();
    for (gpu, cpu) in gpu.iter().zip(cpu) {
        assert!((gpu.0 - cpu.re).abs() <= 2.0e-5 * cpu.re.abs().max(1.0));
        assert!((gpu.1 - cpu.im).abs() <= 2.0e-5 * cpu.im.abs().max(1.0));
    }

    let singular_model = CompiledModel::from_expr(
        &solve(
            matrix([
                [event_scalar("x"), 2.0.into()],
                [event_scalar("x") * 2.0, 4.0.into()],
            ]),
            vector([1.0, 2.0]),
        )
        .component(0),
    )
    .unwrap();
    let singular = WgpuScalarKernel::compile(&context, &singular_model).unwrap();
    assert!(matches!(
        singular.evaluate_batch(&context, &singular_model.params().default_values(), &batch),
        Err(laddu_wgpu::WgpuError::SingularMatrixEvent(0))
    ));
}
