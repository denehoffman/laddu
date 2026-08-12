//! Generated backend-policy parity cases at the workspace boundary.

use std::sync::Arc;

use laddu::{
    autodiff::AutodiffMode,
    compile::CompiledModel,
    data::{
        data::{Dataset, OwnedEvent},
        schema::Schema,
    },
    event_scalar,
    expr::Expr,
    likelihood::{ExtendedNllTerm, Likelihood, LikelihoodEvaluation, NllTerm},
    parameter,
    runtime::{
        CpuOptions, Device, Execution, ExecutionOptions, JitPolicy, MemoryBudget, MemoryPlan,
        NormalizationMode, Precision, ThreadPolicy,
    },
};

#[cfg(feature = "wgpu")]
use laddu::runtime::{GpuBackend, GpuOptions};

#[derive(Clone, Copy, Debug)]
enum StorageMode {
    Resident,
    Streaming,
}

#[derive(Clone, Copy, Debug)]
enum ObjectiveKind {
    Shape,
    Extended,
}

fn dataset(storage: StorageMode) -> Dataset {
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let events = [
        (0.15, 0.7),
        (0.35, 1.2),
        (0.65, 0.5),
        (0.95, 1.8),
        (1.25, 0.9),
        (1.55, 1.1),
        (1.85, 0.6),
    ]
    .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![x], weight));
    let dataset = Dataset::from_events(schema, events)
        .unwrap()
        .with_memory_budget(MemoryBudget::Bytes(16 * 1024));
    match storage {
        StorageMode::Resident => dataset.resident(),
        StorageMode::Streaming => dataset.streaming(),
    }
}

fn model() -> CompiledModel {
    let x = event_scalar("x");
    let scale = Expr::from(parameter!("scale", initial: 0.35));
    CompiledModel::from_expr(&((scale * x.sin() + 1.5).powi(2) + 0.25)).unwrap()
}

fn cpu_execution(
    precision: Precision,
    autodiff: AutodiffMode,
    normalization: NormalizationMode,
    jit: JitPolicy,
) -> Execution {
    Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions {
            threads: ThreadPolicy::Serial,
            jit,
        }),
        precision,
        autodiff,
        normalization,
        memory: MemoryPlan::host(MemoryBudget::Bytes(4 * 1024 * 1024)),
        ..ExecutionOptions::default()
    })
    .unwrap()
}

fn evaluate(
    objective: ObjectiveKind,
    model: &CompiledModel,
    data: &Dataset,
    accepted: &Dataset,
    execution: &Execution,
) -> LikelihoodEvaluation {
    match objective {
        ObjectiveKind::Shape => {
            let term = NllTerm::new("shape", model, data, accepted).unwrap();
            let likelihood = Likelihood::with_execution([term], execution).unwrap();
            likelihood
                .nll_with_gradient(&likelihood.default_params())
                .unwrap()
        }
        ObjectiveKind::Extended => {
            let term = ExtendedNllTerm::new("extended", model, data, accepted).unwrap();
            let likelihood = Likelihood::with_execution([term], execution).unwrap();
            likelihood
                .nll_with_gradient(&likelihood.default_params())
                .unwrap()
        }
    }
}

fn assert_close(label: &str, actual: &LikelihoodEvaluation, expected: &LikelihoodEvaluation) {
    let tolerance = 2.0e-4;
    let value_scale = expected.value().abs().max(1.0);
    assert!(
        (actual.value() - expected.value()).abs() <= tolerance * value_scale,
        "{label}: value {} versus {}",
        actual.value(),
        expected.value()
    );
    assert_eq!(actual.gradient().len(), expected.gradient().len());
    for (index, (actual, expected)) in actual
        .gradient()
        .iter()
        .zip(expected.gradient())
        .enumerate()
    {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "{label}: gradient[{index}] {actual} versus {expected}"
        );
    }
}

fn assert_pool_released(execution: &Execution) {
    for report in execution.memory_pool_reports() {
        assert_eq!(
            report.reserved_bytes, 0,
            "{} retained a memory lease after evaluation",
            report.resource_id
        );
        assert_eq!(report.remaining_bytes, report.effective_bytes);
    }
}

#[test]
fn generated_cpu_backend_policy_matrix_matches_reference() {
    let model = model();
    let reference_execution = cpu_execution(
        Precision::F64,
        AutodiffMode::Forward,
        NormalizationMode::General,
        JitPolicy::Disabled,
    );

    for objective in [ObjectiveKind::Shape, ObjectiveKind::Extended] {
        let reference_data = dataset(StorageMode::Resident);
        let reference = evaluate(
            objective,
            &model,
            &reference_data,
            &reference_data,
            &reference_execution,
        );

        for storage in [StorageMode::Resident, StorageMode::Streaming] {
            for precision in [Precision::F64, Precision::F32] {
                for autodiff in [AutodiffMode::Forward, AutodiffMode::Reverse] {
                    for normalization in [
                        NormalizationMode::Auto,
                        NormalizationMode::General,
                        NormalizationMode::Verify,
                    ] {
                        let execution =
                            cpu_execution(precision, autodiff, normalization, JitPolicy::Disabled);
                        let data = dataset(storage);
                        let actual = evaluate(objective, &model, &data, &data, &execution);
                        assert_close(
                            &format!(
                                "cpu/{objective:?}/{storage:?}/{precision:?}/{autodiff:?}/{normalization:?}"
                            ),
                            &actual,
                            &reference,
                        );
                        assert_pool_released(&execution);
                    }
                }
            }
        }

        #[cfg(feature = "jit")]
        for storage in [StorageMode::Resident, StorageMode::Streaming] {
            for precision in [Precision::F64, Precision::F32] {
                for autodiff in [AutodiffMode::Forward, AutodiffMode::Reverse] {
                    let execution = cpu_execution(
                        precision,
                        autodiff,
                        NormalizationMode::Verify,
                        JitPolicy::Enabled,
                    );
                    let data = dataset(storage);
                    let actual = evaluate(objective, &model, &data, &data, &execution);
                    assert_close(
                        &format!("jit/{objective:?}/{storage:?}/{precision:?}/{autodiff:?}"),
                        &actual,
                        &reference,
                    );
                    assert_pool_released(&execution);
                }
            }
        }
    }

    assert_pool_released(&reference_execution);
}

#[cfg(feature = "wgpu")]
#[test]
fn generated_wgpu_policy_matrix_matches_cpu_when_an_adapter_is_available() {
    let model = model();
    let reference_execution = cpu_execution(
        Precision::F32,
        AutodiffMode::Forward,
        NormalizationMode::General,
        JitPolicy::Disabled,
    );

    for objective in [ObjectiveKind::Shape, ObjectiveKind::Extended] {
        let reference_data = dataset(StorageMode::Resident);
        let reference = evaluate(
            objective,
            &model,
            &reference_data,
            &reference_data,
            &reference_execution,
        );

        for storage in [StorageMode::Resident, StorageMode::Streaming] {
            for autodiff in [AutodiffMode::Forward, AutodiffMode::Reverse] {
                for normalization in [
                    NormalizationMode::Auto,
                    NormalizationMode::General,
                    NormalizationMode::Verify,
                ] {
                    let Ok(execution) = Execution::local(ExecutionOptions {
                        device: Device::Gpu(GpuOptions {
                            backend: GpuBackend::Wgpu,
                            ..GpuOptions::default()
                        }),
                        precision: Precision::F32,
                        autodiff,
                        normalization,
                        memory: MemoryPlan::host_device(
                            MemoryBudget::Bytes(4 * 1024 * 1024),
                            MemoryBudget::Bytes(4 * 1024 * 1024),
                        ),
                        ..ExecutionOptions::default()
                    }) else {
                        return;
                    };
                    let data = dataset(storage);
                    let actual = evaluate(objective, &model, &data, &data, &execution);
                    assert_close(
                        &format!("wgpu/{objective:?}/{storage:?}/{autodiff:?}/{normalization:?}"),
                        &actual,
                        &reference,
                    );
                    assert_pool_released(&execution);
                }
            }
        }
    }
}
