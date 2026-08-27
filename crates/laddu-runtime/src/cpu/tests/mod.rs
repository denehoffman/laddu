pub(super) use std::sync::Arc;

pub(super) use laddu_compile::{CachePolicy, CompileOptions, CompiledModel, ReductionPlan};
pub(super) use laddu_data::{
    RealVec4,
    data::{Dataset, EventBatch, OwnedEvent},
    schema::Schema,
};
pub(super) use laddu_expr::{
    P4Component, atan2, complex, dot, event_p4_component, event_scalar, matmul, matrix, matvec,
    parameter, polar_complex, solve, vector,
};

pub(super) use super::cache::CachedSlot;
pub(super) use super::prepared::resident_cache_plan;

#[cfg(feature = "jit")]
pub(super) use crate::jit::JitPrecision;

use super::*;

fn evaluate(expr: &laddu_expr::Expr) -> Complex64 {
    let model = CompiledModel::from_expr(expr).unwrap();
    let params = Arc::new(model.params().clone()).default_values();
    CpuBackend.prepare(&model).evaluate(&params).unwrap()
}

fn finite_difference(plan: &CpuPlan, params: &ParamValues, parameter: usize) -> Complex64 {
    let h = 1.0e-6;
    let mut plus = params.clone();
    let mut minus = params.clone();
    let id = params.layout().free_params()[parameter];
    let free_id = params.layout().free_id(id).unwrap().unwrap();
    let value = params.get(id).unwrap();
    plus.set_free(free_id, value + h).unwrap();
    minus.set_free(free_id, value - h).unwrap();
    (plan.evaluate(&plus).unwrap() - plan.evaluate(&minus).unwrap()) / (2.0 * h)
}

fn assert_gradient_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() < tolerance,
            "{actual} != {expected}"
        );
    }
}

#[cfg(feature = "jit")]
fn f32_execution(jit: JitPolicy) -> Execution {
    Execution::local(crate::ExecutionOptions {
        device: crate::Device::Cpu(crate::CpuOptions {
            jit,
            ..crate::CpuOptions::default()
        }),
        precision: Precision::F32,
        ..crate::ExecutionOptions::default()
    })
    .unwrap()
}

#[cfg(feature = "jit")]
fn f32_jit_and_interpreter(model: &CompiledModel) -> (CpuPlan, CpuPlan) {
    let automatic = CpuBackend
        .prepare_for_execution(model, &f32_execution(JitPolicy::Auto))
        .unwrap();
    let interpreted = CpuBackend
        .prepare_for_execution(model, &f32_execution(JitPolicy::Disabled))
        .unwrap();

    let Some(ScalarExecutor::Jit(kernel)) = &automatic.scalar_executor else {
        panic!("f32 auto execution should select scalar JIT");
    };
    assert_eq!(kernel.precision(), JitPrecision::F32);
    let GradientExecutor::Jit(kernel) = &automatic.gradient_executor else {
        panic!("f32 auto execution should select gradient JIT");
    };
    assert_eq!(kernel.precision(), JitPrecision::F32);
    if interpreted.scalar_executor.is_some() {
        assert!(matches!(
            interpreted.scalar_executor,
            Some(ScalarExecutor::Interpreter(_))
        ));
    }
    (automatic, interpreted)
}

#[cfg(feature = "jit")]
fn f64_jit_and_interpreter(model: &CompiledModel) -> (CpuPlan, CpuPlan) {
    let automatic = CpuBackend.prepare(model);
    let interpreted = CpuBackend.prepare_with_execution_mode(model, CpuExecutionMode::Interpreter);

    if automatic.scalar_executor.is_some() {
        assert!(matches!(
            automatic.scalar_executor,
            Some(ScalarExecutor::Jit(_))
        ));
    }
    if interpreted.scalar_executor.is_some() {
        assert!(matches!(
            interpreted.scalar_executor,
            Some(ScalarExecutor::Interpreter(_))
        ));
    }
    (automatic, interpreted)
}

#[cfg(feature = "jit")]
fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-6,
        "{actual} != {expected}"
    );
}

#[cfg(feature = "jit")]
fn assert_complex_slices_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_complex_close(*actual, *expected);
    }
}

#[cfg(feature = "jit")]
fn assert_complex_close_f64(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-10,
        "{actual} != {expected}"
    );
}

#[cfg(feature = "jit")]
fn assert_complex_slices_close_f64(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_complex_close_f64(*actual, *expected);
    }
}

#[test]
fn prepared_models_share_cpu_plan_across_executions() {
    let expression = parameter!("scale", initial: 1.0) * event_scalar("x");
    let model = CompiledModel::from_expr(&expression).unwrap();
    let cloned_model = model.clone();

    // Use distinct Execution objects deliberately. The CPU-plan cache is
    // process-wide so the ordinary Python pattern of constructing separate
    // likelihoods for separate bins still reuses the same plan.
    let first = crate::PreparedModel::prepare(&model, &Execution::default()).unwrap();
    let second = crate::PreparedModel::prepare(&cloned_model, &Execution::default()).unwrap();

    let (crate::PreparedModel::Cpu(first), crate::PreparedModel::Cpu(second)) = (first, second)
    else {
        panic!("default execution should prepare CPU models");
    };

    assert!(
        Arc::ptr_eq(&first, &second),
        "structurally identical models should share one CpuPlan"
    );
}

#[test]
fn cpu_plan_cache_distinguishes_event_cache_layouts() {
    let expression = parameter!("scale", initial: 1.0) * event_scalar("x");

    let cached_model = CompiledModel::from_expr(&expression).unwrap();
    let uncached_model = CompiledModel::from_expr_with_options(
        &expression,
        &CompileOptions::default().with_cache_policy(CachePolicy::Off),
    )
    .unwrap();

    // CachePolicy does not alter the optimized expression graph, so the model
    // digest is identical. The CpuPlan cache must independently include the
    // resolved event-cache layout in its key.
    assert_eq!(
        cached_model.optimized_digest(),
        uncached_model.optimized_digest()
    );

    let execution = Execution::default();
    let cached = crate::PreparedModel::prepare(&cached_model, &execution).unwrap();
    let uncached = crate::PreparedModel::prepare(&uncached_model, &execution).unwrap();

    let (crate::PreparedModel::Cpu(cached), crate::PreparedModel::Cpu(uncached)) =
        (cached, uncached)
    else {
        panic!("default execution should prepare CPU models");
    };

    assert!(
        !Arc::ptr_eq(&cached, &uncached),
        "different event-cache layouts require different CpuPlans"
    );
}

mod autodiff;
mod cache;
#[cfg(feature = "jit")]
mod jit;
mod reduction;
mod scalar;
