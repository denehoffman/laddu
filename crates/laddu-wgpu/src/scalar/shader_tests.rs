//! Hardware-independent tests for generated scalar/cache WGSL.

use super::WgpuScalarKernel;
use crate::WgpuPrecision;
use laddu_compile::{CacheLayout, CompiledModel, ExecutablePlan};
use laddu_expr::{
    Expr, dot, event_scalar, matrix, matrix_from_flat, matvec, parameter, solve, vector,
};

fn validate_f64(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::FLOAT64,
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn generated_cache_and_scalar_shaders_preserve_flattened_matrix_slots() {
    let cached_matrix = matrix([
        [event_scalar("x"), event_scalar("y")],
        [event_scalar("y") + 1.0, event_scalar("x") - 1.0],
    ]);
    let expression = dot(
        matvec(cached_matrix, vector([parameter!("a"), parameter!("b")])),
        vector([1.0, 2.0]),
    ) + dot(
        vector([event_scalar("x") + 2.0, event_scalar("y") - 2.0]),
        vector([parameter!("c"), parameter!("d")]),
    );
    let model = CompiledModel::from_expr_with_options(
        &expression,
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let plan = ExecutablePlan::from_model(&model).unwrap();
    let layout = plan.cache_plan().layout();
    let width = layout.width();
    let source =
        WgpuScalarKernel::wgsl(plan.scalar_kernel().unwrap(), &layout, WgpuPrecision::F32).unwrap();
    let cache = WgpuScalarKernel::cache_wgsl(
        plan.cache_kernel().unwrap(),
        plan.cache_input_nodes().len(),
        &layout,
        WgpuPrecision::F32,
    )
    .unwrap();

    assert_eq!(width, 6);
    assert!(source.contains("cache[row * 6u + 5u]"));
    assert!(source.contains("cmul("));
    assert!(cache.contains("arrayLength(&cache)/6u"));
    assert!(cache.contains("cache[row * 6u + 5u]"));
    naga::front::wgsl::parse_str(&source).unwrap();
    naga::front::wgsl::parse_str(&cache).unwrap();
}

#[test]
fn generated_scalar_shader_accepts_supported_solves_and_rejects_oversized_solves() {
    let identity = matrix([[1.0, 0.0], [0.0, 1.0]]);
    let model = CompiledModel::from_expr_with_options(
        &solve(identity, vector([1.0, 2.0])).component(0),
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let plan = ExecutablePlan::from_model_for_fused_backend(&model).unwrap();
    let source = WgpuScalarKernel::wgsl(
        plan.scalar_kernel().unwrap(),
        &CacheLayout::default(),
        WgpuPrecision::F32,
    )
    .unwrap();
    assert!(source.contains("var lu"));
    assert!(source.contains("var piv"));
    assert!(source.contains("cdiv("));
    naga::front::wgsl::parse_str(&source).unwrap();

    let dimension = 17;
    let oversized_matrix = matrix_from_flat(
        dimension,
        dimension,
        (0..dimension * dimension).map(|index| {
            Expr::from(if index / dimension == index % dimension {
                1.0
            } else {
                0.0
            })
        }),
    )
    .unwrap();
    let oversized = CompiledModel::from_expr_with_options(
        &solve(
            oversized_matrix,
            vector((0..dimension).map(|_| Expr::from(1.0))),
        )
        .component(0),
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let oversized = ExecutablePlan::from_model_for_fused_backend(&oversized).unwrap();
    assert!(matches!(
        WgpuScalarKernel::wgsl(
            oversized.scalar_kernel().unwrap(),
            &CacheLayout::default(),
            WgpuPrecision::F32
        ),
        Err(crate::WgpuError::SolveDimensionTooLarge { dimension: 17 })
    ));
}

#[test]
fn generated_f64_cache_scalar_and_gradient_shaders_validate() {
    let expression = event_scalar("x").sin() + event_scalar("y").cos();
    let model = CompiledModel::from_expr_with_options(
        &expression,
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let plan = ExecutablePlan::from_model(&model).unwrap();
    let layout = plan.cache_plan().layout();
    let scalar =
        WgpuScalarKernel::wgsl(plan.scalar_kernel().unwrap(), &layout, WgpuPrecision::F64).unwrap();
    let cache = WgpuScalarKernel::cache_wgsl(
        plan.cache_kernel().unwrap(),
        plan.cache_input_nodes().len(),
        &layout,
        WgpuPrecision::F64,
    )
    .unwrap();
    validate_f64(&scalar);
    validate_f64(&cache);
    assert!(scalar.contains("array<vec2<f64>"));
    assert!(!scalar.contains(" sin("));

    let gradient = laddu_autodiff::gradient_ir(
        plan.scalar_kernel().unwrap(),
        model.params().free_params(),
        laddu_kernel::ir::OutputComponent::Real,
    )
    .unwrap();
    let gradient = WgpuScalarKernel::gradient_wgsl(&gradient, &layout, WgpuPrecision::F64).unwrap();
    validate_f64(&gradient);
}

#[test]
fn generated_f32_gradient_shader_validates() {
    let expression = parameter!("a") * event_scalar("x").sin();
    let model = CompiledModel::from_expr_with_options(
        &expression,
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let plan = ExecutablePlan::from_model(&model).unwrap();
    let layout = plan.cache_plan().layout();
    let gradient = laddu_autodiff::gradient_ir(
        plan.scalar_kernel().unwrap(),
        model.params().free_params(),
        laddu_kernel::ir::OutputComponent::Real,
    )
    .unwrap();
    let source = WgpuScalarKernel::gradient_wgsl(&gradient, &layout, WgpuPrecision::F32).unwrap();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn generated_f64_solve_shader_validates() {
    let model = CompiledModel::from_expr_with_options(
        &solve(matrix([[1.0, 0.0], [0.0, 1.0]]), vector([1.0, 2.0])).component(0),
        &laddu_compile::CompileOptions::without_optimizations(),
    )
    .unwrap();
    let plan = ExecutablePlan::from_model_for_fused_backend(&model).unwrap();
    let source = WgpuScalarKernel::wgsl(
        plan.scalar_kernel().unwrap(),
        &CacheLayout::default(),
        WgpuPrecision::F64,
    )
    .unwrap();
    validate_f64(&source);
}
