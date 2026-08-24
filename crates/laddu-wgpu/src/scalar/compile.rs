//! Private compilation pipeline for the scalar WGPU backend.
//!
//! The public scalar kernel is intentionally a small facade.  Compilation is
//! kept here as a sequence of named stages so that plan validation, target
//! lowering, shader creation, and resource assembly cannot get mixed into
//! execution code.

use laddu_compile::{CacheLayout, CompiledModel, ExecutablePlan};
use laddu_kernel::ir::OutputComponent;

use super::bindings::Binding;
use crate::scalar::EventInput;
use crate::{WgpuContext, WgpuError, WgpuResult};

/// The resources needed to materialize event-dependent cache values.
#[derive(Debug)]
pub(crate) struct CachePipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

/// Pipelines and layouts that share the validated scalar shader contract.
#[derive(Debug)]
pub(crate) struct ExecutionPipelines {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) reduction_pipeline: wgpu::ComputePipeline,
    pub(crate) gradient_reduction_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) reduction_bind_group_layout: wgpu::BindGroupLayout,
}

/// The immutable metadata derived from a validated executable plan.
#[derive(Debug)]
pub(crate) struct KernelMetadata {
    pub(crate) event_inputs: Vec<EventInput>,
    pub(crate) cache_layout: CacheLayout,
    pub(crate) partial_width: usize,
}

/// All device resources produced by compilation.
#[derive(Debug)]
pub(crate) struct CompiledKernel {
    pub(crate) precision: crate::WgpuPrecision,
    pub(crate) cache: Option<CachePipeline>,
    pub(crate) execution: ExecutionPipelines,
    pub(crate) metadata: KernelMetadata,
}

/// Validate options and lower the public model into the backend executable.
fn validate_and_lower(model: &CompiledModel) -> WgpuResult<ExecutablePlan> {
    ExecutablePlan::from_model_for_fused_backend(model)
        .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))
}

fn derive_metadata(
    executable: &ExecutablePlan,
    free_parameters: usize,
) -> WgpuResult<KernelMetadata> {
    let event_inputs = executable
        .cache_inputs()
        .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))?;

    let cache_layout = executable.cache_plan().layout();
    if cache_layout.width() > 0 && executable.cache_kernel().is_none() {
        return Err(WgpuError::UnsupportedInstruction(
            "cache materialization contains unsupported operations".to_string(),
        ));
    }
    Ok(KernelMetadata {
        event_inputs,
        cache_layout,
        partial_width: free_parameters + 1,
    })
}

fn storage_binding(binding: Binding, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: binding.index(),
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn shader_module(context: &WgpuContext, label: &'static str, source: String) -> wgpu::ShaderModule {
    context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

fn pipeline(
    context: &WgpuContext,
    label: &'static str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
    context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
}

fn build_cache_pipeline(context: &WgpuContext, source: &str) -> WgpuResult<CachePipeline> {
    let module = shader_module(context, "laddu cache kernel", source.to_owned());
    let layout = context
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("laddu cache bindings"),
            entries: &[
                storage_binding(Binding::Parameters, true),
                storage_binding(Binding::Cache, false),
                storage_binding(Binding::SolveError, false),
            ],
        });
    Ok(CachePipeline {
        pipeline: pipeline(context, "laddu cache pipeline", &module, &layout, "main"),
        bind_group_layout: layout,
    })
}

fn build_execution_pipelines(
    context: &WgpuContext,
    scalar_source: &str,
    gradient_source: Option<&str>,
) -> WgpuResult<ExecutionPipelines> {
    let module = shader_module(context, "laddu scalar kernel", scalar_source.to_owned());
    let scalar_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("laddu scalar bindings"),
                entries: &[
                    storage_binding(Binding::Parameters, true),
                    storage_binding(Binding::Cache, true),
                    storage_binding(Binding::Output, false),
                    storage_binding(Binding::SolveError, false),
                ],
            });

    let reduction_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("laddu reduction bindings"),
                entries: &[
                    storage_binding(Binding::Parameters, true),
                    storage_binding(Binding::Cache, true),
                    storage_binding(Binding::Weights, true),
                    storage_binding(Binding::Config, true),
                    storage_binding(Binding::Partials, false),
                    storage_binding(Binding::ReductionError, false),
                    storage_binding(Binding::SolveError, false),
                ],
            });
    let gradient_pipeline = gradient_source
        .map(|source| {
            let module = shader_module(context, "laddu scalar gradient kernel", source.to_owned());
            Ok(pipeline(
                context,
                "laddu gradient reduction pipeline",
                &module,
                &reduction_layout,
                "reduce_gradient",
            ))
        })
        .transpose()?;
    Ok(ExecutionPipelines {
        pipeline: pipeline(
            context,
            "laddu scalar pipeline",
            &module,
            &scalar_layout,
            "main",
        ),
        bind_group_layout: scalar_layout,
        reduction_pipeline: pipeline(
            context,
            "laddu reduction pipeline",
            &module,
            &reduction_layout,
            "reduce",
        ),
        gradient_reduction_pipeline: gradient_pipeline,
        reduction_bind_group_layout: reduction_layout,
    })
}

/// Compile a model through the validation, metadata, shader, and resource
/// stages while returning one immutable bundle for the scalar facade.
pub(crate) fn compile(context: &WgpuContext, model: &CompiledModel) -> WgpuResult<CompiledKernel> {
    crate::scalar::WgpuScalarKernel::validate_precision(context.precision())?;
    let executable = validate_and_lower(model)?;
    let ir = executable
        .scalar_kernel()
        .ok_or(WgpuError::MissingScalarKernel)?;
    let free_parameters = model.params().free_params();
    let metadata = derive_metadata(&executable, free_parameters.len())?;
    let scalar_source =
        crate::scalar::WgpuScalarKernel::wgsl(ir, &metadata.cache_layout, context.precision())?;
    let cache_source = executable
        .cache_kernel()
        .map(|ir| {
            crate::scalar::WgpuScalarKernel::cache_wgsl(
                ir,
                metadata.event_inputs.len(),
                &metadata.cache_layout,
                context.precision(),
            )
        })
        .transpose()?;
    let gradient = if free_parameters.is_empty() {
        None
    } else {
        Some(
            laddu_autodiff::gradient_ir(ir, free_parameters, OutputComponent::Real)
                .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))?,
        )
    };
    let gradient_source = gradient
        .as_ref()
        .map(|gradient| {
            crate::scalar::WgpuScalarKernel::gradient_wgsl(
                gradient,
                &metadata.cache_layout,
                context.precision(),
            )
        })
        .transpose()?;
    let cache = cache_source
        .as_deref()
        .map(|source| build_cache_pipeline(context, source))
        .transpose()?;
    let execution = build_execution_pipelines(context, &scalar_source, gradient_source.as_deref())?;
    Ok(CompiledKernel {
        precision: context.precision(),
        cache,
        execution,
        metadata,
    })
}
