use std::sync::mpsc;

use laddu_compile::{CompiledModel, ExecutablePlan, ReductionPlan, ReductionTransform};
use laddu_data::data::{EventBatch, accurate::AccurateF64};
use laddu_expr::{BinaryOp, ExprNode, P4Component, UnaryOp, parameters::ParamValues};
use laddu_kernel::ir::{
    CacheKernelIr, KernelInstruction, KernelValue, KernelValueId, KernelValueKind, ScalarKernelIr,
};
use wgpu::util::DeviceExt;

use crate::{WgpuContext, WgpuError, WgpuResult};

#[derive(Debug)]
pub struct WgpuScalarKernel {
    cache_pipeline: Option<wgpu::ComputePipeline>,
    cache_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    reduction_pipeline: wgpu::ComputePipeline,
    reduction_bind_group_layout: wgpu::BindGroupLayout,
    cache_width: usize,
    event_inputs: Vec<EventInput>,
}

#[derive(Clone, Debug)]
pub struct WgpuPreparedBatch {
    chunks: Vec<WgpuPreparedChunk>,
    len: usize,
    resident_bytes: usize,
}

#[derive(Clone, Debug)]
struct WgpuPreparedChunk {
    input: wgpu::Buffer,
    cache: wgpu::Buffer,
    weights: wgpu::Buffer,
    scratch: ReductionScratch,
    events: usize,
}

#[derive(Clone, Debug)]
struct ReductionScratch {
    params: wgpu::Buffer,
    config: wgpu::Buffer,
    partials: wgpu::Buffer,
    error: wgpu::Buffer,
    staging: wgpu::Buffer,
}

#[derive(Clone, Debug)]
enum EventInput {
    Scalar(String),
    P4(String, P4Component),
}

impl WgpuScalarKernel {
    pub fn compile(context: &WgpuContext, model: &CompiledModel) -> WgpuResult<Self> {
        Self::validate_precision(context.precision())?;
        let executable = ExecutablePlan::from_model(model)
            .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))?;
        let ir = executable
            .scalar_kernel()
            .ok_or(WgpuError::MissingScalarKernel)?;
        let event_inputs = executable
            .cache_input_nodes()
            .iter()
            .map(|node| match executable.graph().node(*node) {
                Some(ExprNode::EventScalar(name)) => Ok(EventInput::Scalar(name.to_string())),
                Some(ExprNode::EventP4Component { name, component }) => {
                    Ok(EventInput::P4(name.to_string(), *component))
                }
                node => Err(WgpuError::UnsupportedInstruction(format!(
                    "computed cache entry {node:?}"
                ))),
            })
            .collect::<WgpuResult<Vec<_>>>()?;
        let mut cache_offsets = Vec::with_capacity(executable.cache_plan().len());
        let mut cache_width = 0;
        for entry in executable.cache_plan().entries() {
            cache_offsets.push(cache_width);
            cache_width += entry.storage_kind().width();
        }
        if cache_width > 0 && executable.cache_kernel().is_none() {
            return Err(WgpuError::UnsupportedInstruction(
                "cache materialization contains unsupported operations".to_string(),
            ));
        }
        let (cache_pipeline, cache_bind_group_layout) = executable
            .cache_kernel()
            .map(|ir| Self::compile_cache_pipeline(context, ir, event_inputs.len()))
            .transpose()?
            .map_or((None, None), |(pipeline, layout)| {
                (Some(pipeline), Some(layout))
            });
        let source = Self::wgsl(ir, &cache_offsets, cache_width)?;
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("laddu scalar kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("laddu scalar bindings"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("laddu scalar pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("laddu scalar pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let reduction_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("laddu reduction bindings"),
                    entries: &[
                        Self::storage_binding(0, true),
                        Self::storage_binding(1, true),
                        Self::storage_binding(3, true),
                        Self::storage_binding(4, true),
                        Self::storage_binding(5, false),
                        Self::storage_binding(6, false),
                    ],
                });
        let reduction_layout =
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("laddu reduction pipeline layout"),
                    bind_group_layouts: &[Some(&reduction_bind_group_layout)],
                    immediate_size: 0,
                });
        let reduction_pipeline =
            context
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("laddu reduction pipeline"),
                    layout: Some(&reduction_layout),
                    module: &module,
                    entry_point: Some("reduce"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        Ok(Self {
            cache_pipeline,
            cache_bind_group_layout,
            pipeline,
            bind_group_layout,
            reduction_pipeline,
            reduction_bind_group_layout,
            cache_width,
            event_inputs,
        })
    }

    fn validate_precision(precision: crate::WgpuPrecision) -> WgpuResult<()> {
        match precision {
            crate::WgpuPrecision::F32 => Ok(()),
            unsupported => Err(WgpuError::UnsupportedKernelPrecision(unsupported)),
        }
    }

    fn compile_cache_pipeline(
        context: &WgpuContext,
        ir: &CacheKernelIr,
        input_slots: usize,
    ) -> WgpuResult<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let source = Self::cache_wgsl(ir, input_slots)?;
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("laddu cache kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("laddu cache bindings"),
                    entries: &[
                        Self::storage_binding(0, true),
                        Self::storage_binding(1, false),
                    ],
                });
        let layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("laddu cache pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("laddu cache pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok((pipeline, bind_group_layout))
    }

    fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn evaluate(&self, context: &WgpuContext, params: &ParamValues) -> WgpuResult<(f64, f64)> {
        self.evaluate_packed(context, params, &[0.0, 0.0], 1)
            .map(|mut values| values.remove(0))
    }

    pub fn evaluate_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        let chunk_len = self.max_chunk_events(context, params, false)?;
        let mut values = Vec::with_capacity(batch.len());
        for start in (0..batch.len()).step_by(chunk_len) {
            let end = (start + chunk_len).min(batch.len());
            let chunk = batch.slice(start, end);
            let inputs = self.pack_batch(&chunk)?;
            values.extend(self.evaluate_packed(context, params, &inputs, chunk.len())?);
        }
        Ok(values)
    }

    pub fn reduce_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let chunk_len = self.max_chunk_events(context, params, true)?;
        let mut total = AccurateF64::zero();
        for start in (0..batch.len()).step_by(chunk_len) {
            let end = (start + chunk_len).min(batch.len());
            let chunk = batch.slice(start, end);
            match self.reduce_chunk(context, params, &chunk, reduction) {
                Ok(value) => total.push(value),
                Err(WgpuError::NonPositiveEvent(index)) => {
                    return Err(WgpuError::NonPositiveEvent(start + index));
                }
                Err(error) => return Err(error),
            }
        }
        Ok(total.finish())
    }

    pub fn prepare_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> WgpuResult<WgpuPreparedBatch> {
        let chunk_len = self.max_chunk_events(context, params, true)?;
        let mut chunks = Vec::new();
        let mut resident_bytes = 0;
        for start in (0..batch.len()).step_by(chunk_len) {
            let end = (start + chunk_len).min(batch.len());
            let chunk = batch.slice(start, end);
            let inputs = self.pack_batch(&chunk)?;
            let weights = (0..chunk.len())
                .map(|row| chunk.weights_at(row) as f32)
                .collect::<Vec<_>>();
            let (input, cache) = self.cache_buffers(context, &inputs, chunk.len());
            let weights_buffer =
                context
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("laddu resident event weights"),
                        contents: bytemuck::cast_slice(&weights),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
            let scratch = Self::reduction_scratch(context, params, chunk.len());
            let mut encoder = context.device().create_command_encoder(&Default::default());
            self.encode_cache_materialization(context, &mut encoder, &input, &cache, chunk.len());
            context.queue().submit([encoder.finish()]);
            resident_bytes += inputs.len() * size_of::<f32>()
                + weights.len() * size_of::<f32>()
                + (chunk.len() * self.cache_width * size_of::<[f32; 2]>()).max(8)
                + params.as_slice().len().max(1) * size_of::<f32>()
                + chunk.len().div_ceil(64) * size_of::<f32>() * 2
                + 8;
            chunks.push(WgpuPreparedChunk {
                input,
                cache,
                weights: weights_buffer,
                scratch,
                events: chunk.len(),
            });
        }
        Ok(WgpuPreparedBatch {
            chunks,
            len: batch.len(),
            resident_bytes,
        })
    }

    pub fn refresh_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
        prepared: &mut WgpuPreparedBatch,
    ) -> WgpuResult<bool> {
        let chunk_len = self.max_chunk_events(context, params, true)?;
        let expected = batch.len().div_ceil(chunk_len);
        if prepared.chunks.len() != expected {
            return Ok(false);
        }
        for (chunk_index, start) in (0..batch.len()).step_by(chunk_len).enumerate() {
            let end = (start + chunk_len).min(batch.len());
            let batch_chunk = batch.slice(start, end);
            let prepared_chunk = &prepared.chunks[chunk_index];
            if prepared_chunk.events != batch_chunk.len() {
                return Ok(false);
            }
            let inputs = self.pack_batch(&batch_chunk)?;
            let weights = (0..batch_chunk.len())
                .map(|row| batch_chunk.weights_at(row) as f32)
                .collect::<Vec<_>>();
            context
                .queue()
                .write_buffer(&prepared_chunk.input, 0, bytemuck::cast_slice(&inputs));
            context.queue().write_buffer(
                &prepared_chunk.weights,
                0,
                bytemuck::cast_slice(&weights),
            );
            let mut encoder = context.device().create_command_encoder(&Default::default());
            self.encode_cache_materialization(
                context,
                &mut encoder,
                &prepared_chunk.input,
                &prepared_chunk.cache,
                prepared_chunk.events,
            );
            context.queue().submit([encoder.finish()]);
        }
        prepared.len = batch.len();
        Ok(true)
    }

    pub fn reduce_prepared_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &WgpuPreparedBatch,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        let mut total = AccurateF64::zero();
        let mut offset = 0;
        for chunk in &batch.chunks {
            match self.reduce_prepared_chunk(context, params, chunk, reduction) {
                Ok(value) => total.push(value),
                Err(WgpuError::NonPositiveEvent(index)) => {
                    return Err(WgpuError::NonPositiveEvent(offset + index));
                }
                Err(error) => return Err(error),
            }
            offset += chunk.events;
        }
        Ok(total.finish())
    }

    fn reduce_prepared_chunk(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        chunk: &WgpuPreparedChunk,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        let parameters = Self::parameter_values(params);
        let mode = Self::reduction_mode(reduction);
        context
            .queue()
            .write_buffer(&chunk.scratch.params, 0, bytemuck::cast_slice(&parameters));
        context
            .queue()
            .write_buffer(&chunk.scratch.config, 0, bytemuck::bytes_of(&mode));
        context
            .queue()
            .write_buffer(&chunk.scratch.error, 0, bytemuck::bytes_of(&u32::MAX));
        self.execute_prepared_reduce(
            context,
            &chunk.cache,
            &chunk.weights,
            chunk.events,
            &chunk.scratch,
        )
    }

    fn parameter_values(params: &ParamValues) -> Vec<f32> {
        let values = params
            .as_slice()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        if values.is_empty() { vec![0.0] } else { values }
    }

    fn reduction_mode(reduction: ReductionPlan) -> u32 {
        match reduction.transform() {
            ReductionTransform::Real => 0,
            ReductionTransform::PositiveReal => 1,
            ReductionTransform::LogPositiveReal => 2,
        }
    }

    fn reduction_scratch(
        context: &WgpuContext,
        params: &ParamValues,
        events: usize,
    ) -> ReductionScratch {
        let parameters = Self::parameter_values(params);
        let groups = events.div_ceil(64);
        let params = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared parameters"),
                contents: bytemuck::cast_slice(&parameters),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let config = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared reduction config"),
                contents: bytemuck::bytes_of(&0_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let partials = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu prepared reduction partials"),
            size: (groups * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared reduction error"),
                contents: bytemuck::bytes_of(&u32::MAX),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu prepared reduction readback"),
            size: (groups * 4 + 4).max(8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        ReductionScratch {
            params,
            config,
            partials,
            error,
            staging,
        }
    }

    fn execute_prepared_reduce(
        &self,
        context: &WgpuContext,
        cache: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        events: usize,
        scratch: &ReductionScratch,
    ) -> WgpuResult<f64> {
        let groups = events.div_ceil(64);
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu prepared reduction bind group"),
                layout: &self.reduction_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scratch.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scratch.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: scratch.partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: scratch.error.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.reduction_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.partials,
            0,
            &scratch.staging,
            0,
            (groups * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&scratch.error, 0, &scratch.staging, (groups * 4) as u64, 4);
        context.queue().submit([encoder.finish()]);
        let slice = scratch.staging.slice(..(groups * 4 + 4) as u64);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| WgpuError::DevicePoll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let invalid = words[groups];
        let mut total = AccurateF64::zero();
        for bits in &words[..groups] {
            total.push(f32::from_bits(*bits) as f64);
        }
        drop(mapped);
        scratch.staging.unmap();
        if invalid != u32::MAX {
            return Err(WgpuError::NonPositiveEvent(invalid as usize));
        }
        Ok(total.finish())
    }

    fn reduce_chunk(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        let inputs = self.pack_batch(batch)?;
        let weights = (0..batch.len())
            .map(|row| batch.weights_at(row) as f32)
            .collect::<Vec<_>>();
        let (input_buffer, cache_buffer) = self.cache_buffers(context, &inputs, batch.len());
        let weights_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu event weights"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        self.encode_cache_materialization(
            context,
            &mut encoder,
            &input_buffer,
            &cache_buffer,
            batch.len(),
        );
        context.queue().submit([encoder.finish()]);
        self.reduce_buffers(
            context,
            params,
            &cache_buffer,
            &weights_buffer,
            batch.len(),
            reduction,
        )
    }

    fn reduce_buffers(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        cache_buffer: &wgpu::Buffer,
        weights_buffer: &wgpu::Buffer,
        events: usize,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        let parameters = params
            .as_slice()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let parameters = if parameters.is_empty() {
            vec![0.0]
        } else {
            parameters
        };
        let mode = match reduction.transform() {
            ReductionTransform::Real => 0_u32,
            ReductionTransform::PositiveReal => 1,
            ReductionTransform::LogPositiveReal => 2,
        };
        let groups = events.div_ceil(64);
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: bytemuck::cast_slice(&parameters),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let config_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu reduction config"),
                    contents: bytemuck::bytes_of(&mode),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let partials = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu reduction partials"),
            size: (groups * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu reduction error"),
                contents: bytemuck::bytes_of(&u32::MAX),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging_size = (groups * 4 + 4) as u64;
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu reduction readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu reduction bind group"),
                layout: &self.reduction_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: error.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.reduction_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partials, 0, &staging, 0, (groups * 4) as u64);
        encoder.copy_buffer_to_buffer(&error, 0, &staging, (groups * 4) as u64, 4);
        context.queue().submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| WgpuError::DevicePoll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let invalid = words[groups];
        if invalid != u32::MAX {
            return Err(WgpuError::NonPositiveEvent(invalid as usize));
        }
        let mut total = AccurateF64::zero();
        for bits in &words[..groups] {
            total.push(f32::from_bits(*bits) as f64);
        }
        Ok(total.finish())
    }

    fn max_chunk_events(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        reduction: bool,
    ) -> WgpuResult<usize> {
        let input_bytes = self.event_inputs.len() * size_of::<[f32; 2]>();
        let cache_bytes = self.cache_width * size_of::<[f32; 2]>();
        let result_bytes = if reduction {
            size_of::<f32>() + 1
        } else {
            2 * size_of::<[f32; 2]>()
        };
        let per_event = input_bytes + cache_bytes + result_bytes;
        let parameter_bytes = params.as_slice().len().max(1) * size_of::<f32>();
        let fixed_bytes = parameter_bytes + if reduction { 16 } else { 0 };
        let max_binding = context
            .info()
            .max_buffer_size
            .min(context.info().max_storage_buffer_binding_size as u64)
            .min(usize::MAX as u64) as usize;
        let mut max_events = u32::MAX as usize;
        for width in [input_bytes, cache_bytes, if reduction { 4 } else { 8 }] {
            if width > 0 {
                max_events = max_events.min(max_binding / width);
            }
        }
        if let Some(budget) = context.memory_budget() {
            let available = budget.saturating_sub(fixed_bytes);
            max_events = max_events.min(available / per_event.max(1));
            if max_events == 0 {
                return Err(WgpuError::MemoryBudgetTooSmall {
                    required: fixed_bytes + per_event.max(1),
                    available: budget,
                });
            }
        }
        if max_events == 0 {
            return Err(WgpuError::MemoryBudgetTooSmall {
                required: fixed_bytes + per_event.max(1),
                available: max_binding,
            });
        }
        Ok(max_events)
    }

    fn pack_batch(&self, batch: &EventBatch) -> WgpuResult<Vec<f32>> {
        let mut inputs = Vec::with_capacity(batch.len() * self.event_inputs.len() * 2);
        for row in 0..batch.len() {
            for input in &self.event_inputs {
                let value = match input {
                    EventInput::Scalar(name) => {
                        let col = batch
                            .schema()
                            .scalar_index(name)
                            .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                        batch.scalar_at(col, row)
                    }
                    EventInput::P4(name, component) => {
                        let col = batch
                            .schema()
                            .p4_index(name)
                            .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                        let p4 = batch.p4_at(col, row);
                        match component {
                            P4Component::Px => p4.x,
                            P4Component::Py => p4.y,
                            P4Component::Pz => p4.z,
                            P4Component::E => p4.t,
                        }
                    }
                };
                inputs.extend([value as f32, 0.0]);
            }
        }
        if inputs.is_empty() {
            inputs.extend([0.0, 0.0]);
        }
        Ok(inputs)
    }

    fn evaluate_packed(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        inputs: &[f32],
        events: usize,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        if events == 0 {
            return Ok(Vec::new());
        }
        let values = params
            .as_slice()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let values = if values.is_empty() { vec![0.0] } else { values };
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let (input_buffer, cache_buffer) = self.cache_buffers(context, inputs, events);
        let output_size = (events * 8) as u64;
        let output = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu scalar bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        self.encode_cache_materialization(
            context,
            &mut encoder,
            &input_buffer,
            &cache_buffer,
            events,
        );
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((events as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        context.queue().submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| WgpuError::DevicePoll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&mapped);
        let values = result
            .chunks_exact(2)
            .map(|value| (value[0] as f64, value[1] as f64))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(values)
    }

    fn cache_buffers(
        &self,
        context: &WgpuContext,
        inputs: &[f32],
        events: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let input = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu event inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let cache_size = (events * self.cache_width * 8).max(8) as u64;
        let cache = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu event cache"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        (input, cache)
    }

    fn encode_cache_materialization(
        &self,
        context: &WgpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        cache: &wgpu::Buffer,
        events: usize,
    ) {
        let (Some(pipeline), Some(layout)) = (&self.cache_pipeline, &self.cache_bind_group_layout)
        else {
            return;
        };
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu cache bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((events as u32).div_ceil(64), 1, 1);
    }

    fn scalar_prelude() -> &'static str {
        "fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }\n\
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let d=b.x*b.x+b.y*b.y; return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }\n\
fn csqrt(z: vec2<f32>) -> vec2<f32> { let m=length(z); let re=sqrt(max(0.0, 0.5*(m+z.x))); let im=sqrt(max(0.0, 0.5*(m-z.x))); return vec2(re, select(-im, im, z.y >= 0.0)); }\n\
fn cexp(z: vec2<f32>) -> vec2<f32> { let e=exp(z.x); return vec2(e*cos(z.y), e*sin(z.y)); }\n\
fn csin(z: vec2<f32>) -> vec2<f32> { return vec2(sin(z.x)*cosh(z.y), cos(z.x)*sinh(z.y)); }\n\
fn ccos(z: vec2<f32>) -> vec2<f32> { return vec2(cos(z.x)*cosh(z.y), -sin(z.x)*sinh(z.y)); }\n\
fn clog(z: vec2<f32>) -> vec2<f32> { return vec2(log(length(z)), atan2(z.y, z.x)); }\n\
fn cpowi(z: vec2<f32>, exponent: i32) -> vec2<f32> { var result=vec2(1.0, 0.0); var base=z; var n=abs(exponent); loop { if (n == 0) { break; } if ((n & 1) == 1) { result=cmul(result, base); } base=cmul(base, base); n=n/2; } if (exponent < 0) { return cdiv(vec2(1.0, 0.0), result); } return result; }\n"
    }

    fn aggregate(elements: impl IntoIterator<Item = String>, width: usize) -> String {
        format!(
            "array<vec2<f32>, {width}>({})",
            elements.into_iter().collect::<Vec<_>>().join(", ")
        )
    }

    fn emit_values(
        values: &[KernelValue],
        cached: impl Fn(usize, KernelValueKind) -> String,
    ) -> WgpuResult<String> {
        let mut source = String::new();
        let v = |id: KernelValueId| format!("v{}", id.index());
        for (index, value) in values.iter().enumerate() {
            let expr = match &value.instruction {
                KernelInstruction::Cached(slot) => cached(*slot, value.kind),
                KernelInstruction::RealConstant(x) => format!("vec2<f32>({:?}, 0.0)", *x as f32),
                KernelInstruction::ComplexConstant(x) => {
                    format!("vec2<f32>({:?}, {:?})", x.re as f32, x.im as f32)
                }
                KernelInstruction::Parameter(id) => format!("vec2<f32>(p[{}], 0.0)", id.index()),
                KernelInstruction::Unary {
                    op: UnaryOp::Neg,
                    input,
                } => format!("-{}", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Real,
                    input,
                } => format!("vec2<f32>({}.x, 0.0)", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Imag,
                    input,
                } => format!("vec2<f32>({}.y, 0.0)", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Conj,
                    input,
                } => format!("vec2<f32>({}.x, -{}.y)", v(*input), v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::NormSqr,
                    input,
                } => format!("vec2<f32>(dot({0}, {0}), 0.0)", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Sqrt,
                    input,
                } => format!("csqrt({})", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Exp,
                    input,
                } => format!("cexp({})", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Sin,
                    input,
                } => format!("csin({})", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Cos,
                    input,
                } => format!("ccos({})", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Log,
                    input,
                } => format!("clog({})", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::PowI(power),
                    input,
                } => format!("cpowi({}, {power})", v(*input)),
                KernelInstruction::Binary { op, lhs, rhs } => match op {
                    BinaryOp::Add => format!("{} + {}", v(*lhs), v(*rhs)),
                    BinaryOp::Sub => format!("{} - {}", v(*lhs), v(*rhs)),
                    BinaryOp::Mul => format!("cmul({}, {})", v(*lhs), v(*rhs)),
                    BinaryOp::Div => format!("cdiv({}, {})", v(*lhs), v(*rhs)),
                    BinaryOp::Atan2 => {
                        format!("vec2<f32>(atan2({}.x, {}.x), 0.0)", v(*lhs), v(*rhs))
                    }
                },
                KernelInstruction::Add(ids) => {
                    ids.iter().map(|id| v(*id)).collect::<Vec<_>>().join(" + ")
                }
                KernelInstruction::Mul(ids) => ids
                    .iter()
                    .map(|id| v(*id))
                    .reduce(|a, b| format!("cmul({a}, {b})"))
                    .unwrap(),
                KernelInstruction::Complex { re, im } => {
                    format!("vec2<f32>({}.x, {}.x)", v(*re), v(*im))
                }
                KernelInstruction::Vector(elements) => {
                    Self::aggregate(elements.iter().map(|element| v(*element)), elements.len())
                }
                KernelInstruction::Matrix { elements, .. } => {
                    Self::aggregate(elements.iter().map(|element| v(*element)), elements.len())
                }
                KernelInstruction::Component { input, index } => {
                    format!("{}[{index}]", v(*input))
                }
                KernelInstruction::MatrixElement { input, row, col } => {
                    let KernelValueKind::Matrix { cols, .. } = values[input.index()].kind else {
                        unreachable!("matrix-element IR was validated")
                    };
                    format!("{}[{}]", v(*input), row * cols + col)
                }
                KernelInstruction::Dot { lhs, rhs } => {
                    let KernelValueKind::Vector { len } = values[lhs.index()].kind else {
                        unreachable!("dot-product IR was validated")
                    };
                    (0..len)
                        .map(|element| {
                            format!("cmul({}[{element}], {}[{element}])", v(*lhs), v(*rhs))
                        })
                        .reduce(|a, b| format!("{a} + {b}"))
                        .unwrap()
                }
                KernelInstruction::MatVec { matrix, vector } => {
                    let KernelValueKind::Matrix { rows, cols } = values[matrix.index()].kind else {
                        unreachable!("matrix-vector IR was validated")
                    };
                    Self::aggregate(
                        (0..rows).map(|row| {
                            (0..cols)
                                .map(|col| {
                                    format!(
                                        "cmul({}[{}], {}[{col}])",
                                        v(*matrix),
                                        row * cols + col,
                                        v(*vector)
                                    )
                                })
                                .reduce(|a, b| format!("{a} + {b}"))
                                .unwrap()
                        }),
                        rows,
                    )
                }
                KernelInstruction::MatMul { lhs, rhs } => {
                    let KernelValueKind::Matrix { rows, cols: inner } = values[lhs.index()].kind
                    else {
                        unreachable!("matrix-matrix IR was validated")
                    };
                    let KernelValueKind::Matrix { cols, .. } = values[rhs.index()].kind else {
                        unreachable!("matrix-matrix IR was validated")
                    };
                    Self::aggregate(
                        (0..rows).flat_map(|row| {
                            (0..cols).map(move |col| {
                                (0..inner)
                                    .map(|element| {
                                        format!(
                                            "cmul({}[{}], {}[{}])",
                                            v(*lhs),
                                            row * inner + element,
                                            v(*rhs),
                                            element * cols + col
                                        )
                                    })
                                    .reduce(|a, b| format!("{a} + {b}"))
                                    .unwrap()
                            })
                        }),
                        rows * cols,
                    )
                }
                instruction => {
                    return Err(WgpuError::UnsupportedInstruction(format!(
                        "{instruction:?}"
                    )));
                }
            };
            source.push_str(&format!("let v{index} = {expr};\n"));
        }
        Ok(source)
    }

    fn cache_wgsl(ir: &CacheKernelIr, input_slots: usize) -> WgpuResult<String> {
        let mut output_offsets = Vec::with_capacity(ir.outputs().len());
        let mut output_width = 0;
        for output in ir.outputs() {
            output_offsets.push(output_width);
            output_width += ir.values()[output.index()].kind.width();
        }
        let mut source = format!(
            "@group(0) @binding(0) var<storage, read> inputs: array<vec2<f32>>;\n@group(0) @binding(1) var<storage, read_write> cache: array<vec2<f32>>;\n{}@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nlet row=gid.x;\nif (row >= arrayLength(&cache)/{output_width}u) {{ return; }}\n",
            Self::scalar_prelude()
        );
        source.push_str(&Self::emit_values(ir.values(), |slot, _| {
            format!("inputs[row * {input_slots}u + {slot}u]")
        })?);
        for (slot, output) in ir.outputs().iter().enumerate() {
            let width = ir.values()[output.index()].kind.width();
            for element in 0..width {
                let value = if width == 1 {
                    format!("v{}", output.index())
                } else {
                    format!("v{}[{element}]", output.index())
                };
                source.push_str(&format!(
                    "cache[row * {output_width}u + {}u] = {value};\n",
                    output_offsets[slot] + element
                ));
            }
        }
        source.push_str("}\n");
        Ok(source)
    }

    fn wgsl(
        ir: &ScalarKernelIr,
        cache_offsets: &[usize],
        cache_width: usize,
    ) -> WgpuResult<String> {
        let mut source = format!(
            "@group(0) @binding(0) var<storage, read> p: array<f32>;\n@group(0) @binding(1) var<storage, read> cache: array<vec2<f32>>;\n@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;\n@group(0) @binding(3) var<storage, read> weights: array<f32>;\n@group(0) @binding(4) var<storage, read> config: array<u32>;\n@group(0) @binding(5) var<storage, read_write> partials: array<f32>;\n@group(0) @binding(6) var<storage, read_write> reduction_error: array<atomic<u32>>;\nvar<workgroup> sums: array<f32, 64>;\n{}fn model(row: u32) -> vec2<f32> {{\n",
            Self::scalar_prelude()
        );
        source.push_str(&Self::emit_values(ir.values(), |slot, kind| {
            let offset = cache_offsets[slot];
            if kind.width() == 1 {
                format!("cache[row * {cache_width}u + {offset}u]")
            } else {
                Self::aggregate(
                    (0..kind.width()).map(|element| {
                        format!("cache[row * {cache_width}u + {}u]", offset + element)
                    }),
                    kind.width(),
                )
            }
        })?);
        let v = |id: KernelValueId| format!("v{}", id.index());
        source.push_str(&format!(
            "return {};\n}}\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nif (gid.x >= arrayLength(&out)) {{ return; }}\nout[gid.x] = model(gid.x);\n}}\n@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\nvar contribution = 0.0;\nif (gid.x < arrayLength(&weights)) {{\nlet value = model(gid.x).x;\nif (config[0] == 0u) {{ contribution = value; }} else if (value <= 0.0) {{ atomicMin(&reduction_error[0], gid.x); }} else if (config[0] == 1u) {{ contribution = value; }} else {{ contribution = log(value); }}\ncontribution *= weights[gid.x];\n}}\nsums[lid.x] = contribution;\nworkgroupBarrier();\nvar stride = 32u;\nloop {{\nif (lid.x < stride) {{ sums[lid.x] += sums[lid.x + stride]; }}\nworkgroupBarrier();\nif (stride == 1u) {{ break; }}\nstride /= 2u;\n}}\nif (lid.x == 0u) {{ partials[wid.x] = sums[0]; }}\n}}\n",
            v(ir.root())
        ));
        Ok(source)
    }
}

impl WgpuPreparedBatch {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use laddu_compile::{CompiledModel, ReductionPlan};
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{
        Expr, complex, dot, event_scalar, matmul, matrix, matvec, parameter, solve, vector,
    };
    use laddu_runtime::{CpuBackend, CpuOptions, Device, Execution, ExecutionOptions, Precision};

    use crate::{WgpuBackend, WgpuOptions, WgpuPrecision, WgpuScalarKernel};

    #[test]
    fn scalar_kernel_rejects_unimplemented_precisions() {
        assert!(WgpuScalarKernel::validate_precision(WgpuPrecision::F32).is_ok());
        assert!(matches!(
            WgpuScalarKernel::validate_precision(WgpuPrecision::F64),
            Err(crate::WgpuError::UnsupportedKernelPrecision(
                WgpuPrecision::F64
            ))
        ));
    }

    #[test]
    fn aggregate_wgsl_uses_flattened_cache_offsets_and_rejects_solves() {
        let cached_matrix = matrix([
            [event_scalar("x"), event_scalar("y")],
            [event_scalar("y") + 1.0, event_scalar("x") - 1.0],
        ]);
        let matrix_term = dot(
            matvec(cached_matrix, vector([parameter!("a"), parameter!("b")])),
            vector([1.0, 2.0]),
        );
        let vector_term = dot(
            vector([event_scalar("x") + 2.0, event_scalar("y") - 2.0]),
            vector([parameter!("c"), parameter!("d")]),
        );
        let expression = matrix_term + vector_term;
        let model = CompiledModel::from_expr_with_options(
            &expression,
            &laddu_compile::CompileOptions::without_optimizations(),
        )
        .unwrap();
        let executable = laddu_compile::ExecutablePlan::from_model(&model).unwrap();
        let mut offsets = Vec::new();
        let mut width = 0;
        for entry in executable.cache_plan().entries() {
            offsets.push(width);
            width += entry.storage_kind().width();
        }
        let source =
            WgpuScalarKernel::wgsl(executable.scalar_kernel().unwrap(), &offsets, width).unwrap();
        let cache_source = WgpuScalarKernel::cache_wgsl(
            executable.cache_kernel().unwrap(),
            executable.cache_input_nodes().len(),
        )
        .unwrap();

        assert_eq!(width, 6);
        assert!(source.contains("array<vec2<f32>, 4>"));
        assert!(source.contains("array<vec2<f32>, 2>"));
        assert!(source.contains("cache[row * 6u + 5u]"));
        assert!(source.contains("cmul("));
        assert!(cache_source.contains("arrayLength(&cache)/6u"));
        assert!(cache_source.contains("cache[row * 6u + 5u]"));

        let solve_model = CompiledModel::from_expr_with_options(
            &solve(matrix([[1.0, 0.0], [0.0, 1.0]]), vector([1.0, 2.0])).component(0),
            &laddu_compile::CompileOptions::without_optimizations(),
        )
        .unwrap();
        let solve_ir = laddu_compile::ExecutablePlan::from_model(&solve_model)
            .unwrap()
            .scalar_kernel()
            .unwrap()
            .clone();
        assert!(matches!(
            WgpuScalarKernel::wgsl(&solve_ir, &[], 0),
            Err(crate::WgpuError::UnsupportedInstruction(_))
        ));
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
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let cpu = CpuBackend
            .prepare_for_execution(&model, &execution)
            .unwrap()
            .evaluate(&params)
            .unwrap();

        assert_eq!(gpu, (cpu.re, cpu.im));
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
        assert!(kernel.max_chunk_events(&context, &params, false).unwrap() < batch.len());
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
            Err(crate::WgpuError::NonPositiveEvent(11))
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

        assert!(kernel.cache_pipeline.is_some());
        assert_eq!(kernel.event_inputs.len(), 2);

        let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F64,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let cpu = CpuBackend
            .prepare_for_execution(&model, &execution)
            .unwrap()
            .evaluate_batch(&params, &batch)
            .unwrap();

        for (gpu, cpu) in gpu.iter().zip(cpu) {
            assert!((gpu.0 - cpu.re).abs() <= 1.0e-5 * cpu.re.abs().max(1.0));
            assert!((gpu.1 - cpu.im).abs() <= 1.0e-5 * cpu.im.abs().max(1.0));
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
        assert!(kernel.max_chunk_events(&context, &params, false).unwrap() < batch.len());
        let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F64,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let cpu = CpuBackend
            .prepare_for_execution(&model, &execution)
            .unwrap()
            .evaluate_batch(&params, &batch)
            .unwrap();

        for (gpu, cpu) in gpu.iter().zip(cpu) {
            assert!((gpu.0 - cpu.re).abs() <= 1.0e-5 * cpu.re.abs().max(1.0));
            assert!((gpu.1 - cpu.im).abs() <= 1.0e-5 * cpu.im.abs().max(1.0));
        }
    }
}
