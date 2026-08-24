//! Reduction execution and resident reduction resources for the scalar WGPU backend.

use laddu_compile::{ReductionPlan, ReductionTransform};
use laddu_data::data::{EventBatch, accurate::AccurateF64};
use laddu_expr::parameters::ParamValues;
use laddu_memory::FootprintOverflow;
use wgpu::util::DeviceExt;

use super::batch::{BoundBatch, ChunkPlan, bind_batch, rebase_error};
use super::memory::{
    GpuMemoryLayout, REDUCTION_STATUS_WORDS, STATUS_SENTINEL, STATUS_WORD_BYTES, status_bytes,
};
use crate::readback::{decode_status, submit_and_readback};
use crate::scalar::WgpuScalarKernel;
use crate::{WgpuContext, WgpuError, WgpuResult};

/// Event-dependent buffers prepared for repeated WebGPU reductions.
#[derive(Clone, Debug)]
pub struct WgpuPreparedBatch {
    pub(super) chunks: Vec<WgpuPreparedChunk>,
    pub(super) len: usize,
    pub(super) resident_bytes: usize,
}

#[derive(Clone, Debug)]
pub(super) struct WgpuPreparedChunk {
    pub(super) input: wgpu::Buffer,
    pub(super) cache: wgpu::Buffer,
    pub(super) weights: wgpu::Buffer,
    pub(super) scratch: ReductionScratch,
    pub(super) events: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ReductionScratch {
    pub(super) params: wgpu::Buffer,
    pub(super) config: wgpu::Buffer,
    pub(super) partials: wgpu::Buffer,
    pub(super) error: wgpu::Buffer,
    pub(super) solve_error: wgpu::Buffer,
    pub(super) staging: wgpu::Buffer,
}

/// Borrowed inputs for one direct reduction dispatch.
struct ReductionRequest<'a> {
    params: &'a ParamValues,
    cache: &'a wgpu::Buffer,
    weights: &'a wgpu::Buffer,
    solve_error: &'a wgpu::Buffer,
    events: usize,
    reduction: ReductionPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub(crate) enum ReductionMode {
    Real = 0,
    PositiveReal = 1,
    LogPositiveReal = 2,
}

impl ReductionMode {
    pub(crate) const fn code(self) -> u32 {
        self as u32
    }
    pub(crate) fn from_plan(plan: ReductionPlan) -> Self {
        match plan.transform() {
            ReductionTransform::Real => Self::Real,
            ReductionTransform::PositiveReal => Self::PositiveReal,
            ReductionTransform::LogPositiveReal => Self::LogPositiveReal,
        }
    }
}

impl WgpuScalarKernel {
    /// Applies a weighted reduction directly to an event batch.
    ///
    /// # Errors
    ///
    /// Returns an error when memory geometry, event bindings, GPU execution,
    /// or reduction status decoding fails.
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
        let plan = ChunkPlan::for_batch(batch.len(), chunk_len);
        let bound = bind_batch(batch, &self.event_inputs)?;
        let mut total = AccurateF64::zero();
        for range in plan.ranges {
            match self.reduce_bound_chunk(context, params, &bound, range.clone(), reduction) {
                Ok(value) => total.push(value),
                Err(error) => return Err(rebase_error(error, range.start)),
            }
        }
        Ok(total.finish())
    }

    /// Materializes event-dependent GPU buffers for repeated reductions.
    ///
    /// # Errors
    ///
    /// Returns an error when memory geometry, event bindings, cache
    /// materialization, or GPU readback fails.
    pub fn prepare_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> WgpuResult<WgpuPreparedBatch> {
        let chunk_len = self.max_chunk_events(context, params, true)?;
        let layout = self
            .memory_layout(params)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let mut chunks = Vec::new();
        let mut resident_bytes = 0usize;
        let plan = ChunkPlan::for_batch(batch.len(), chunk_len);
        let bound = bind_batch(batch, &self.event_inputs)?;
        for range in &plan.ranges {
            let packed = self.pack_bound_chunk(&bound, range.clone());
            let (input, cache) = self.cache_buffers(context, &packed.inputs, packed.events)?;
            let weights = context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu resident event weights"),
                    contents: &packed.weights,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            let scratch = self.reduction_scratch(context, params, packed.events)?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            self.encode_cache_materialization(
                context,
                &mut encoder,
                &input,
                &cache,
                &scratch.solve_error,
                packed.events,
            );
            context.queue().submit([encoder.finish()]);
            if let Some(index) = Self::read_error(context, &scratch.solve_error)? {
                return Err(WgpuError::SingularMatrixEvent(packed.start + index));
            }
            let chunk_bytes = layout
                .prepared_resident_bytes(packed.events)
                .and_then(|bytes| usize::try_from(bytes).map_err(|_| FootprintOverflow::Conversion))
                .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                })?;
            resident_bytes =
                resident_bytes
                    .checked_add(chunk_bytes)
                    .ok_or(WgpuError::MemoryBudgetTooSmall {
                        required: usize::MAX,
                        available: context.memory_budget().unwrap_or(usize::MAX),
                    })?;
            chunks.push(WgpuPreparedChunk {
                input,
                cache,
                weights,
                scratch,
                events: packed.events,
            });
        }
        Ok(WgpuPreparedBatch {
            chunks,
            len: batch.len(),
            resident_bytes,
        })
    }

    /// Reuses prepared allocations while replacing event values and weights.
    ///
    /// # Errors
    ///
    /// Returns an error when memory geometry, event bindings, cache
    /// materialization, or GPU readback fails.
    pub fn refresh_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
        prepared: &mut WgpuPreparedBatch,
    ) -> WgpuResult<bool> {
        let chunk_len = self.max_chunk_events(context, params, true)?;
        let plan = ChunkPlan::for_batch(batch.len(), chunk_len);
        if !plan.matches_event_counts(prepared.chunks.iter().map(|chunk| chunk.events)) {
            return Ok(false);
        }
        let bound = bind_batch(batch, &self.event_inputs)?;
        let packed_chunks = plan
            .ranges
            .iter()
            .map(|range| self.pack_bound_chunk(&bound, range.clone()))
            .collect::<Vec<_>>();
        for (packed, prepared_chunk) in packed_chunks.iter().zip(&prepared.chunks) {
            context
                .queue()
                .write_buffer(&prepared_chunk.input, 0, &packed.inputs);
            context
                .queue()
                .write_buffer(&prepared_chunk.weights, 0, &packed.weights);
            context.queue().write_buffer(
                &prepared_chunk.scratch.solve_error,
                0,
                bytemuck::bytes_of(&STATUS_SENTINEL),
            );
            let mut encoder = context.device().create_command_encoder(&Default::default());
            self.encode_cache_materialization(
                context,
                &mut encoder,
                &prepared_chunk.input,
                &prepared_chunk.cache,
                &prepared_chunk.scratch.solve_error,
                packed.events,
            );
            context.queue().submit([encoder.finish()]);
            if let Some(index) = Self::read_error(context, &prepared_chunk.scratch.solve_error)? {
                return Err(WgpuError::SingularMatrixEvent(packed.start + index));
            }
        }
        prepared.len = batch.len();
        Ok(true)
    }

    /// Applies a weighted reduction to a prepared batch.
    ///
    /// # Errors
    ///
    /// Returns an error when GPU execution, readback, or reduction status
    /// decoding fails.
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
                Err(error) => return Err(rebase_error(error, offset)),
            }
            offset += chunk.events;
        }
        Ok(total.finish())
    }

    /// Applies a weighted reduction and computes its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns an error when the gradient pipeline is unavailable, GPU
    /// execution fails, or status decoding reports an event failure.
    pub fn reduce_prepared_batch_with_gradient(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &WgpuPreparedBatch,
        reduction: ReductionPlan,
    ) -> WgpuResult<(f64, Vec<f64>)> {
        if self.partial_width == 1 {
            return self
                .reduce_prepared_batch(context, params, batch, reduction)
                .map(|value| (value, Vec::new()));
        }
        let gradient_len = self.partial_width - 1;
        let mut total = AccurateF64::zero();
        let mut gradient = (0..gradient_len)
            .map(|_| AccurateF64::zero())
            .collect::<Vec<_>>();
        let mut offset = 0;
        for chunk in &batch.chunks {
            match self.reduce_prepared_chunk_with_gradient(context, params, chunk, reduction) {
                Ok((value, values)) => {
                    total.push(value);
                    for (sum, value) in gradient.iter_mut().zip(values) {
                        sum.push(value);
                    }
                }
                Err(error) => return Err(rebase_error(error, offset)),
            }
            offset += chunk.events;
        }
        Ok((
            total.finish(),
            gradient.into_iter().map(AccurateF64::finish).collect(),
        ))
    }

    fn write_reduction_inputs(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        chunk: &WgpuPreparedChunk,
        reduction: ReductionPlan,
    ) {
        let parameters = self.parameter_values(params);
        let mode = Self::reduction_mode(reduction);
        context
            .queue()
            .write_buffer(&chunk.scratch.params, 0, &parameters);
        context
            .queue()
            .write_buffer(&chunk.scratch.config, 0, bytemuck::bytes_of(&mode));
        context.queue().write_buffer(
            &chunk.scratch.error,
            0,
            bytemuck::bytes_of(&STATUS_SENTINEL),
        );
        context.queue().write_buffer(
            &chunk.scratch.solve_error,
            0,
            bytemuck::bytes_of(&STATUS_SENTINEL),
        );
    }

    fn reduce_prepared_chunk_with_gradient(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        chunk: &WgpuPreparedChunk,
        reduction: ReductionPlan,
    ) -> WgpuResult<(f64, Vec<f64>)> {
        self.write_reduction_inputs(context, params, chunk, reduction);
        self.execute_prepared_gradient_reduce(
            context,
            &chunk.cache,
            &chunk.weights,
            chunk.events,
            &chunk.scratch,
        )
    }

    fn reduce_prepared_chunk(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        chunk: &WgpuPreparedChunk,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        self.write_reduction_inputs(context, params, chunk, reduction);
        self.execute_prepared_reduce(
            context,
            &chunk.cache,
            &chunk.weights,
            chunk.events,
            &chunk.scratch,
        )
    }

    fn reduction_mode(reduction: ReductionPlan) -> u32 {
        ReductionMode::from_plan(reduction).code()
    }

    fn reduction_scratch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        events: usize,
    ) -> WgpuResult<ReductionScratch> {
        let layout = self
            .memory_layout(params)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let parameters = self.parameter_values(params);
        let params = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared parameters"),
                contents: &parameters,
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
            size: layout
                .partial_buffer_bytes(events, STATUS_WORD_BYTES)
                .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                })?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared reduction error"),
                contents: bytemuck::bytes_of(&STATUS_SENTINEL),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let solve_error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu prepared solve error"),
                contents: bytemuck::bytes_of(&STATUS_SENTINEL),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu prepared reduction readback"),
            size: layout.staging_buffer_bytes(events).map_err(|_| {
                WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                }
            })?,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(ReductionScratch {
            params,
            config,
            partials,
            error,
            solve_error,
            staging,
        })
    }

    fn execute_prepared_reduce(
        &self,
        context: &WgpuContext,
        cache: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        events: usize,
        scratch: &ReductionScratch,
    ) -> WgpuResult<f64> {
        let groups = GpuMemoryLayout::groups(events);
        let layout = self
            .buffer_layout()
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let partial_bytes = usize::try_from(layout.scalar_partial_bytes(events).map_err(|_| {
            WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            }
        })?)
        .map_err(|_| WgpuError::MemoryBudgetTooSmall {
            required: usize::MAX,
            available: context.memory_budget().unwrap_or(usize::MAX),
        })?;
        let bind_group = super::dispatch::ReductionBindings {
            params: &scratch.params,
            cache,
            weights,
            config: &scratch.config,
            partials: &scratch.partials,
            error: &scratch.error,
            solve_error: &scratch.solve_error,
        }
        .bind_group(
            context,
            &self.execution.reduction_bind_group_layout,
            "laddu prepared reduction bind group",
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.execution.reduction_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.partials,
            0,
            &scratch.staging,
            0,
            partial_bytes as u64,
        );
        encoder.copy_buffer_to_buffer(
            &scratch.error,
            0,
            &scratch.staging,
            partial_bytes as u64,
            STATUS_WORD_BYTES as u64,
        );
        encoder.copy_buffer_to_buffer(
            &scratch.solve_error,
            0,
            &scratch.staging,
            (partial_bytes + STATUS_WORD_BYTES) as u64,
            STATUS_WORD_BYTES as u64,
        );
        submit_and_readback(
            context,
            encoder,
            &scratch.staging,
            (partial_bytes + status_bytes(REDUCTION_STATUS_WORDS)) as u64,
            |mapped| {
                let status = decode_status(&mapped[partial_bytes..], REDUCTION_STATUS_WORDS)?;
                if let Some(error) = status.error() {
                    return Err(error);
                }
                let mut total = AccurateF64::zero();
                for value in self.decode_scalars(&mapped[..partial_bytes]) {
                    total.push(value);
                }
                Ok(total.finish())
            },
        )
    }

    fn execute_prepared_gradient_reduce(
        &self,
        context: &WgpuContext,
        cache: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        events: usize,
        scratch: &ReductionScratch,
    ) -> WgpuResult<(f64, Vec<f64>)> {
        let pipeline = self
            .execution
            .gradient_reduction_pipeline
            .as_ref()
            .ok_or_else(|| {
                WgpuError::UnsupportedInstruction("model has no free parameters".into())
            })?;
        let groups = GpuMemoryLayout::groups(events);
        let layout = self
            .buffer_layout()
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let partial_bytes =
            usize::try_from(layout.partial_buffer_bytes(events, 0).map_err(|_| {
                WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                }
            })?)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let bind_group = super::dispatch::ReductionBindings {
            params: &scratch.params,
            cache,
            weights,
            config: &scratch.config,
            partials: &scratch.partials,
            error: &scratch.error,
            solve_error: &scratch.solve_error,
        }
        .bind_group(
            context,
            &self.execution.reduction_bind_group_layout,
            "laddu prepared gradient reduction bind group",
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.partials,
            0,
            &scratch.staging,
            0,
            partial_bytes as u64,
        );
        encoder.copy_buffer_to_buffer(
            &scratch.error,
            0,
            &scratch.staging,
            partial_bytes as u64,
            STATUS_WORD_BYTES as u64,
        );
        encoder.copy_buffer_to_buffer(
            &scratch.solve_error,
            0,
            &scratch.staging,
            (partial_bytes + STATUS_WORD_BYTES) as u64,
            STATUS_WORD_BYTES as u64,
        );
        submit_and_readback(
            context,
            encoder,
            &scratch.staging,
            (partial_bytes + status_bytes(REDUCTION_STATUS_WORDS)) as u64,
            |mapped| {
                let status = decode_status(&mapped[partial_bytes..], REDUCTION_STATUS_WORDS)?;
                if let Some(error) = status.error() {
                    return Err(error);
                }
                let partials = self.decode_scalars(&mapped[..partial_bytes]);
                let mut sums = (0..self.partial_width)
                    .map(|_| AccurateF64::zero())
                    .collect::<Vec<_>>();
                for group in 0..groups {
                    for component in 0..self.partial_width {
                        sums[component].push(partials[group * self.partial_width + component]);
                    }
                }
                let mut values = sums.into_iter().map(AccurateF64::finish);
                Ok((values.next().unwrap_or(0.0), values.collect()))
            },
        )
    }

    fn reduce_bound_chunk(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &BoundBatch<'_>,
        range: std::ops::Range<usize>,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        let events = range.len();
        let inputs = self.encode_scalars(&batch.pack_range(range.clone()));
        let weights = batch.pack_weights(range);
        let weight_bytes = self.encode_scalars(&weights);
        let (input_buffer, cache_buffer) = self.cache_buffers(context, &inputs, events)?;
        let solve_error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu batch solve error"),
                contents: bytemuck::bytes_of(&STATUS_SENTINEL),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let weights_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu event weights"),
                    contents: &weight_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        self.encode_cache_materialization(
            context,
            &mut encoder,
            &input_buffer,
            &cache_buffer,
            &solve_error,
            events,
        );
        context.queue().submit([encoder.finish()]);
        self.reduce_buffers(
            context,
            ReductionRequest {
                params,
                cache: &cache_buffer,
                weights: &weights_buffer,
                solve_error: &solve_error,
                events,
                reduction,
            },
        )
    }

    fn reduce_buffers(
        &self,
        context: &WgpuContext,
        request: ReductionRequest<'_>,
    ) -> WgpuResult<f64> {
        let layout =
            self.memory_layout(request.params)
                .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                })?;
        let parameters = self.parameter_values(request.params);
        let mode = Self::reduction_mode(request.reduction);
        let groups = GpuMemoryLayout::groups(request.events);
        let partial_bytes =
            usize::try_from(layout.scalar_partial_bytes(request.events).map_err(|_| {
                WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                }
            })?)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: &parameters,
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
            size: partial_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu reduction error"),
                contents: bytemuck::bytes_of(&STATUS_SENTINEL),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging_size = (partial_bytes + status_bytes(REDUCTION_STATUS_WORDS)) as u64;
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu reduction readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = super::dispatch::ReductionBindings {
            params: &params_buffer,
            cache: request.cache,
            weights: request.weights,
            config: &config_buffer,
            partials: &partials,
            error: &error,
            solve_error: request.solve_error,
        }
        .bind_group(
            context,
            &self.execution.reduction_bind_group_layout,
            "laddu reduction bind group",
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.execution.reduction_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partials, 0, &staging, 0, partial_bytes as u64);
        encoder.copy_buffer_to_buffer(
            &error,
            0,
            &staging,
            partial_bytes as u64,
            STATUS_WORD_BYTES as u64,
        );
        encoder.copy_buffer_to_buffer(
            request.solve_error,
            0,
            &staging,
            (partial_bytes + STATUS_WORD_BYTES) as u64,
            STATUS_WORD_BYTES as u64,
        );
        submit_and_readback(context, encoder, &staging, staging_size, |mapped| {
            let status = decode_status(&mapped[partial_bytes..], REDUCTION_STATUS_WORDS)?;
            if let Some(error) = status.error() {
                return Err(error);
            }
            let mut total = AccurateF64::zero();
            for value in self.decode_scalars(&mapped[..partial_bytes]) {
                total.push(value);
            }
            Ok(total.finish())
        })
    }
}

impl WgpuPreparedBatch {
    /// Returns the number of prepared events.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the prepared batch contains no events.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the estimated number of resident GPU bytes.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}
