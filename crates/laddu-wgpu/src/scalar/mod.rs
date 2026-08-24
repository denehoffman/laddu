use laddu_compile::{CacheLayout, CompiledModel};
use laddu_data::data::EventBatch;
use laddu_expr::parameters::ParamValues;
use laddu_memory::{FootprintOverflow, MemoryFootprint};

use crate::readback::{decode_singular_status, submit_and_readback};
mod batch;
mod bindings;
mod compile;
mod dispatch;
pub(crate) mod memory;
mod reduction;
mod wgsl;

use crate::{WgpuContext, WgpuError, WgpuResult};
use batch::{ChunkPlan, bind_batch, rebase_error};
use compile::CachePipeline;
use memory::{GpuMemoryLayout, STATUS_WORD_BYTES};

/// A compiled scalar model kernel and its reduction pipelines.
#[derive(Debug)]
pub struct WgpuScalarKernel {
    pub(super) precision: crate::WgpuPrecision,
    pub(super) cache: Option<CachePipeline>,
    pub(super) execution: compile::ExecutionPipelines,
    pub(super) cache_layout: CacheLayout,
    pub(super) partial_width: usize,
    pub(super) event_inputs: Vec<EventInput>,
}

pub use reduction::WgpuPreparedBatch;

pub(crate) use laddu_compile::CacheInput as EventInput;

impl WgpuScalarKernel {
    /// Estimates tracked GPU bytes for a prepared reduction batch.
    pub fn prepared_memory_estimate(&self, params: &ParamValues, events: usize) -> usize {
        self.prepared_memory_footprint(params)
            .map(|footprint| usize::try_from(footprint.peak_bytes(events)).unwrap_or(usize::MAX))
            .unwrap_or(usize::MAX)
    }

    /// Returns the checked fixed/per-event footprint for prepared reduction
    /// buffers.
    #[doc(hidden)]
    pub fn prepared_memory_footprint(
        &self,
        params: &ParamValues,
    ) -> Result<MemoryFootprint, FootprintOverflow> {
        self.memory_layout(params)?.prepared_footprint()
    }

    pub(super) fn memory_layout(
        &self,
        params: &ParamValues,
    ) -> Result<GpuMemoryLayout, FootprintOverflow> {
        GpuMemoryLayout::new(
            self.scalar_size(),
            self.event_inputs.len(),
            self.cache_layout.width(),
            self.partial_width,
            params.as_slice().len(),
        )
    }

    pub(super) fn buffer_layout(&self) -> Result<GpuMemoryLayout, FootprintOverflow> {
        GpuMemoryLayout::new(
            self.scalar_size(),
            self.event_inputs.len(),
            self.cache_layout.width(),
            self.partial_width,
            0,
        )
    }

    /// Compiles a model's scalar kernel and reduction pipelines for `context`.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuError`] when precision is unsupported, model lowering
    /// fails, required kernels are missing, or WGSL pipeline creation fails.
    pub fn compile(context: &WgpuContext, model: &CompiledModel) -> WgpuResult<Self> {
        let compiled = compile::compile(context, model)?;
        let compile::CompiledKernel {
            precision,
            cache,
            execution,
            metadata,
        } = compiled;
        Ok(Self {
            precision,
            cache,
            execution,
            cache_layout: metadata.cache_layout,
            partial_width: metadata.partial_width,
            event_inputs: metadata.event_inputs,
        })
    }

    pub(crate) fn validate_precision(precision: crate::WgpuPrecision) -> WgpuResult<()> {
        match precision {
            crate::WgpuPrecision::F32 | crate::WgpuPrecision::F64 => Ok(()),
            unsupported => Err(WgpuError::UnsupportedKernelPrecision(unsupported)),
        }
    }

    pub(super) fn read_error(
        context: &WgpuContext,
        error: &wgpu::Buffer,
    ) -> WgpuResult<Option<usize>> {
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu error readback"),
            size: STATUS_WORD_BYTES as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(error, 0, &staging, 0, STATUS_WORD_BYTES as u64);
        submit_and_readback(
            context,
            encoder,
            &staging,
            STATUS_WORD_BYTES as u64,
            |mapped| Ok(decode_singular_status(mapped)?.singular_event),
        )
    }

    /// Evaluates a model with no event-dependent inputs.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuError`] when parameters are incompatible, GPU execution
    /// or buffer mapping fails, or a singular solve is encountered.
    pub fn evaluate(&self, context: &WgpuContext, params: &ParamValues) -> WgpuResult<(f64, f64)> {
        let inputs = self.encode_scalars(&[0.0, 0.0]);
        self.evaluate_packed(context, params, &inputs, 1)
            .map(|mut values| values.remove(0))
    }

    /// Evaluates the model for every event in a batch.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuError`] when parameters or event columns are incompatible,
    /// GPU execution fails, or an event contains a singular solve.
    pub fn evaluate_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        let chunk_len = self.max_chunk_events(context, params, false)?;
        let plan = ChunkPlan::for_batch(batch.len(), chunk_len);
        let bound = bind_batch(batch, &self.event_inputs)?;
        let mut values = Vec::with_capacity(batch.len());
        for range in plan.ranges {
            let inputs = self.encode_scalars(&bound.pack_range(range.clone()));
            match self.evaluate_packed(context, params, &inputs, range.len()) {
                Ok(chunk_values) => values.extend(chunk_values),
                Err(error) => return Err(rebase_error(error, range.start)),
            }
        }
        Ok(values)
    }

    pub(super) fn scalar_size(&self) -> usize {
        match self.precision {
            crate::WgpuPrecision::F32 => size_of::<f32>(),
            crate::WgpuPrecision::F64 => size_of::<f64>(),
            crate::WgpuPrecision::Auto => unreachable!("kernel precision is resolved"),
        }
    }

    pub(super) fn encode_scalars(&self, values: &[f64]) -> Vec<u8> {
        match self.precision {
            crate::WgpuPrecision::F32 => {
                bytemuck::cast_slice(&values.iter().map(|value| *value as f32).collect::<Vec<_>>())
                    .to_vec()
            }
            crate::WgpuPrecision::F64 => bytemuck::cast_slice(values).to_vec(),
            crate::WgpuPrecision::Auto => unreachable!("kernel precision is resolved"),
        }
    }

    pub(super) fn decode_scalars(&self, bytes: &[u8]) -> Vec<f64> {
        match self.precision {
            crate::WgpuPrecision::F32 => bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .map(|value| *value as f64)
                .collect(),
            crate::WgpuPrecision::F64 => bytemuck::cast_slice::<u8, f64>(bytes).to_vec(),
            crate::WgpuPrecision::Auto => unreachable!("kernel precision is resolved"),
        }
    }

    pub(super) fn parameter_values(&self, params: &ParamValues) -> Vec<u8> {
        if params.as_slice().is_empty() {
            self.encode_scalars(&[0.0])
        } else {
            self.encode_scalars(params.as_slice())
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod shader_tests;
