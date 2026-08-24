//! Private dispatch resource views shared by scalar and reduction execution.

use super::bindings::Binding;
use super::memory::{GpuMemoryLayout, STATUS_SENTINEL, STATUS_WORD_BYTES};
use crate::WgpuContext;
use crate::readback::{decode_singular_status, submit_and_readback};
use crate::scalar::WgpuScalarKernel;
use crate::{WgpuError, WgpuResult};
use laddu_expr::parameters::ParamValues;
use wgpu::util::DeviceExt;

/// The seven storage bindings consumed by the reduction shader family.
pub(crate) struct ReductionBindings<'a> {
    pub(crate) params: &'a wgpu::Buffer,
    pub(crate) cache: &'a wgpu::Buffer,
    pub(crate) weights: &'a wgpu::Buffer,
    pub(crate) config: &'a wgpu::Buffer,
    pub(crate) partials: &'a wgpu::Buffer,
    pub(crate) error: &'a wgpu::Buffer,
    pub(crate) solve_error: &'a wgpu::Buffer,
}

impl WgpuScalarKernel {
    pub(crate) fn evaluate_packed(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        inputs: &[u8],
        events: usize,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        if events == 0 {
            return Ok(Vec::new());
        }
        let values = self.parameter_values(params);
        let layout = self
            .memory_layout(params)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: &values,
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let (input_buffer, cache_buffer) = self.cache_buffers(context, inputs, events)?;
        let output_size = layout
            .event_bytes(layout.output_bytes, events)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let output = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar readback"),
            size: output_size.checked_add(STATUS_WORD_BYTES as u64).ok_or(
                WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                },
            )?,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let solve_error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu scalar solve error"),
                contents: bytemuck::bytes_of(&STATUS_SENTINEL),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu scalar bind group"),
                layout: &self.execution.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: Binding::Parameters.index(),
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Cache.index(),
                        resource: cache_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Output.index(),
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::SolveError.index(),
                        resource: solve_error.as_entire_binding(),
                    },
                ],
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
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.execution.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(GpuMemoryLayout::groups(events) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        encoder.copy_buffer_to_buffer(
            &solve_error,
            0,
            &staging,
            output_size,
            STATUS_WORD_BYTES as u64,
        );
        submit_and_readback(
            context,
            encoder,
            &staging,
            output_size + STATUS_WORD_BYTES as u64,
            |mapped| {
                let status = decode_singular_status(&mapped[output_size as usize..])?;
                if let Some(error) = status.error() {
                    return Err(error);
                }
                let result = self.decode_scalars(&mapped[..output_size as usize]);
                Ok(result
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|value| (value[0], value[1]))
                    .collect())
            },
        )
    }

    pub(super) fn cache_buffers(
        &self,
        context: &WgpuContext,
        inputs: &[u8],
        events: usize,
    ) -> WgpuResult<(wgpu::Buffer, wgpu::Buffer)> {
        let layout = self
            .buffer_layout()
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let input = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu event inputs"),
                contents: inputs,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let cache_size =
            layout
                .cache_buffer_bytes(events)
                .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                    required: usize::MAX,
                    available: context.memory_budget().unwrap_or(usize::MAX),
                })?;
        let cache = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu event cache"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok((input, cache))
    }

    pub(super) fn encode_cache_materialization(
        &self,
        context: &WgpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        cache: &wgpu::Buffer,
        solve_error: &wgpu::Buffer,
        events: usize,
    ) {
        let Some(cache_pipeline) = self.cache.as_ref() else {
            return;
        };
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu cache bind group"),
                layout: &cache_pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: Binding::Parameters.index(),
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Cache.index(),
                        resource: cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::SolveError.index(),
                        resource: solve_error.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&cache_pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(GpuMemoryLayout::groups(events) as u32, 1, 1);
    }
}

impl ReductionBindings<'_> {
    pub(crate) fn bind_group(
        &self,
        context: &WgpuContext,
        layout: &wgpu::BindGroupLayout,
        label: &'static str,
    ) -> wgpu::BindGroup {
        context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: Binding::Parameters.index(),
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Cache.index(),
                        resource: self.cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Weights.index(),
                        resource: self.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Config.index(),
                        resource: self.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::Partials.index(),
                        resource: self.partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::ReductionError.index(),
                        resource: self.error.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: Binding::SolveError.index(),
                        resource: self.solve_error.as_entire_binding(),
                    },
                ],
            })
    }
}
