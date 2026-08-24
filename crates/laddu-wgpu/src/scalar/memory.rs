//! Backend-local memory geometry, allocation policy, and dispatch constants.

use laddu_memory::{FootprintOverflow, MemoryFootprint};

use crate::scalar::WgpuScalarKernel;
use crate::{WgpuContext, WgpuError, WgpuResult};

pub(crate) const WORKGROUP_SIZE: usize = 64;
pub(crate) const COMPLEX_COMPONENTS: usize = 2;
pub(crate) const STATUS_WORD_BYTES: usize = size_of::<u32>();
pub(crate) const STATUS_SENTINEL: u32 = u32::MAX;
pub(crate) const REDUCTION_STATUS_WORDS: usize = 2;
pub(crate) const PREPARED_AUXILIARY_WORDS: usize = 5;
pub(crate) const MIN_BUFFER_BYTES: usize = 8;

pub(crate) const fn workgroups(events: usize) -> usize {
    events / WORKGROUP_SIZE + (!events.is_multiple_of(WORKGROUP_SIZE)) as usize
}

pub(crate) const fn status_bytes(words: usize) -> usize {
    words * STATUS_WORD_BYTES
}

/// Checked byte geometry shared by WGPU planning and buffer construction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuMemoryLayout {
    scalar_size: u64,
    pub(crate) input_bytes: MemoryFootprint,
    pub(crate) cache_bytes: MemoryFootprint,
    pub(crate) weight_bytes: MemoryFootprint,
    pub(crate) output_bytes: MemoryFootprint,
    pub(crate) partial_bytes: MemoryFootprint,
    pub(crate) parameter_bytes: MemoryFootprint,
}

impl GpuMemoryLayout {
    pub(crate) fn new(
        scalar_size: usize,
        event_inputs: usize,
        cache_width: usize,
        partial_width: usize,
        parameter_count: usize,
    ) -> Result<Self, FootprintOverflow> {
        let scalar_size = u64::try_from(scalar_size).map_err(|_| FootprintOverflow::Conversion)?;
        let scalar = MemoryFootprint::per_event(scalar_size);
        Ok(Self {
            scalar_size,
            input_bytes: scalar
                .checked_scale_usize(event_inputs)?
                .checked_scale(COMPLEX_COMPONENTS as u64)?,
            cache_bytes: scalar
                .checked_scale_usize(cache_width)?
                .checked_scale(COMPLEX_COMPONENTS as u64)?,
            weight_bytes: scalar,
            output_bytes: scalar.checked_scale(COMPLEX_COMPONENTS as u64)?,
            partial_bytes: scalar.checked_scale_usize(partial_width)?,
            parameter_bytes: MemoryFootprint::fixed(scalar_size)
                .checked_scale_usize(parameter_count.max(1))?,
        })
    }

    pub(crate) fn prepared_footprint(self) -> Result<MemoryFootprint, FootprintOverflow> {
        let per_event = self
            .input_bytes
            .checked_add(self.cache_bytes)?
            .checked_add(self.partial_bytes)?
            .checked_add(MemoryFootprint::per_event(1))?;
        self.parameter_bytes
            .checked_add(MemoryFootprint::fixed(16))?
            .checked_add(per_event)
    }

    pub(crate) fn evaluation_footprint(self) -> Result<MemoryFootprint, FootprintOverflow> {
        let per_event = self
            .input_bytes
            .checked_add(self.cache_bytes)?
            .checked_add(self.output_bytes.checked_scale(2)?)?;
        self.parameter_bytes.checked_add(per_event)
    }

    pub(crate) fn groups(events: usize) -> usize {
        workgroups(events)
    }

    pub(crate) fn event_bytes(
        self,
        footprint: MemoryFootprint,
        events: usize,
    ) -> Result<u64, FootprintOverflow> {
        footprint.checked_peak_bytes(events)
    }

    pub(crate) fn input_buffer_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        let bytes = self.event_bytes(self.input_bytes, events)?;
        if bytes == 0 {
            self.scalar_size
                .checked_mul(COMPLEX_COMPONENTS as u64)
                .ok_or(FootprintOverflow::Multiplication)
        } else {
            Ok(bytes)
        }
    }

    pub(crate) fn cache_buffer_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        Ok(self
            .event_bytes(self.cache_bytes, events)?
            .max(MIN_BUFFER_BYTES as u64))
    }

    pub(crate) fn weight_buffer_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        self.event_bytes(self.weight_bytes, events)
    }

    pub(crate) fn partial_buffer_bytes(
        self,
        events: usize,
        minimum: usize,
    ) -> Result<u64, FootprintOverflow> {
        let groups = Self::groups(events);
        let bytes = self
            .partial_bytes
            .bytes_per_event
            .checked_mul(u64::try_from(groups).map_err(|_| FootprintOverflow::Conversion)?)
            .ok_or(FootprintOverflow::Multiplication)?;
        Ok(bytes.max(minimum as u64))
    }

    pub(crate) fn scalar_partial_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        self.scalar_size
            .checked_mul(
                u64::try_from(Self::groups(events)).map_err(|_| FootprintOverflow::Conversion)?,
            )
            .ok_or(FootprintOverflow::Multiplication)
    }

    pub(crate) fn staging_buffer_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        self.partial_buffer_bytes(events, 0)?
            .checked_add(status_bytes(REDUCTION_STATUS_WORDS) as u64)
            .map(|bytes| bytes.max(MIN_BUFFER_BYTES as u64))
            .ok_or(FootprintOverflow::Addition)
    }

    pub(crate) fn prepared_resident_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        let partial = self.partial_buffer_bytes(events, 0)?;
        let input = self.input_buffer_bytes(events)?;
        let weights = self.weight_buffer_bytes(events)?;
        let cache = self.cache_buffer_bytes(events)?;
        let partials = partial
            .checked_mul(2)
            .ok_or(FootprintOverflow::Multiplication)?;
        input
            .checked_add(weights)
            .and_then(|bytes| bytes.checked_add(cache))
            .and_then(|bytes| bytes.checked_add(self.parameter_bytes.fixed_bytes))
            .and_then(|bytes| bytes.checked_add(partials))
            // In addition to partials and staging, each prepared chunk owns
            // a config word plus independent reduction and solve status
            // buffers. Count all of those allocations in resident usage.
            .and_then(|bytes| bytes.checked_add(status_bytes(PREPARED_AUXILIARY_WORDS) as u64))
            .ok_or(FootprintOverflow::Addition)
    }

    pub(crate) fn binding_limit(self, max_binding: u64, reduction: bool) -> usize {
        let mut limit = u64::from(u32::MAX);
        let widths = [
            self.input_bytes.bytes_per_event,
            self.cache_bytes.bytes_per_event,
            if reduction {
                self.partial_bytes.bytes_per_event
            } else {
                self.output_bytes.bytes_per_event
            },
        ];
        for width in widths {
            if let Some(events) = max_binding.checked_div(width) {
                limit = limit.min(events);
            }
        }
        usize::try_from(limit).unwrap_or(usize::MAX)
    }
}

impl WgpuScalarKernel {
    pub(crate) fn max_chunk_events(
        &self,
        context: &WgpuContext,
        params: &laddu_expr::parameters::ParamValues,
        reduction: bool,
    ) -> WgpuResult<usize> {
        let layout = self
            .memory_layout(params)
            .map_err(|_| WgpuError::MemoryBudgetTooSmall {
                required: usize::MAX,
                available: context.memory_budget().unwrap_or(usize::MAX),
            })?;
        let footprint = if reduction {
            layout.prepared_footprint()
        } else {
            layout.evaluation_footprint()
        }
        .map_err(|_| WgpuError::MemoryBudgetTooSmall {
            required: usize::MAX,
            available: context.memory_budget().unwrap_or(usize::MAX),
        })?;
        let max_binding = context
            .info()
            .max_buffer_size
            .min(context.info().max_storage_buffer_binding_size)
            .min(usize::MAX as u64);
        let mut max_events = layout.binding_limit(max_binding, reduction);
        if let Some(budget) = context.memory_budget() {
            let available = u64::try_from(budget)
                .unwrap_or(u64::MAX)
                .saturating_sub(footprint.fixed_bytes);
            let per_event = footprint.bytes_per_event.max(1);
            max_events =
                max_events.min(usize::try_from(available / per_event).unwrap_or(usize::MAX));
            if max_events == 0 {
                return Err(WgpuError::MemoryBudgetTooSmall {
                    required: usize::try_from(footprint.fixed_bytes.saturating_add(per_event))
                        .unwrap_or(usize::MAX),
                    available: budget,
                });
            }
        }
        if max_events == 0 {
            return Err(WgpuError::MemoryBudgetTooSmall {
                required: usize::try_from(
                    footprint
                        .fixed_bytes
                        .saturating_add(footprint.bytes_per_event.max(1)),
                )
                .unwrap_or(usize::MAX),
                available: usize::try_from(max_binding).unwrap_or(usize::MAX),
            });
        }
        Ok(max_events)
    }
}
