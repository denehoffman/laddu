//! Private batch planning and packed-input representation for WGPU execution.

use std::ops::Range;

use laddu_data::RealVec4;
use laddu_data::data::EventBatch;
use laddu_expr::P4Component;

use crate::WgpuError;
use crate::scalar::EventInput;

/// Stable chunk ranges shared by direct evaluation and resident preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChunkPlan {
    pub(crate) ranges: Vec<Range<usize>>,
}

impl ChunkPlan {
    pub(crate) fn for_batch(batch_len: usize, chunk_len: usize) -> Self {
        assert!(chunk_len > 0, "chunk plans require a non-zero chunk length");
        Self {
            ranges: (0..batch_len)
                .step_by(chunk_len)
                .map(|start| start..start.saturating_add(chunk_len).min(batch_len))
                .collect(),
        }
    }

    pub(crate) fn matches_event_counts<I>(&self, prepared: I) -> bool
    where
        I: ExactSizeIterator<Item = usize>,
    {
        self.ranges.len() == prepared.len()
            && self
                .ranges
                .iter()
                .zip(prepared)
                .all(|(range, events)| range.len() == events)
    }
}

/// Host-side packed inputs and weights for one planned chunk.
pub(crate) struct PackedChunk {
    pub(crate) start: usize,
    pub(crate) events: usize,
    pub(crate) inputs: Vec<u8>,
    pub(crate) weights: Vec<u8>,
}

enum BoundEventInput<'a> {
    Scalar(&'a [f64]),
    P4(&'a [RealVec4], P4Component),
}

/// Schema-bound columns for one complete batch operation.
///
/// Binding is intentionally separate from packing: chunk traversal can reuse
/// these borrowed columns without slicing the `EventBatch` (which would copy
/// every selected column for every chunk).
pub(crate) struct BoundBatch<'a> {
    inputs: Vec<BoundEventInput<'a>>,
    batch: &'a EventBatch,
}

impl BoundBatch<'_> {
    pub(crate) fn pack_range(&self, range: Range<usize>) -> Vec<f64> {
        if self.inputs.is_empty() {
            return vec![0.0, 0.0];
        }
        let width = self.inputs.len().max(1) * 2;
        let mut values = Vec::with_capacity(range.len() * width);
        for row in range {
            for bound_input in &self.inputs {
                match bound_input {
                    BoundEventInput::Scalar(column) => values.extend([column[row], 0.0]),
                    BoundEventInput::P4(column, component) => {
                        let value = &column[row];
                        let value = match component {
                            P4Component::Px => value.px,
                            P4Component::Py => value.py,
                            P4Component::Pz => value.pz,
                            P4Component::E => value.e,
                        };
                        values.extend([value, 0.0]);
                    }
                }
            }
        }
        values
    }

    pub(crate) fn pack_weights(&self, range: Range<usize>) -> Vec<f64> {
        range.map(|row| self.batch.weights_at(row)).collect()
    }
}

pub(crate) fn bind_batch<'a>(
    batch: &'a EventBatch,
    inputs: &[EventInput],
) -> Result<BoundBatch<'a>, WgpuError> {
    let bound = inputs
        .iter()
        .map(|input| match input {
            EventInput::Scalar(name) => {
                let binding = batch
                    .schema()
                    .bind_scalar(name)
                    .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                batch
                    .scalar_column_bound(&binding)
                    .map(BoundEventInput::Scalar)
                    .map_err(|_| WgpuError::MissingEventColumn(name.clone()))
            }
            EventInput::P4(name, component) => {
                let binding = batch
                    .schema()
                    .bind_p4(name)
                    .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                batch
                    .p4_column_bound(&binding)
                    .map(|column| BoundEventInput::P4(column, *component))
                    .map_err(|_| WgpuError::MissingEventColumn(name.clone()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BoundBatch {
        inputs: bound,
        batch,
    })
}

impl crate::scalar::WgpuScalarKernel {
    pub(crate) fn pack_bound_chunk(
        &self,
        batch: &BoundBatch<'_>,
        range: Range<usize>,
    ) -> PackedChunk {
        let start = range.start;
        let events = range.len();
        PackedChunk {
            start,
            events,
            inputs: self.encode_scalars(&batch.pack_range(range.clone())),
            weights: self.encode_scalars(&batch.pack_weights(range)),
        }
    }
}

/// Rebase an event-local backend error to its original batch index.
pub(crate) fn rebase_error(error: WgpuError, start: usize) -> WgpuError {
    match error {
        WgpuError::NonPositiveEvent(index) => WgpuError::NonPositiveEvent(start + index),
        WgpuError::SingularMatrixEvent(index) => WgpuError::SingularMatrixEvent(start + index),
        other => other,
    }
}
