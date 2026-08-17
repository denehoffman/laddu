use crate::{
    data::EventBatch,
    schema::{Precision, Schema},
};
use laddu_memory::{FootprintOverflow, MemoryFootprint};

/// Checked, format-neutral layout information for event-batch planning.
///
/// The logical schema and a concrete [`EventBatch`] can disagree about
/// whether an explicit weight allocation is present. A schema-derived layout
/// uses the schema's declared weight column, while a batch-derived layout uses
/// the allocation actually retained by that batch. This distinction keeps
/// source planning conservative without charging implicit unit weights to
/// in-memory batches.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLayout {
    p4s: usize,
    scalars: usize,
    schema_weight: bool,
    explicit_weight: bool,
}

impl BatchLayout {
    /// Derives a layout from a logical schema.
    pub fn from_schema(schema: &Schema) -> Self {
        Self {
            p4s: schema.n_p4s(),
            scalars: schema.n_scalars(),
            schema_weight: schema.has_weight(),
            explicit_weight: schema.has_weight(),
        }
    }

    /// Derives a layout from a concrete batch, retaining both logical and
    /// allocated weight state.
    pub fn from_batch(batch: &EventBatch) -> Self {
        let schema = batch.schema();
        Self {
            p4s: schema.n_p4s(),
            scalars: schema.n_scalars(),
            schema_weight: schema.has_weight(),
            explicit_weight: batch.weights_column().is_some(),
        }
    }

    /// Creates a layout from counts and explicit weight state.
    pub const fn new(
        p4s: usize,
        scalars: usize,
        schema_weight: bool,
        explicit_weight: bool,
    ) -> Self {
        Self {
            p4s,
            scalars,
            schema_weight,
            explicit_weight,
        }
    }

    /// Returns the number of four-momentum columns.
    pub const fn n_p4s(self) -> usize {
        self.p4s
    }

    /// Returns the number of scalar columns.
    pub const fn n_scalars(self) -> usize {
        self.scalars
    }

    /// Returns whether the logical schema declares a weight column.
    pub const fn schema_has_weight(self) -> bool {
        self.schema_weight
    }

    /// Returns whether the represented batch allocates an explicit weight
    /// column.
    pub const fn has_explicit_weight(self) -> bool {
        self.explicit_weight
    }

    /// Returns bytes per event for the represented batch at `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout cannot be represented in
    /// bytes.
    pub fn bytes_per_event(self, precision: Precision) -> Result<u64, FootprintOverflow> {
        self.bytes_per_event_for(precision, self.explicit_weight)
    }

    /// Returns bytes per event for the schema-declared layout at `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout cannot be represented in
    /// bytes.
    pub fn schema_bytes_per_event(self, precision: Precision) -> Result<u64, FootprintOverflow> {
        self.bytes_per_event_for(precision, self.schema_weight)
    }

    /// Returns the represented batch's per-event footprint at `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout cannot be represented in
    /// bytes.
    pub fn footprint(self, precision: Precision) -> Result<MemoryFootprint, FootprintOverflow> {
        Ok(MemoryFootprint::per_event(self.bytes_per_event(precision)?))
    }

    /// Returns the schema-declared per-event footprint at `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout cannot be represented in
    /// bytes.
    pub fn schema_footprint(
        self,
        precision: Precision,
    ) -> Result<MemoryFootprint, FootprintOverflow> {
        Ok(MemoryFootprint::per_event(
            self.schema_bytes_per_event(precision)?,
        ))
    }

    /// Returns the represented batch footprint after `copies` simultaneous
    /// copies/stages.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout or copy count cannot be
    /// represented in bytes.
    pub fn working_set(
        self,
        precision: Precision,
        copies: usize,
    ) -> Result<MemoryFootprint, FootprintOverflow> {
        self.footprint(precision)?.checked_scale_usize(copies)
    }

    /// Returns the schema-declared footprint after `copies` simultaneous
    /// copies/stages.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout or copy count cannot be
    /// represented in bytes.
    pub fn schema_working_set(
        self,
        precision: Precision,
        copies: usize,
    ) -> Result<MemoryFootprint, FootprintOverflow> {
        self.schema_footprint(precision)?
            .checked_scale_usize(copies)
    }

    /// Adds a fixed allocation to the represented batch footprint.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when the layout or combined footprint
    /// cannot be represented in bytes.
    pub fn with_fixed(
        self,
        precision: Precision,
        fixed_bytes: u64,
    ) -> Result<MemoryFootprint, FootprintOverflow> {
        MemoryFootprint::fixed(fixed_bytes).checked_add(self.footprint(precision)?)
    }

    fn bytes_per_event_for(
        self,
        precision: Precision,
        include_weight: bool,
    ) -> Result<u64, FootprintOverflow> {
        let p4_values = self
            .p4s
            .checked_mul(4)
            .ok_or(FootprintOverflow::Multiplication)?;
        let values = p4_values
            .checked_add(self.scalars)
            .ok_or(FootprintOverflow::Addition)?
            .checked_add(usize::from(include_weight))
            .ok_or(FootprintOverflow::Addition)?;
        let values = u64::try_from(values).map_err(|_| FootprintOverflow::Conversion)?;
        let width = match precision {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        values
            .checked_mul(width)
            .ok_or(FootprintOverflow::Multiplication)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{RealVec4, data::EventBatch};

    fn schema(weight: bool) -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["x", "y"], weight).unwrap())
    }

    #[test]
    fn schema_layout_calculates_both_precisions() {
        let layout = BatchLayout::from_schema(&schema(true));
        assert_eq!(layout.schema_bytes_per_event(Precision::F64).unwrap(), 56);
        assert_eq!(layout.schema_bytes_per_event(Precision::F32).unwrap(), 28);
        assert_eq!(
            layout.schema_working_set(Precision::F64, 2).unwrap(),
            MemoryFootprint::new(0, 112)
        );
    }

    #[test]
    fn batch_layout_charges_only_allocated_explicit_weights() {
        let schema = schema(true);
        let batch = EventBatch::new(
            Arc::clone(&schema),
            vec![Arc::from([RealVec4::new(0.0, 0.0, 0.0, 0.0)])],
            vec![Arc::from([1.0]), Arc::from([2.0])],
            None,
        )
        .unwrap();
        let layout = BatchLayout::from_batch(&batch);
        assert!(layout.schema_has_weight());
        assert!(!layout.has_explicit_weight());
        assert_eq!(layout.bytes_per_event(Precision::F64).unwrap(), 48);
        assert_eq!(layout.schema_bytes_per_event(Precision::F64).unwrap(), 56);
    }

    #[test]
    fn checked_layout_arithmetic_reports_overflow() {
        let layout = BatchLayout::new(usize::MAX, usize::MAX, true, true);
        assert_eq!(
            layout.bytes_per_event(Precision::F64),
            Err(FootprintOverflow::Multiplication)
        );
        assert_eq!(
            MemoryFootprint::new(u64::MAX, 0).checked_add(MemoryFootprint::fixed(1)),
            Err(FootprintOverflow::Addition)
        );
    }
}
