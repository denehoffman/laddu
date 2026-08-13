use laddu_kernel::ir::{KernelValueId, KernelValueKind};

use super::ReverseState;
use crate::{AutodiffError, AutodiffResult};

// `ScalarKernelIr` validation guarantees that every operand ID exists and
// that instruction operands have compatible shapes. Keep ID lookup
// infallible under that boundary contract, while reporting derivative-rule
// shape assumptions as `InvalidKernel` so an internal rule mismatch never
// becomes a user-visible panic.
impl ReverseState<'_> {
    pub(super) fn kind(&self, id: KernelValueId) -> KernelValueKind {
        self.primal.values()[id.index()].kind
    }

    pub(super) fn expect_matrix_kind(
        &self,
        id: KernelValueId,
        operation: &str,
    ) -> AutodiffResult<(usize, usize)> {
        match self.kind(id) {
            KernelValueKind::Matrix { rows, cols } => Ok((rows, cols)),
            kind => Err(AutodiffError::InvalidKernel(format!(
                "{operation} expected matrix value {}, found {kind:?}",
                id.index(),
            ))),
        }
    }

    pub(super) fn expect_vector_kind(
        &self,
        id: KernelValueId,
        operation: &str,
    ) -> AutodiffResult<usize> {
        match self.kind(id) {
            KernelValueKind::Vector { len } => Ok(len),
            kind => Err(AutodiffError::InvalidKernel(format!(
                "{operation} expected vector value {}, found {kind:?}",
                id.index(),
            ))),
        }
    }

    pub(super) fn flat_index(
        row: usize,
        cols: usize,
        col: usize,
        operation: &str,
    ) -> AutodiffResult<usize> {
        row.checked_mul(cols)
            .and_then(|offset| offset.checked_add(col))
            .ok_or_else(|| {
                AutodiffError::InvalidKernel(format!(
                    "{operation} row-major index overflow for row {row}, column {col}, and width {cols}",
                ))
            })
    }
}
