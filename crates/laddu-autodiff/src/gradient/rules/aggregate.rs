use laddu_expr::UnaryOp;
use laddu_kernel::ir::KernelValueId;

use super::super::ReverseState;
use crate::AutodiffResult;

impl ReverseState<'_> {
    pub(in crate::gradient) fn complex_pullback(
        &mut self,
        re: KernelValueId,
        im: KernelValueId,
        adjoint: KernelValueId,
    ) -> AutodiffResult<()> {
        let re_contribution = self.unary(UnaryOp::Real, adjoint)?;
        let im_contribution = self.unary(UnaryOp::Imag, adjoint)?;
        self.accumulate(re, &[re_contribution])?;
        self.accumulate(im, &[im_contribution])
    }

    pub(in crate::gradient) fn aggregate_pullback(
        &mut self,
        entries: &[KernelValueId],
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        for (entry, contribution) in entries.iter().zip(adjoint) {
            self.accumulate(*entry, &[*contribution])?;
        }
        Ok(())
    }
}
