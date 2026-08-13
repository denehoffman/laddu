use laddu_kernel::ir::{KernelValue, KernelValueId};

use crate::{AutodiffError, AutodiffResult};

pub(super) struct AdjointStore {
    pending: Vec<ValueContributions>,
    resolved: Vec<Option<ResolvedAdjoint>>,
}

struct ValueContributions(Vec<Vec<KernelValueId>>);

#[derive(Clone)]
struct ResolvedAdjoint(Vec<KernelValueId>);

impl AdjointStore {
    pub(super) fn new(values: &[KernelValue]) -> Self {
        Self {
            pending: values
                .iter()
                .map(|value| ValueContributions(vec![Vec::new(); value.kind.width()]))
                .collect(),
            resolved: vec![None; values.len()],
        }
    }

    pub(super) fn add_value(
        &mut self,
        target: KernelValueId,
        contribution: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let pending = &mut self.pending[target.index()].0;
        if pending.len() != contribution.len() {
            return Err(AutodiffError::InvalidKernel(format!(
                "gradient contribution width {} does not match value {} width {}",
                contribution.len(),
                target.index(),
                pending.len(),
            )));
        }
        for (target, contribution) in pending.iter_mut().zip(contribution) {
            target.push(*contribution);
        }
        Ok(())
    }

    pub(super) fn add_element(
        &mut self,
        target: KernelValueId,
        element: usize,
        contribution: KernelValueId,
    ) -> AutodiffResult<()> {
        let pending = &mut self.pending[target.index()].0;
        let width = pending.len();
        let Some(target_element) = pending.get_mut(element) else {
            return Err(AutodiffError::InvalidKernel(format!(
                "gradient contribution element {element} exceeds value {} width {width}",
                target.index(),
            )));
        };
        target_element.push(contribution);
        Ok(())
    }

    pub(super) fn take_pending(
        &mut self,
        primal: KernelValueId,
    ) -> Option<Vec<Vec<KernelValueId>>> {
        let pending = &mut self.pending[primal.index()].0;
        if pending.iter().all(Vec::is_empty) {
            return None;
        }
        Some(pending.iter_mut().map(std::mem::take).collect())
    }

    pub(super) fn set_resolved(&mut self, primal: KernelValueId, values: Vec<KernelValueId>) {
        debug_assert_eq!(self.pending[primal.index()].0.len(), values.len());
        self.resolved[primal.index()] = Some(ResolvedAdjoint(values));
    }

    pub(super) fn resolved(&self, primal: KernelValueId) -> Option<&[KernelValueId]> {
        self.resolved[primal.index()]
            .as_ref()
            .map(|adjoint| adjoint.0.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use laddu_kernel::ir::{KernelInstruction, KernelValueClass, KernelValueKind};

    use super::*;

    fn value(kind: KernelValueKind) -> KernelValue {
        KernelValue {
            kind,
            class: KernelValueClass::Invariant,
            instruction: KernelInstruction::RealConstant(0.0),
        }
    }

    #[test]
    fn reports_value_width_mismatch() {
        let mut store = AdjointStore::new(&[value(KernelValueKind::Vector { len: 2 })]);
        let error = store
            .add_value(
                KernelValueId::from_index(0),
                &[KernelValueId::from_index(1)],
            )
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot differentiate kernel instruction: gradient contribution width 1 does not match value 0 width 2"
        );
    }

    #[test]
    fn reports_element_out_of_bounds() {
        let mut store = AdjointStore::new(&[value(KernelValueKind::Real)]);
        let error = store
            .add_element(
                KernelValueId::from_index(0),
                1,
                KernelValueId::from_index(1),
            )
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot differentiate kernel instruction: gradient contribution element 1 exceeds value 0 width 1"
        );
    }
}
