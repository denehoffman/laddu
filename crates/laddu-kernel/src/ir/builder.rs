use super::*;

impl KernelIrBuilder {
    /// Creates a builder initialized with a scalar kernel's values.
    pub fn from_scalar(ir: &ScalarKernelIr) -> Self {
        Self {
            values: ir.values.clone(),
        }
    }

    /// Appends an instruction after validating its operands and inferred type.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when an operand does not precede the new
    /// instruction, the operand shapes are incompatible, or the instruction's
    /// value kind cannot be inferred.
    pub fn push(&mut self, instruction: KernelInstruction) -> Result<KernelValueId, KernelIrError> {
        let index = self.values.len();
        instruction.validate_operand_order(index)?;
        let kind = instruction
            .expected_kind(&self.values, index)?
            .ok_or_else(|| {
                KernelInstruction::shape_error(
                    index,
                    "derived instruction",
                    "instruction requires an explicitly supplied value kind",
                )
            })?;
        let class = instruction.expected_class(&self.values);
        let id = KernelValueId::from_index(index);
        self.values.push(KernelValue {
            kind,
            class,
            instruction,
        });
        Ok(id)
    }

    /// Finishes the builder as a validated gradient kernel.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the accumulated primal IR is invalid,
    /// `primal_root` is not scalar, a gradient output is out of bounds, or a
    /// gradient output is not real-valued.
    pub fn finish_gradient(
        self,
        primal_root: KernelValueId,
        outputs: Vec<KernelValueId>,
        component: OutputComponent,
    ) -> Result<GradientKernelIr, KernelIrError> {
        GradientKernelIr::new(self.values, primal_root, outputs, component)
    }

    /// Returns the values accumulated so far.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }
}
