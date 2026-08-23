use super::*;

fn validate_root_bounds(values: &[KernelValue], root: KernelValueId) -> Result<(), KernelIrError> {
    if root.index() >= values.len() {
        return Err(KernelIrError::RootOutOfBounds {
            root: root.index(),
            len: values.len(),
        });
    }
    Ok(())
}

fn validate_scalar_root(
    values: &[KernelValue],
    root: KernelValueId,
    operation: &'static str,
    message: &'static str,
) -> Result<(), KernelIrError> {
    if !values[root.index()].kind.is_scalar() {
        return Err(KernelInstruction::shape_error(
            root.index(),
            operation,
            message,
        ));
    }
    Ok(())
}

fn validate_cache_outputs(
    values: &[KernelValue],
    outputs: &[KernelValueId],
) -> Result<(), KernelIrError> {
    for output in outputs {
        if output.index() >= values.len() {
            return Err(KernelIrError::CacheOutputOutOfBounds {
                output: output.index(),
                len: values.len(),
            });
        }
    }
    Ok(())
}

fn validate_gradient_outputs(
    values: &[KernelValue],
    outputs: &[KernelValueId],
) -> Result<(), KernelIrError> {
    for output in outputs {
        let Some(value) = values.get(output.index()) else {
            return Err(KernelIrError::GradientOutOfBounds {
                output: output.index(),
                len: values.len(),
            });
        };
        if value.kind != KernelValueKind::Real {
            return Err(KernelIrError::GradientKindMismatch {
                output: output.index(),
                actual: value.kind,
            });
        }
    }
    Ok(())
}

fn required_values(values: &[KernelValue], outputs: &[KernelValueId]) -> Vec<bool> {
    let mut required = vec![false; values.len()];
    let mut pending = outputs.to_vec();
    while let Some(id) = pending.pop() {
        if required[id.index()] {
            continue;
        }
        required[id.index()] = true;
        values[id.index()]
            .instruction
            .for_each_operand(|operand| pending.push(operand));
    }
    required
}

impl ScalarKernelIr {
    /// Validates values and constructs a scalar kernel rooted at `root`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the value list is empty, `root` or an
    /// operand is out of bounds, values are not topologically ordered, a
    /// value's kind or class is inconsistent with its instruction, or the
    /// root is not scalar.
    pub fn new(values: Vec<KernelValue>, root: KernelValueId) -> Result<Self, KernelIrError> {
        let ir = Self { values, root };
        ir.validate()?;
        Ok(ir)
    }

    /// Revalidates ordering, types, classes, and the scalar root.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the IR is empty, its root or an operand
    /// is out of bounds, its values are not topologically ordered, or a
    /// value's kind, class, or shape is inconsistent with its instruction.
    pub fn validate(&self) -> Result<(), KernelIrError> {
        if self.values.is_empty() {
            return validate_graph(&self.values);
        }
        validate_root_bounds(&self.values, self.root)?;
        validate_graph(&self.values)?;
        validate_scalar_root(
            &self.values,
            self.root,
            "kernel root",
            "root must be scalar",
        )
    }

    /// Returns all IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    /// Returns the scalar output identifier.
    pub fn root(&self) -> KernelValueId {
        self.root
    }

    /// Returns a mask of values needed to evaluate the scalar root.
    pub fn required_values(&self) -> Vec<bool> {
        required_values(&self.values, std::slice::from_ref(&self.root))
    }
}

impl CacheKernelIr {
    /// Validates values and constructs a cache kernel with the given outputs.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when `outputs` is empty, an output or operand
    /// is out of bounds, values are not topologically ordered, or a value's
    /// kind, class, or shape is inconsistent with its instruction.
    pub fn new(
        values: Vec<KernelValue>,
        outputs: Vec<KernelValueId>,
    ) -> Result<Self, KernelIrError> {
        if outputs.is_empty() {
            return Err(KernelIrError::EmptyCacheOutputs);
        }
        validate_graph(&values)?;
        validate_cache_outputs(&values, &outputs)?;
        Ok(Self { values, outputs })
    }

    /// Returns all IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    /// Returns cache output identifiers in storage order.
    pub fn outputs(&self) -> &[KernelValueId] {
        &self.outputs
    }
}

impl GradientKernelIr {
    /// Validates and constructs a gradient kernel.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the primal IR is invalid, the primal
    /// root is not scalar, a gradient output is out of bounds, or a gradient
    /// output is not real-valued.
    pub fn new(
        values: Vec<KernelValue>,
        primal_root: KernelValueId,
        outputs: Vec<KernelValueId>,
        component: OutputComponent,
    ) -> Result<Self, KernelIrError> {
        let ir = Self {
            values,
            primal_root,
            outputs,
            component,
        };
        ir.validate()?;
        Ok(ir)
    }

    /// Revalidates the primal root and real gradient outputs.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the primal IR is invalid, the primal
    /// root is not scalar, a gradient output is out of bounds, or a gradient
    /// output is not real-valued.
    pub fn validate(&self) -> Result<(), KernelIrError> {
        if self.values.is_empty() {
            return validate_graph(&self.values);
        }
        validate_root_bounds(&self.values, self.primal_root)?;
        validate_graph(&self.values)?;
        validate_scalar_root(
            &self.values,
            self.primal_root,
            "gradient primal root",
            "primal root must be scalar",
        )?;
        validate_gradient_outputs(&self.values, &self.outputs)
    }

    /// Returns all primal and derivative IR values in topological order.
    pub fn values(&self) -> &[KernelValue] {
        &self.values
    }

    /// Returns the primal scalar output identifier.
    pub fn primal_root(&self) -> KernelValueId {
        self.primal_root
    }

    /// Returns derivative output identifiers.
    pub fn outputs(&self) -> &[KernelValueId] {
        &self.outputs
    }

    /// Returns the differentiated component of the complex primal.
    pub fn component(&self) -> OutputComponent {
        self.component
    }

    /// Returns a mask of values needed to evaluate the derivative outputs.
    pub fn required_values(&self) -> Vec<bool> {
        required_values(&self.values, &self.outputs)
    }
}
