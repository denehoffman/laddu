use std::fmt::{Debug, Display};

use auto_ops::impl_op_ex;
use dyn_clone::DynClone;
#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
use crate::{
    data::{Dataset, DatasetMetadata, Event},
    LadduResult,
};

/// Standard methods for extracting some value from an event view.
#[typetag::serde(tag = "type")]
pub trait Variable: DynClone + Send + Sync + Debug + Display {
    /// Bind the variable to dataset metadata so that any referenced names can be resolved to
    /// concrete indices. Implementations that do not require metadata may keep the default
    /// no-op.
    fn bind(&mut self, _metadata: &DatasetMetadata) -> LadduResult<()> {
        Ok(())
    }

    /// This method extracts a single value (like a mass) from an event access view.
    fn value(&self, event: &Event<'_>) -> f64;

    /// This method distributes [`Variable::value`] over each event in a [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Variable::value_on`] instead.
    fn value_on_local(&self, dataset: &Dataset) -> LadduResult<Vec<f64>> {
        let mut variable = dyn_clone::clone_box(self);
        variable.bind(dataset.metadata())?;
        #[cfg(feature = "rayon")]
        let local_values: Vec<f64> = (0..dataset.n_events_local())
            .into_par_iter()
            .map(|event_index| {
                let event = dataset.event_view(event_index);
                variable.value(&event)
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let local_values: Vec<f64> = (0..dataset.n_events_local())
            .map(|event_index| {
                let event = dataset.event_view(event_index);
                variable.value(&event)
            })
            .collect();
        Ok(local_values)
    }

    /// This method distributes the [`Variable::value`] method over each [`EventData`] in a
    /// [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Variable::value_on`] instead.
    #[cfg(feature = "mpi")]
    fn value_on_mpi(&self, dataset: &Dataset, world: &SimpleCommunicator) -> LadduResult<Vec<f64>> {
        let local_weights = self.value_on_local(dataset)?;
        let n_events = dataset.n_events();
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_weights, &mut partitioned_buffer);
        }
        Ok(buffer)
    }

    /// This method distributes the [`Variable::value`] method over each [`EventData`] in a
    /// [`Dataset`].
    fn value_on(&self, dataset: &Dataset) -> LadduResult<Vec<f64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.value_on_mpi(dataset, &world);
            }
        }
        self.value_on_local(dataset)
    }

    /// Create an [`VariableExpression`] that evaluates to `self == val`
    fn eq(&self, val: f64) -> VariableExpression
    where
        Self: std::marker::Sized + 'static,
    {
        VariableExpression::Eq(dyn_clone::clone_box(self), val)
    }

    /// Create an [`VariableExpression`] that evaluates to `self < val`
    fn lt(&self, val: f64) -> VariableExpression
    where
        Self: std::marker::Sized + 'static,
    {
        VariableExpression::Lt(dyn_clone::clone_box(self), val)
    }

    /// Create an [`VariableExpression`] that evaluates to `self > val`
    fn gt(&self, val: f64) -> VariableExpression
    where
        Self: std::marker::Sized + 'static,
    {
        VariableExpression::Gt(dyn_clone::clone_box(self), val)
    }

    /// Create an [`VariableExpression`] that evaluates to `self >= val`
    fn ge(&self, val: f64) -> VariableExpression
    where
        Self: std::marker::Sized + 'static,
    {
        self.gt(val).or(&self.eq(val))
    }

    /// Create an [`VariableExpression`] that evaluates to `self <= val`
    fn le(&self, val: f64) -> VariableExpression
    where
        Self: std::marker::Sized + 'static,
    {
        self.lt(val).or(&self.eq(val))
    }
}
dyn_clone::clone_trait_object!(Variable);

/// Expressions which can be used to compare [`Variable`]s to [`f64`]s.
#[derive(Clone, Debug)]
pub enum VariableExpression {
    /// Expression which is true when the variable is equal to the float.
    Eq(Box<dyn Variable>, f64),
    /// Expression which is true when the variable is less than the float.
    Lt(Box<dyn Variable>, f64),
    /// Expression which is true when the variable is greater than the float.
    Gt(Box<dyn Variable>, f64),
    /// Expression which is true when both inner expressions are true.
    And(Box<VariableExpression>, Box<VariableExpression>),
    /// Expression which is true when either inner expression is true.
    Or(Box<VariableExpression>, Box<VariableExpression>),
    /// Expression which is true when the inner expression is false.
    Not(Box<VariableExpression>),
}

impl VariableExpression {
    /// Construct an [`VariableExpression::And`] from the current expression and another.
    pub fn and(&self, rhs: &VariableExpression) -> VariableExpression {
        VariableExpression::And(Box::new(self.clone()), Box::new(rhs.clone()))
    }

    /// Construct an [`VariableExpression::Or`] from the current expression and another.
    pub fn or(&self, rhs: &VariableExpression) -> VariableExpression {
        VariableExpression::Or(Box::new(self.clone()), Box::new(rhs.clone()))
    }

    /// Comple the [`VariableExpression`] into a [`CompiledExpression`] bound to the supplied
    /// metadata so that all variable references are resolved.
    pub(crate) fn compile(
        &self,
        metadata: &DatasetMetadata,
    ) -> LadduResult<CompiledVariableExpression> {
        let mut compiled = compile_expression(self.clone());
        compiled.bind(metadata)?;
        Ok(compiled)
    }
}
impl Display for VariableExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableExpression::Eq(var, val) => {
                write!(f, "({} == {})", var, val)
            }
            VariableExpression::Lt(var, val) => {
                write!(f, "({} < {})", var, val)
            }
            VariableExpression::Gt(var, val) => {
                write!(f, "({} > {})", var, val)
            }
            VariableExpression::And(lhs, rhs) => {
                write!(f, "({} & {})", lhs, rhs)
            }
            VariableExpression::Or(lhs, rhs) => {
                write!(f, "({} | {})", lhs, rhs)
            }
            VariableExpression::Not(inner) => {
                write!(f, "!({})", inner)
            }
        }
    }
}

/// A method which negates the given expression.
pub fn not(expr: &VariableExpression) -> VariableExpression {
    VariableExpression::Not(Box::new(expr.clone()))
}

#[rustfmt::skip]
impl_op_ex!(& |lhs: &VariableExpression, rhs: &VariableExpression| -> VariableExpression{ lhs.and(rhs) });
#[rustfmt::skip]
impl_op_ex!(| |lhs: &VariableExpression, rhs: &VariableExpression| -> VariableExpression{ lhs.or(rhs) });
#[rustfmt::skip]
impl_op_ex!(! |exp: &VariableExpression| -> VariableExpression{ not(exp) });

#[derive(Debug)]
enum Opcode {
    PushEq(usize, f64),
    PushLt(usize, f64),
    PushGt(usize, f64),
    And,
    Or,
    Not,
}

pub(crate) struct CompiledVariableExpression {
    bytecode: Vec<Opcode>,
    variables: Vec<Box<dyn Variable>>,
}

impl CompiledVariableExpression {
    pub fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        for variable in &mut self.variables {
            variable.bind(metadata)?;
        }
        Ok(())
    }

    /// Evaluate the [`CompiledExpression`] on a given named event view.
    pub fn evaluate(&self, event: &Event<'_>) -> bool {
        let mut stack = Vec::with_capacity(self.bytecode.len());

        for op in &self.bytecode {
            match op {
                Opcode::PushEq(i, val) => stack.push(self.variables[*i].value(event) == *val),
                Opcode::PushLt(i, val) => stack.push(self.variables[*i].value(event) < *val),
                Opcode::PushGt(i, val) => stack.push(self.variables[*i].value(event) > *val),
                Opcode::Not => {
                    let a = stack.pop().unwrap();
                    stack.push(!a);
                }
                Opcode::And => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(a && b);
                }
                Opcode::Or => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(a || b);
                }
            }
        }

        stack.pop().unwrap()
    }
}

pub(crate) fn compile_expression(expr: VariableExpression) -> CompiledVariableExpression {
    let mut bytecode = Vec::new();
    let mut variables: Vec<Box<dyn Variable>> = Vec::new();

    fn compile(
        expr: VariableExpression,
        bytecode: &mut Vec<Opcode>,
        variables: &mut Vec<Box<dyn Variable>>,
    ) {
        match expr {
            VariableExpression::Eq(var, val) => {
                variables.push(var);
                bytecode.push(Opcode::PushEq(variables.len() - 1, val));
            }
            VariableExpression::Lt(var, val) => {
                variables.push(var);
                bytecode.push(Opcode::PushLt(variables.len() - 1, val));
            }
            VariableExpression::Gt(var, val) => {
                variables.push(var);
                bytecode.push(Opcode::PushGt(variables.len() - 1, val));
            }
            VariableExpression::And(lhs, rhs) => {
                compile(*lhs, bytecode, variables);
                compile(*rhs, bytecode, variables);
                bytecode.push(Opcode::And);
            }
            VariableExpression::Or(lhs, rhs) => {
                compile(*lhs, bytecode, variables);
                compile(*rhs, bytecode, variables);
                bytecode.push(Opcode::Or);
            }
            VariableExpression::Not(inner) => {
                compile(*inner, bytecode, variables);
                bytecode.push(Opcode::Not);
            }
        }
    }

    compile(expr, &mut bytecode, &mut variables);

    CompiledVariableExpression {
        bytecode,
        variables,
    }
}
