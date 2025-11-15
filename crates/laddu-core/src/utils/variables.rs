use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
use crate::{
    data::{Dataset, DatasetMetadata, EventData},
    utils::{
        enums::{Channel, Frame},
        vectors::{Vec3, Vec4},
    },
    LadduError, LadduResult,
};

use auto_ops::impl_op_ex;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

/// Standard methods for extracting some value out of an [`EventData`].
#[typetag::serde(tag = "type")]
pub trait Variable: DynClone + Send + Sync + Debug + Display {
    /// Bind the variable to dataset metadata so that any referenced names can be resolved to
    /// concrete indices. Implementations that do not require metadata may keep the default
    /// no-op.
    fn bind(&mut self, _metadata: &DatasetMetadata) -> LadduResult<()> {
        Ok(())
    }

    /// This method takes an [`EventData`] and extracts a single value (like the mass of a particle).
    fn value(&self, event: &EventData) -> f64;

    /// This method distributes the [`Variable::value`] method over each [`EventData`] in a
    /// [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Variable::value_on`] instead.
    fn value_on_local(&self, dataset: &Dataset) -> LadduResult<Vec<f64>> {
        let mut variable = dyn_clone::clone_box(self);
        variable.bind(dataset.metadata())?;
        #[cfg(feature = "rayon")]
        let local_values: Vec<f64> = dataset
            .events
            .par_iter()
            .map(|e| variable.value(e))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let local_values: Vec<f64> = dataset.events.iter().map(|e| variable.value(e)).collect();
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
    pub(crate) fn compile(&self, metadata: &DatasetMetadata) -> LadduResult<CompiledExpression> {
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

pub(crate) struct CompiledExpression {
    bytecode: Vec<Opcode>,
    variables: Vec<Box<dyn Variable>>,
}

impl CompiledExpression {
    pub fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        for variable in &mut self.variables {
            variable.bind(metadata)?;
        }
        Ok(())
    }

    /// Evaluate the [`CompiledExpression`] on a given [`EventData`].
    pub fn evaluate(&self, event: &EventData) -> bool {
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

pub(crate) fn compile_expression(expr: VariableExpression) -> CompiledExpression {
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

    CompiledExpression {
        bytecode,
        variables,
    }
}

fn names_to_string(names: &[String]) -> String {
    names.join(", ")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct P4Selection {
    names: Vec<String>,
    #[serde(skip, default)]
    indices: Vec<usize>,
}

impl P4Selection {
    fn new_many<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            names: names.into_iter().map(Into::into).collect(),
            indices: Vec::new(),
        }
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let mut resolved = Vec::with_capacity(self.names.len());
        for name in &self.names {
            let idx = metadata
                .p4_index(name)
                .ok_or_else(|| LadduError::UnknownName {
                    category: "p4",
                    name: name.clone(),
                })?;
            resolved.push(idx);
        }
        self.indices = resolved;
        Ok(())
    }

    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn names(&self) -> &[String] {
        &self.names
    }
}

/// A vertex participating in a [`Topology`].
///
/// Each vertex is represented as the sum of one or more four-momenta identified by their
/// metadata names. This wrapper intentionally mimics [`P4Selection`] so that a topology can be
/// bound once and reused across several kinematic variables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyVertex {
    selection: P4Selection,
}

impl TopologyVertex {
    fn from_names(names: Vec<String>) -> Self {
        Self {
            selection: P4Selection::new_many(names),
        }
    }

    /// Returns the metadata names contributing to this vertex.
    pub fn names(&self) -> &[String] {
        self.selection.names()
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.selection.bind(metadata)
    }

    fn momentum(&self, event: &EventData) -> Vec4 {
        event.get_p4_sum(self.selection.indices())
    }
}

/// Helper trait to convert common particle specifications into [`TopologyVertex`] instances.
pub trait IntoTopologyVertex {
    /// Convert the input into a [`TopologyVertex`].
    fn into_vertex(self) -> TopologyVertex;
}

impl IntoTopologyVertex for TopologyVertex {
    fn into_vertex(self) -> TopologyVertex {
        self
    }
}

impl<'a> IntoTopologyVertex for &'a TopologyVertex {
    fn into_vertex(self) -> TopologyVertex {
        self.clone()
    }
}

impl IntoTopologyVertex for String {
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(vec![self])
    }
}

impl IntoTopologyVertex for &String {
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(vec![self.clone()])
    }
}

impl IntoTopologyVertex for &str {
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(vec![self.to_string()])
    }
}

impl<S> IntoTopologyVertex for Vec<S>
where
    S: Into<String>,
{
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(self.into_iter().map(Into::into).collect())
    }
}

impl<'a, S> IntoTopologyVertex for &'a [S]
where
    S: Clone + Into<String>,
{
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(self.iter().cloned().map(Into::into).collect())
    }
}

impl<S, const N: usize> IntoTopologyVertex for [S; N]
where
    S: Into<String>,
{
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(self.into_iter().map(Into::into).collect())
    }
}

impl<'a, S, const N: usize> IntoTopologyVertex for &'a [S; N]
where
    S: Clone + Into<String>,
{
    fn into_vertex(self) -> TopologyVertex {
        TopologyVertex::from_names(self.iter().cloned().map(Into::into).collect())
    }
}

/// A reusable 2-to-2 reaction description shared by several kinematic variables.
///
/// A topology records the four canonical vertices $`k_1 + k_2 \to k_3 + k_4`$ (matching the
/// traditional $`p_1 \ldots p_4`$ notation) in the center-of-momentum frame. When one vertex is
/// omitted, it is reconstructed by enforcing four-momentum conservation, which is unambiguous in
/// that frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Topology {
    /// All four vertices are explicitly provided.
    Full {
        /// Incoming beam (`k1`, often labeled `p1`).
        k1: TopologyVertex,
        /// Target or recoil stand-in (`k2`, `p2`).
        k2: TopologyVertex,
        /// Produced resonance (`k3`, `p3`).
        k3: TopologyVertex,
        /// Recoil particle (`k4`, `p4`).
        k4: TopologyVertex,
    },
    /// The beam (`k1`) is reconstructed.
    MissingK1 {
        /// Target or recoil stand-in (`k2`).
        k2: TopologyVertex,
        /// Produced resonance (`k3`).
        k3: TopologyVertex,
        /// Recoil particle (`k4`).
        k4: TopologyVertex,
    },
    /// The target (`k2`) is reconstructed.
    MissingK2 {
        /// Incoming beam (`k1`).
        k1: TopologyVertex,
        /// Produced resonance (`k3`).
        k3: TopologyVertex,
        /// Recoil particle (`k4`).
        k4: TopologyVertex,
    },
    /// The resonance (`k3`) is reconstructed.
    MissingK3 {
        /// Incoming beam (`k1`).
        k1: TopologyVertex,
        /// Target or recoil stand-in (`k2`).
        k2: TopologyVertex,
        /// Recoil particle (`k4`).
        k4: TopologyVertex,
    },
    /// The recoil (`k4`) is reconstructed.
    MissingK4 {
        /// Incoming beam (`k1`).
        k1: TopologyVertex,
        /// Target or recoil stand-in (`k2`).
        k2: TopologyVertex,
        /// Produced resonance (`k3`).
        k3: TopologyVertex,
    },
}

impl Topology {
    /// Construct a topology with all four vertices explicitly defined.
    pub fn new<K1, K2, K3, K4>(k1: K1, k2: K2, k3: K3, k4: K4) -> Self
    where
        K1: IntoTopologyVertex,
        K2: IntoTopologyVertex,
        K3: IntoTopologyVertex,
        K4: IntoTopologyVertex,
    {
        Self::Full {
            k1: k1.into_vertex(),
            k2: k2.into_vertex(),
            k3: k3.into_vertex(),
            k4: k4.into_vertex(),
        }
    }

    /// Construct a topology when the beam (`k1`) is omitted.
    pub fn missing_k1<K2, K3, K4>(k2: K2, k3: K3, k4: K4) -> Self
    where
        K2: IntoTopologyVertex,
        K3: IntoTopologyVertex,
        K4: IntoTopologyVertex,
    {
        Self::MissingK1 {
            k2: k2.into_vertex(),
            k3: k3.into_vertex(),
            k4: k4.into_vertex(),
        }
    }

    /// Construct a topology when the target (`k2`) is omitted.
    pub fn missing_k2<K1, K3, K4>(k1: K1, k3: K3, k4: K4) -> Self
    where
        K1: IntoTopologyVertex,
        K3: IntoTopologyVertex,
        K4: IntoTopologyVertex,
    {
        Self::MissingK2 {
            k1: k1.into_vertex(),
            k3: k3.into_vertex(),
            k4: k4.into_vertex(),
        }
    }

    /// Construct a topology when the resonance (`k3`) is omitted.
    pub fn missing_k3<K1, K2, K4>(k1: K1, k2: K2, k4: K4) -> Self
    where
        K1: IntoTopologyVertex,
        K2: IntoTopologyVertex,
        K4: IntoTopologyVertex,
    {
        Self::MissingK3 {
            k1: k1.into_vertex(),
            k2: k2.into_vertex(),
            k4: k4.into_vertex(),
        }
    }

    /// Construct a topology when the recoil (`k4`) is omitted.
    pub fn missing_k4<K1, K2, K3>(k1: K1, k2: K2, k3: K3) -> Self
    where
        K1: IntoTopologyVertex,
        K2: IntoTopologyVertex,
        K3: IntoTopologyVertex,
    {
        Self::MissingK4 {
            k1: k1.into_vertex(),
            k2: k2.into_vertex(),
            k3: k3.into_vertex(),
        }
    }

    /// Bind every vertex to dataset metadata so the particle names resolve to indices.
    pub fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        match self {
            Topology::Full { k1, k2, k3, k4 } => {
                k1.bind(metadata)?;
                k2.bind(metadata)?;
                k3.bind(metadata)?;
                k4.bind(metadata)?;
            }
            Topology::MissingK1 { k2, k3, k4 } => {
                k2.bind(metadata)?;
                k3.bind(metadata)?;
                k4.bind(metadata)?;
            }
            Topology::MissingK2 { k1, k3, k4 } => {
                k1.bind(metadata)?;
                k3.bind(metadata)?;
                k4.bind(metadata)?;
            }
            Topology::MissingK3 { k1, k2, k4 } => {
                k1.bind(metadata)?;
                k2.bind(metadata)?;
                k4.bind(metadata)?;
            }
            Topology::MissingK4 { k1, k2, k3 } => {
                k1.bind(metadata)?;
                k2.bind(metadata)?;
                k3.bind(metadata)?;
            }
        }
        Ok(())
    }

    /// Convenience helper returning the beam four-momentum (`k1`).
    pub fn k1(&self, event: &EventData) -> Vec4 {
        match self {
            Topology::Full { k1, .. }
            | Topology::MissingK2 { k1, .. }
            | Topology::MissingK3 { k1, .. }
            | Topology::MissingK4 { k1, .. } => k1.momentum(event),
            Topology::MissingK1 { k2, k3, k4 } => {
                k3.momentum(event) + k4.momentum(event) - k2.momentum(event)
            }
        }
    }

    /// Convenience helper returning the target four-momentum (`k2`).
    pub fn k2(&self, event: &EventData) -> Vec4 {
        match self {
            Topology::Full { k2, .. }
            | Topology::MissingK1 { k2, .. }
            | Topology::MissingK3 { k2, .. }
            | Topology::MissingK4 { k2, .. } => k2.momentum(event),
            Topology::MissingK2 { k1, k3, k4 } => {
                k3.momentum(event) + k4.momentum(event) - k1.momentum(event)
            }
        }
    }

    /// Convenience helper returning the resonance four-momentum (`k3`).
    pub fn k3(&self, event: &EventData) -> Vec4 {
        match self {
            Topology::Full { k3, .. }
            | Topology::MissingK1 { k3, .. }
            | Topology::MissingK2 { k3, .. }
            | Topology::MissingK4 { k3, .. } => k3.momentum(event),
            Topology::MissingK3 { k1, k2, k4 } => {
                k1.momentum(event) + k2.momentum(event) - k4.momentum(event)
            }
        }
    }

    /// Convenience helper returning the recoil four-momentum (`k4`).
    pub fn k4(&self, event: &EventData) -> Vec4 {
        match self {
            Topology::Full { k4, .. }
            | Topology::MissingK1 { k4, .. }
            | Topology::MissingK2 { k4, .. }
            | Topology::MissingK3 { k4, .. } => k4.momentum(event),
            Topology::MissingK4 { k1, k2, k3 } => {
                k1.momentum(event) + k2.momentum(event) - k3.momentum(event)
            }
        }
    }

    /// Returns the resolved names for `k1` if it was explicitly provided.
    pub fn k1_names(&self) -> Option<&[String]> {
        match self {
            Topology::Full { k1, .. }
            | Topology::MissingK2 { k1, .. }
            | Topology::MissingK3 { k1, .. }
            | Topology::MissingK4 { k1, .. } => Some(k1.names()),
            Topology::MissingK1 { .. } => None,
        }
    }

    /// Returns the resolved names for `k2` if it was explicitly provided.
    pub fn k2_names(&self) -> Option<&[String]> {
        match self {
            Topology::Full { k2, .. }
            | Topology::MissingK1 { k2, .. }
            | Topology::MissingK3 { k2, .. }
            | Topology::MissingK4 { k2, .. } => Some(k2.names()),
            Topology::MissingK2 { .. } => None,
        }
    }

    /// Returns the resolved names for `k3` if it was explicitly provided.
    pub fn k3_names(&self) -> Option<&[String]> {
        match self {
            Topology::Full { k3, .. }
            | Topology::MissingK1 { k3, .. }
            | Topology::MissingK2 { k3, .. }
            | Topology::MissingK4 { k3, .. } => Some(k3.names()),
            Topology::MissingK3 { .. } => None,
        }
    }

    /// Returns the resolved names for `k4` if it was explicitly provided.
    pub fn k4_names(&self) -> Option<&[String]> {
        match self {
            Topology::Full { k4, .. }
            | Topology::MissingK1 { k4, .. }
            | Topology::MissingK2 { k4, .. }
            | Topology::MissingK3 { k4, .. } => Some(k4.names()),
            Topology::MissingK4 { .. } => None,
        }
    }
}

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Topology(k1=[{}], k2=[{}], k3=[{}], k4=[{}])",
            format_topology_names(self.k1_names()),
            format_topology_names(self.k2_names()),
            format_topology_names(self.k3_names()),
            format_topology_names(self.k4_names())
        )
    }
}

fn format_topology_names(names: Option<&[String]>) -> String {
    match names {
        Some(names) if !names.is_empty() => names_to_string(names),
        Some(_) => String::new(),
        None => "<reconstructed>".to_string(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AuxSelection {
    name: String,
    #[serde(skip, default)]
    index: Option<usize>,
}

impl AuxSelection {
    fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            index: None,
        }
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let idx = metadata
            .aux_index(&self.name)
            .ok_or_else(|| LadduError::UnknownName {
                category: "aux",
                name: self.name.clone(),
            })?;
        self.index = Some(idx);
        Ok(())
    }

    fn index(&self) -> usize {
        self.index.expect("AuxSelection must be bound before use")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A struct for obtaining the mass of a particle by indexing the four-momenta of an event, adding
/// together multiple four-momenta if more than one entry is given.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mass {
    constituents: TopologyVertex,
}
impl Mass {
    /// Create a new [`Mass`] from the sum of the four-momenta identified by `constituents` in the
    /// [`EventData`]'s `p4s` field.
    pub fn new<C>(constituents: C) -> Self
    where
        C: IntoTopologyVertex,
    {
        Self {
            constituents: constituents.into_vertex(),
        }
    }
}
impl Display for Mass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mass(constituents=[{}])",
            names_to_string(self.constituents.names())
        )
    }
}
#[typetag::serde]
impl Variable for Mass {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.constituents.bind(metadata)
    }

    fn value(&self, event: &EventData) -> f64 {
        self.constituents.momentum(event).m()
    }
}

/// A struct for obtaining the $`\cos\theta`$ (cosine of the polar angle) of a decay product in
/// a given reference frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    topology: Topology,
    daughter: TopologyVertex,
    frame: Frame,
}
impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CosTheta(topology={}, daughter=[{}], frame={})",
            self.topology,
            names_to_string(self.daughter.names()),
            self.frame
        )
    }
}
impl CosTheta {
    /// Construct the angle given a [`Topology`] describing the production kinematics along with a
    /// decay daughter of the `k3` resonance. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<D>(topology: Topology, daughter: D, frame: Frame) -> Self
    where
        D: IntoTopologyVertex,
    {
        Self {
            topology,
            daughter: daughter.into_vertex(),
            frame,
        }
    }
}
#[typetag::serde]
impl Variable for CosTheta {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.topology.bind(metadata)?;
        self.daughter.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        let beam = self.topology.k1(event);
        let recoil = self.topology.k4(event);
        let daughter = self.daughter.momentum(event);
        let resonance = self.topology.k3(event);
        let daughter_res = daughter.boost(&-resonance.beta());
        match self.frame {
            Frame::Helicity => {
                let recoil_res = recoil.boost(&-resonance.beta());
                let z = -recoil_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vec3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.costheta()
            }
            Frame::GottfriedJackson => {
                let beam_res = beam.boost(&-resonance.beta());
                let z = beam_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vec3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.costheta()
            }
        }
    }
}

/// A struct for obtaining the $`\phi`$ angle (azimuthal angle) of a decay product in a given
/// reference frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phi {
    topology: Topology,
    daughter: TopologyVertex,
    frame: Frame,
}
impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phi(topology={}, daughter=[{}], frame={})",
            self.topology,
            names_to_string(self.daughter.names()),
            self.frame
        )
    }
}
impl Phi {
    /// Construct the angle given a [`Topology`] describing the production kinematics along with a
    /// daughter of the resonance defined by `k3`. See [`Frame`] for options regarding the
    /// reference frame.
    pub fn new<D>(topology: Topology, daughter: D, frame: Frame) -> Self
    where
        D: IntoTopologyVertex,
    {
        Self {
            topology,
            daughter: daughter.into_vertex(),
            frame,
        }
    }
}
#[typetag::serde]
impl Variable for Phi {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.topology.bind(metadata)?;
        self.daughter.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        let beam = self.topology.k1(event);
        let recoil = self.topology.k4(event);
        let daughter = self.daughter.momentum(event);
        let resonance = self.topology.k3(event);
        let daughter_res = daughter.boost(&-resonance.beta());
        match self.frame {
            Frame::Helicity => {
                let recoil_res = recoil.boost(&-resonance.beta());
                let z = -recoil_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vec3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.phi()
            }
            Frame::GottfriedJackson => {
                let beam_res = beam.boost(&-resonance.beta());
                let z = beam_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vec3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.phi()
            }
        }
    }
}

/// A struct for obtaining both spherical angles at the same time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Angles {
    /// See [`CosTheta`].
    pub costheta: CosTheta,
    /// See [`Phi`].
    pub phi: Phi,
}

impl Display for Angles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Angles(topology={}, daughter=[{}], frame={})",
            self.costheta.topology,
            names_to_string(self.costheta.daughter.names()),
            self.costheta.frame
        )
    }
}
impl Angles {
    /// Construct the angles given a [`Topology`] along with the daughter selection.
    /// See [`Frame`] for options regarding the reference frame.
    pub fn new<D>(topology: Topology, daughter: D, frame: Frame) -> Self
    where
        D: IntoTopologyVertex,
    {
        let daughter_vertex = daughter.into_vertex();
        let costheta = CosTheta::new(topology.clone(), daughter_vertex.clone(), frame);
        let phi = Phi::new(topology, daughter_vertex, frame);
        Self { costheta, phi }
    }
}

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolAngle {
    topology: Topology,
    angle_aux: AuxSelection,
}
impl Display for PolAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolAngle(topology={}, angle_aux={})",
            self.topology,
            self.angle_aux.name(),
        )
    }
}
impl PolAngle {
    /// Constructs the polarization angle given a [`Topology`] describing the production plane and
    /// the auxiliary column storing the precomputed angle.
    pub fn new<A>(topology: Topology, angle_aux: A) -> Self
    where
        A: Into<String>,
    {
        Self {
            topology,
            angle_aux: AuxSelection::new(angle_aux.into()),
        }
    }
}
#[typetag::serde]
impl Variable for PolAngle {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.topology.bind(metadata)?;
        self.angle_aux.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        event.aux[self.angle_aux.index()]
    }
}

/// A struct defining the polarization magnitude for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolMagnitude {
    magnitude_aux: AuxSelection,
}
impl Display for PolMagnitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolMagnitude(magnitude_aux={})",
            self.magnitude_aux.name(),
        )
    }
}
impl PolMagnitude {
    /// Constructs the polarization magnitude given the named auxiliary column containing the
    /// magnitude value.
    pub fn new<S: Into<String>>(magnitude_aux: S) -> Self {
        Self {
            magnitude_aux: AuxSelection::new(magnitude_aux.into()),
        }
    }
}
#[typetag::serde]
impl Variable for PolMagnitude {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.magnitude_aux.bind(metadata)
    }

    fn value(&self, event: &EventData) -> f64 {
        event.aux[self.magnitude_aux.index()]
    }
}

/// A struct for obtaining both the polarization angle and magnitude at the same time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polarization {
    /// See [`PolMagnitude`].
    pub pol_magnitude: PolMagnitude,
    /// See [`PolAngle`].
    pub pol_angle: PolAngle,
}
impl Display for Polarization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Polarization(topology={}, magnitude_aux={}, angle_aux={})",
            self.pol_angle.topology,
            self.pol_magnitude.magnitude_aux.name(),
            self.pol_angle.angle_aux.name(),
        )
    }
}
impl Polarization {
    /// Constructs the polarization angle and magnitude given a [`Topology`] and distinct
    /// auxiliary columns for magnitude and angle.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude_aux` and `angle_aux` refer to the same auxiliary column name.
    pub fn new<M, A>(topology: Topology, magnitude_aux: M, angle_aux: A) -> Self
    where
        M: Into<String>,
        A: Into<String>,
    {
        let magnitude_aux = magnitude_aux.into();
        let angle_aux = angle_aux.into();
        assert!(
            magnitude_aux != angle_aux,
            "Polarization magnitude and angle must reference distinct auxiliary columns"
        );
        Self {
            pol_magnitude: PolMagnitude::new(magnitude_aux),
            pol_angle: PolAngle::new(topology, angle_aux),
        }
    }
}

/// A struct used to calculate Mandelstam variables ($`s`$, $`t`$, or $`u`$).
///
/// By convention, the metric is chosen to be $`(+---)`$ and the variables are defined as follows
/// (ignoring factors of $`c`$):
///
/// $`s = (p_1 + p_2)^2 = (p_3 + p_4)^2`$
///
/// $`t = (p_1 - p_3)^2 = (p_4 - p_2)^2`$
///
/// $`u = (p_1 - p_4)^2 = (p_3 - p_2)^2`$
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mandelstam {
    topology: Topology,
    channel: Channel,
}
impl Display for Mandelstam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mandelstam(topology={}, channel={})",
            self.topology, self.channel,
        )
    }
}
impl Mandelstam {
    /// Constructs the Mandelstam variable for the given `channel` using the supplied [`Topology`].
    pub fn new(topology: Topology, channel: Channel) -> Self {
        Self { topology, channel }
    }
}

#[typetag::serde]
impl Variable for Mandelstam {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.topology.bind(metadata)
    }

    fn value(&self, event: &EventData) -> f64 {
        match self.channel {
            Channel::S => {
                let k1 = self.topology.k1(event);
                let k2 = self.topology.k2(event);
                (k1 + k2).mag2()
            }
            Channel::T => {
                let k1 = self.topology.k1(event);
                let k3 = self.topology.k3(event);
                (k1 - k3).mag2()
            }
            Channel::U => {
                let k1 = self.topology.k1(event);
                let k4 = self.topology.k4(event);
                (k1 - k4).mag2()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{test_dataset, test_event};
    use approx::assert_relative_eq;

    fn topology_test_metadata() -> DatasetMetadata {
        DatasetMetadata::new(
            vec!["beam", "target", "resonance", "recoil"],
            Vec::<String>::new(),
        )
        .expect("topology metadata should be valid")
    }

    fn topology_test_event() -> EventData {
        let p1 = Vec4::new(0.0, 0.0, 3.0, 3.5);
        let p2 = Vec4::new(0.0, 0.0, -3.0, 3.5);
        let p3 = Vec4::new(0.5, -0.25, 1.0, 1.9);
        let p4 = p1 + p2 - p3;
        EventData {
            p4s: vec![p1, p2, p3, p4],
            aux: vec![],
            weight: 1.0,
        }
    }

    fn reaction_topology() -> Topology {
        Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton")
    }

    #[test]
    fn test_topology_accepts_varied_inputs() {
        let topo = Topology::new(
            "particle1",
            ["particle2a", "particle2b"],
            &["particle3"],
            "particle4".to_string(),
        );
        assert_eq!(
            topo.k1_names()
                .unwrap()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["particle1"]
        );
        assert_eq!(
            topo.k2_names()
                .unwrap()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["particle2a", "particle2b"]
        );
        let missing = Topology::missing_k2("particle1", vec!["particle3"], "particle4");
        assert!(missing.k2_names().is_none());
        assert!(missing.to_string().contains("<reconstructed>"));
    }

    #[test]
    fn test_topology_reconstructs_missing_vertices() {
        let metadata = topology_test_metadata();
        let event = topology_test_event();

        let mut full = Topology::new("beam", "target", "resonance", "recoil");
        full.bind(&metadata).unwrap();
        assert_relative_eq!(full.k3(&event), event.p4s[2], epsilon = 1e-12);

        let mut missing_k1 = Topology::missing_k1("target", "resonance", "recoil");
        missing_k1.bind(&metadata).unwrap();
        assert!(missing_k1.k1_names().is_none());
        assert_relative_eq!(missing_k1.k1(&event), event.p4s[0], epsilon = 1e-12);

        let mut missing_k2 = Topology::missing_k2("beam", "resonance", "recoil");
        missing_k2.bind(&metadata).unwrap();
        assert!(missing_k2.k2_names().is_none());
        assert_relative_eq!(missing_k2.k2(&event), event.p4s[1], epsilon = 1e-12);

        let mut missing_k3 = Topology::missing_k3("beam", "target", "recoil");
        missing_k3.bind(&metadata).unwrap();
        assert!(missing_k3.k3_names().is_none());
        assert_relative_eq!(missing_k3.k3(&event), event.p4s[2], epsilon = 1e-12);

        let mut missing_k4 = Topology::missing_k4("beam", "target", "resonance");
        missing_k4.bind(&metadata).unwrap();
        assert!(missing_k4.k4_names().is_none());
        assert_relative_eq!(missing_k4.k4(&event), event.p4s[3], epsilon = 1e-12);
    }

    #[test]
    fn test_mass_single_particle() {
        let dataset = test_dataset();
        let mut mass = Mass::new("proton");
        mass.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(mass.value(&dataset[0]), 1.007);
    }

    #[test]
    fn test_mass_multiple_particles() {
        let dataset = test_dataset();
        let mut mass = Mass::new(["kshort1", "kshort2"]);
        mass.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(mass.value(&dataset[0]), 1.3743786309153077);
    }

    #[test]
    fn test_mass_display() {
        let mass = Mass::new(["kshort1", "kshort2"]);
        assert_eq!(mass.to_string(), "Mass(constituents=[kshort1, kshort2])");
    }

    #[test]
    fn test_costheta_helicity() {
        let dataset = test_dataset();
        let mut costheta = CosTheta::new(reaction_topology(), "kshort1", Frame::Helicity);
        costheta.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(costheta.value(&dataset[0]), -0.4611175068834238);
    }

    #[test]
    fn test_costheta_display() {
        let costheta = CosTheta::new(reaction_topology(), "kshort1", Frame::Helicity);
        assert_eq!(
            costheta.to_string(),
            "CosTheta(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), daughter=[kshort1], frame=Helicity)"
        );
    }

    #[test]
    fn test_phi_helicity() {
        let dataset = test_dataset();
        let mut phi = Phi::new(reaction_topology(), "kshort1", Frame::Helicity);
        phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(phi.value(&dataset[0]), -2.657462587335066);
    }

    #[test]
    fn test_phi_display() {
        let phi = Phi::new(reaction_topology(), "kshort1", Frame::Helicity);
        assert_eq!(
            phi.to_string(),
            "Phi(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), daughter=[kshort1], frame=Helicity)"
        );
    }

    #[test]
    fn test_costheta_gottfried_jackson() {
        let dataset = test_dataset();
        let mut costheta = CosTheta::new(reaction_topology(), "kshort1", Frame::GottfriedJackson);
        costheta.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(costheta.value(&dataset[0]), 0.09198832278031577);
    }

    #[test]
    fn test_phi_gottfried_jackson() {
        let dataset = test_dataset();
        let mut phi = Phi::new(reaction_topology(), "kshort1", Frame::GottfriedJackson);
        phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(phi.value(&dataset[0]), -2.713913199133907);
    }

    #[test]
    fn test_angles() {
        let dataset = test_dataset();
        let mut angles = Angles::new(reaction_topology(), "kshort1", Frame::Helicity);
        angles.costheta.bind(dataset.metadata()).unwrap();
        angles.phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(angles.costheta.value(&dataset[0]), -0.4611175068834238);
        assert_relative_eq!(angles.phi.value(&dataset[0]), -2.657462587335066);
    }

    #[test]
    fn test_angles_display() {
        let angles = Angles::new(reaction_topology(), "kshort1", Frame::Helicity);
        assert_eq!(
            angles.to_string(),
            "Angles(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), daughter=[kshort1], frame=Helicity)"
        );
    }

    #[test]
    fn test_pol_angle() {
        let dataset = test_dataset();
        let mut pol_angle = PolAngle::new(reaction_topology(), "pol_angle");
        pol_angle.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(pol_angle.value(&dataset[0]), 1.93592989);
    }

    #[test]
    fn test_pol_angle_display() {
        let pol_angle = PolAngle::new(reaction_topology(), "pol_angle");
        assert_eq!(
            pol_angle.to_string(),
            "PolAngle(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), angle_aux=pol_angle)"
        );
    }

    #[test]
    fn test_pol_magnitude() {
        let dataset = test_dataset();
        let mut pol_magnitude = PolMagnitude::new("pol_magnitude");
        pol_magnitude.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(pol_magnitude.value(&dataset[0]), 0.38562805);
    }

    #[test]
    fn test_pol_magnitude_display() {
        let pol_magnitude = PolMagnitude::new("pol_magnitude");
        assert_eq!(
            pol_magnitude.to_string(),
            "PolMagnitude(magnitude_aux=pol_magnitude)"
        );
    }

    #[test]
    fn test_polarization() {
        let dataset = test_dataset();
        let mut polarization = Polarization::new(reaction_topology(), "pol_magnitude", "pol_angle");
        polarization.pol_angle.bind(dataset.metadata()).unwrap();
        polarization.pol_magnitude.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(polarization.pol_angle.value(&dataset[0]), 1.93592989);
        assert_relative_eq!(polarization.pol_magnitude.value(&dataset[0]), 0.38562805);
    }

    #[test]
    fn test_polarization_display() {
        let polarization = Polarization::new(reaction_topology(), "pol_magnitude", "pol_angle");
        assert_eq!(
            polarization.to_string(),
            "Polarization(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), magnitude_aux=pol_magnitude, angle_aux=pol_angle)"
        );
    }

    #[test]
    fn test_mandelstam() {
        let dataset = test_dataset();
        let metadata = dataset.metadata();
        let mut s = Mandelstam::new(reaction_topology(), Channel::S);
        let mut t = Mandelstam::new(reaction_topology(), Channel::T);
        let mut u = Mandelstam::new(reaction_topology(), Channel::U);
        for variable in [&mut s, &mut t, &mut u] {
            variable.bind(metadata).unwrap();
        }
        let event = &dataset[0];
        assert_relative_eq!(s.value(event), 18.504011052120063);
        assert_relative_eq!(t.value(event), -0.19222859969898076);
        assert_relative_eq!(u.value(event), -14.404198931464428);
        let mut direct_topology = reaction_topology();
        direct_topology.bind(metadata).unwrap();
        let k2 = direct_topology.k2(event);
        let k3 = direct_topology.k3(event);
        let k4 = direct_topology.k4(event);
        assert_relative_eq!(s.value(event), (k3 + k4).mag2());
        assert_relative_eq!(t.value(event), (k2 - k4).mag2());
        assert_relative_eq!(u.value(event), (k3 - k2).mag2());
        let m2_beam = test_event().get_p4_sum([0]).m2();
        let m2_recoil = test_event().get_p4_sum([1]).m2();
        let m2_res = test_event().get_p4_sum([2, 3]).m2();
        assert_relative_eq!(
            s.value(event) + t.value(event) + u.value(event) - m2_beam - m2_recoil - m2_res,
            1.00,
            epsilon = 1e-2
        );
        // Note: not very accurate, but considering the values in test_event only go to about 3
        // decimal places, this is probably okay
    }

    #[test]
    fn test_mandelstam_display() {
        let s = Mandelstam::new(reaction_topology(), Channel::S);
        assert_eq!(
            s.to_string(),
            "Mandelstam(topology=Topology(k1=[beam], k2=[<reconstructed>], k3=[kshort1, kshort2], k4=[proton]), channel=s)"
        );
    }

    #[test]
    fn test_variable_value_on() {
        let dataset = test_dataset();
        let mass = Mass::new(["kshort1", "kshort2"]);

        let values = mass.value_on(&dataset).unwrap();
        assert_eq!(values.len(), 1);
        assert_relative_eq!(values[0], 1.3743786309153077);
    }
}
