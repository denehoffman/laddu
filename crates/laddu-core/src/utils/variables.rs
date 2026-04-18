use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
use crate::{
    data::{Dataset, DatasetMetadata, EventData, NamedEventView},
    utils::{
        enums::{Channel, Frame},
        reaction::{Particle, Reaction},
        vectors::{Vec3, Vec4},
    },
    LadduError, LadduResult,
};

use auto_ops::impl_op_ex;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

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
    fn value(&self, event: &NamedEventView<'_>) -> f64;

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

    /// Evaluate the [`CompiledExpression`] on a given named event view.
    pub fn evaluate(&self, event: &NamedEventView<'_>) -> bool {
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

/// A reusable selection that may span one or more four-momentum names.
///
/// Instances are constructed from metadata-facing identifiers and later bound to
/// column indices so that variable evaluators can resolve aliases or grouped
/// particles efficiently.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P4Selection {
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

    pub(crate) fn with_indices<I, S>(names: I, indices: Vec<usize>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            names: names.into_iter().map(Into::into).collect(),
            indices,
        }
    }

    /// Returns the metadata names contributing to this selection.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub(crate) fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let mut resolved = Vec::with_capacity(self.names.len());
        for name in &self.names {
            metadata.append_indices_for_name(name, &mut resolved)?;
        }
        self.indices = resolved;
        Ok(())
    }

    /// The resolved column indices backing this selection.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub(crate) fn momentum(&self, event: &EventData) -> Vec4 {
        event.get_p4_sum(self.indices())
    }
}

/// Helper trait to convert common particle specifications into [`P4Selection`] instances.
pub trait IntoP4Selection {
    /// Convert the input into a [`P4Selection`].
    fn into_selection(self) -> P4Selection;
}

impl IntoP4Selection for P4Selection {
    fn into_selection(self) -> P4Selection {
        self
    }
}

impl IntoP4Selection for &P4Selection {
    fn into_selection(self) -> P4Selection {
        self.clone()
    }
}

impl IntoP4Selection for String {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self])
    }
}

impl IntoP4Selection for &String {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self.clone()])
    }
}

impl IntoP4Selection for &str {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self.to_string()])
    }
}

impl<S> IntoP4Selection for Vec<S>
where
    S: Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S> IntoP4Selection for &[S]
where
    S: Clone + Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.iter().cloned().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S, const N: usize> IntoP4Selection for [S; N]
where
    S: Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S, const N: usize> IntoP4Selection for &[S; N]
where
    S: Clone + Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.iter().cloned().map(Into::into).collect::<Vec<_>>())
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

/// Source for a mass variable.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum MassSource {
    Selection(P4Selection),
    Reaction {
        reaction: Box<Reaction>,
        particle: Particle,
    },
}

/// A struct for obtaining the invariant mass of a selected or reaction-defined particle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mass {
    source: MassSource,
}
impl Mass {
    /// Create a new [`Mass`] from the sum of the four-momenta identified by `constituents` in the
    /// [`EventData`]'s `p4s` field.
    pub fn new<C>(constituents: C) -> Self
    where
        C: IntoP4Selection,
    {
        Self {
            source: MassSource::Selection(constituents.into_selection()),
        }
    }

    /// Create a new [`Mass`] for a particle resolved through a [`Reaction`].
    pub fn from_reaction(reaction: Reaction, particle: Particle) -> Self {
        Self {
            source: MassSource::Reaction {
                reaction: Box::new(reaction),
                particle,
            },
        }
    }
}
impl Display for Mass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            MassSource::Selection(constituents) => {
                write!(
                    f,
                    "Mass(constituents=[{}])",
                    names_to_string(constituents.names())
                )
            }
            MassSource::Reaction { particle, .. } => write!(f, "Mass(particle={})", particle),
        }
    }
}
#[typetag::serde]
impl Variable for Mass {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        match &mut self.source {
            MassSource::Selection(constituents) => constituents.bind(metadata),
            MassSource::Reaction { .. } => Ok(()),
        }
    }
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        match &self.source {
            MassSource::Selection(constituents) => constituents
                .indices()
                .iter()
                .map(|index| event.p4_at(*index))
                .sum::<Vec4>()
                .m(),
            MassSource::Reaction { reaction, particle } => reaction
                .p4(event, particle)
                .unwrap_or_else(|err| panic!("failed to evaluate reaction mass: {err}"))
                .m(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AngleSource {
    reaction: Box<Reaction>,
    parent: Particle,
    daughter: Particle,
}

impl AngleSource {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        Ok(())
    }

    fn costheta(&self, event: &NamedEventView<'_>, frame: Frame) -> f64 {
        self.reaction
            .angles_value(event, &self.parent, &self.daughter, frame)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction costheta: {err}"))
            .costheta()
    }

    fn phi(&self, event: &NamedEventView<'_>, frame: Frame) -> f64 {
        self.reaction
            .angles_value(event, &self.parent, &self.daughter, frame)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction phi: {err}"))
            .phi()
    }
}

/// A struct for obtaining the $`\cos\theta`$ (cosine of the polar angle) of a decay product in
/// a given reference frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    source: AngleSource,
    frame: Frame,
}
impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CosTheta(parent={}, daughter={}, frame={})",
            self.source.parent, self.source.daughter, self.frame
        )
    }
}
impl CosTheta {
    /// Construct an angle for a reaction daughter in the specified parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: Particle,
        daughter: Particle,
        frame: Frame,
    ) -> Self {
        Self {
            source: AngleSource {
                reaction: Box::new(reaction),
                parent,
                daughter,
            },
            frame,
        }
    }
}

#[typetag::serde]
impl Variable for CosTheta {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        self.source.costheta(event, self.frame)
    }
}

/// A struct for obtaining the $`\phi`$ angle (azimuthal angle) of a decay product in a given
/// reference frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phi {
    source: AngleSource,
    frame: Frame,
}
impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phi(parent={}, daughter={}, frame={})",
            self.source.parent, self.source.daughter, self.frame
        )
    }
}
impl Phi {
    /// Construct an angle for a reaction daughter in the specified parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: Particle,
        daughter: Particle,
        frame: Frame,
    ) -> Self {
        Self {
            source: AngleSource {
                reaction: Box::new(reaction),
                parent,
                daughter,
            },
            frame,
        }
    }
}
#[typetag::serde]
impl Variable for Phi {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        self.source.phi(event, self.frame)
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

/// A pair of variables that define spherical decay angles.
pub trait AngleVariables: Debug + Display + Send + Sync {
    /// Return the variable used for `cos(theta)`.
    fn costheta_variable(&self) -> Box<dyn Variable>;

    /// Return the variable used for `phi`.
    fn phi_variable(&self) -> Box<dyn Variable>;
}

impl AngleVariables for Angles {
    fn costheta_variable(&self) -> Box<dyn Variable> {
        Box::new(self.costheta.clone())
    }

    fn phi_variable(&self) -> Box<dyn Variable> {
        Box::new(self.phi.clone())
    }
}

impl Display for Angles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Angles(parent={}, daughter={}, frame={})",
            self.costheta.source.parent, self.costheta.source.daughter, self.costheta.frame
        )
    }
}
impl Angles {
    /// Construct reaction-derived angle variables for a daughter in its parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: Particle,
        daughter: Particle,
        frame: Frame,
    ) -> Self {
        let costheta =
            CosTheta::from_reaction(reaction.clone(), parent.clone(), daughter.clone(), frame);
        let phi = Phi::from_reaction(reaction, parent, daughter, frame);
        Self { costheta, phi }
    }
}

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolAngle {
    reaction: Reaction,
    angle_aux: AuxSelection,
}
impl Display for PolAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolAngle(reaction={:?}, angle_aux={})",
            self.reaction.topology(),
            self.angle_aux.name(),
        )
    }
}
impl PolAngle {
    /// Constructs the polarization angle given a [`Reaction`] describing the production plane and
    /// the auxiliary column storing the precomputed angle.
    pub fn new<A>(reaction: Reaction, angle_aux: A) -> Self
    where
        A: Into<String>,
    {
        Self {
            reaction,
            angle_aux: AuxSelection::new(angle_aux.into()),
        }
    }
}
#[typetag::serde]
impl Variable for PolAngle {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        self.angle_aux.bind(metadata)?;
        Ok(())
    }
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        let resolved = self
            .reaction
            .resolve_two_to_two(event)
            .unwrap_or_else(|err| panic!("failed to evaluate polarization angle: {err}"));
        let beam = resolved.p1();
        let recoil = resolved.p4();
        let pol_angle = event.aux_at(self.angle_aux.index());
        let polarization = Vec3::new(pol_angle.cos(), pol_angle.sin(), 0.0);
        let y = beam.vec3().cross(&-recoil.vec3()).unit();
        let numerator = y.dot(&polarization);
        let denominator = beam.vec3().unit().dot(&polarization.cross(&y));
        f64::atan2(numerator, denominator)
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
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        event.aux_at(self.magnitude_aux.index())
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
            "Polarization(reaction={:?}, magnitude_aux={}, angle_aux={})",
            self.pol_angle.reaction.topology(),
            self.pol_magnitude.magnitude_aux.name(),
            self.pol_angle.angle_aux.name(),
        )
    }
}
impl Polarization {
    /// Constructs the polarization angle and magnitude given a [`Reaction`] and distinct
    /// auxiliary columns for magnitude and angle.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude_aux` and `angle_aux` refer to the same auxiliary column name.
    pub fn new<M, A>(reaction: Reaction, magnitude_aux: M, angle_aux: A) -> Self
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
            pol_angle: PolAngle::new(reaction, angle_aux),
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
    reaction: Reaction,
    channel: Channel,
}
impl Display for Mandelstam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mandelstam(channel={})", self.channel)
    }
}
impl Mandelstam {
    /// Constructs the Mandelstam variable for the given `channel` using the supplied [`Reaction`].
    pub fn new(reaction: Reaction, channel: Channel) -> Self {
        Self { reaction, channel }
    }
}

#[typetag::serde]
impl Variable for Mandelstam {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        Ok(())
    }
    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        let resolved = self
            .reaction
            .resolve_two_to_two(event)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction Mandelstam: {err}"));
        match self.channel {
            Channel::S => resolved.s(),
            Channel::T => resolved.t(),
            Channel::U => resolved.u(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_dataset;
    use approx::assert_relative_eq;

    fn reaction() -> (Reaction, Particle, Particle, Particle) {
        let beam = Particle::measured("beam", "beam");
        let target = Particle::missing("target");
        let kshort1 = Particle::measured("K_S1", "kshort1");
        let kshort2 = Particle::measured("K_S2", "kshort2");
        let kk = Particle::composite("KK", [&kshort1, &kshort2]).unwrap();
        let proton = Particle::measured("proton", "proton");
        (
            Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap(),
            kk,
            kshort1,
            kshort2,
        )
    }

    #[test]
    fn test_mass_single_particle() {
        let dataset = test_dataset();
        let mut mass = Mass::new("proton");
        mass.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(mass.value(&event), 1.007);
    }

    #[test]
    fn test_mass_multiple_particles() {
        let dataset = test_dataset();
        let mut mass = Mass::new(["kshort1", "kshort2"]);
        mass.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(mass.value(&event), 1.3743786309153077);
    }

    #[test]
    fn test_mass_display() {
        let mass = Mass::new(["kshort1", "kshort2"]);
        assert_eq!(mass.to_string(), "Mass(constituents=[kshort1, kshort2])");
    }

    #[test]
    fn test_costheta_helicity() {
        let dataset = test_dataset();
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let mut costheta = decay.costheta(&kshort1, Frame::Helicity).unwrap();
        costheta.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(costheta.value(&event), -0.4611175068834202);
    }

    #[test]
    fn test_costheta_display() {
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let costheta = decay.costheta(&kshort1, Frame::Helicity).unwrap();
        assert_eq!(
            costheta.to_string(),
            "CosTheta(parent=KK, daughter=K_S1, frame=Helicity)"
        );
    }

    #[test]
    fn test_phi_helicity() {
        let dataset = test_dataset();
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let mut phi = decay.phi(&kshort1, Frame::Helicity).unwrap();
        phi.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(phi.value(&event), -2.657462587335066);
    }

    #[test]
    fn test_phi_display() {
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let phi = decay.phi(&kshort1, Frame::Helicity).unwrap();
        assert_eq!(
            phi.to_string(),
            "Phi(parent=KK, daughter=K_S1, frame=Helicity)"
        );
    }

    #[test]
    fn test_costheta_gottfried_jackson() {
        let dataset = test_dataset();
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let mut costheta = decay.costheta(&kshort1, Frame::GottfriedJackson).unwrap();
        costheta.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(costheta.value(&event), 0.09198832278032032);
    }

    #[test]
    fn test_phi_gottfried_jackson() {
        let dataset = test_dataset();
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let mut phi = decay.phi(&kshort1, Frame::GottfriedJackson).unwrap();
        phi.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(phi.value(&event), -2.7139131991339056);
    }

    #[test]
    fn test_angles() {
        let dataset = test_dataset();
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let mut angles = decay.angles(&kshort1, Frame::Helicity).unwrap();
        angles.costheta.bind(dataset.metadata()).unwrap();
        angles.phi.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(angles.costheta.value(&event), -0.4611175068834202);
        assert_relative_eq!(angles.phi.value(&event), -2.657462587335066);
    }

    #[test]
    fn test_angles_display() {
        let (reaction, kk, kshort1, _) = reaction();
        let decay = reaction.decay(&kk).unwrap();
        let angles = decay.angles(&kshort1, Frame::Helicity).unwrap();
        assert_eq!(
            angles.to_string(),
            "Angles(parent=KK, daughter=K_S1, frame=Helicity)"
        );
    }

    #[test]
    fn test_pol_angle() {
        let dataset = test_dataset();
        let (reaction, _, _, _) = reaction();
        let mut pol_angle = reaction.pol_angle("pol_angle");
        pol_angle.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(pol_angle.value(&event), 1.935929887818673);
    }

    #[test]
    fn test_pol_magnitude() {
        let dataset = test_dataset();
        let mut pol_magnitude = PolMagnitude::new("pol_magnitude");
        pol_magnitude.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(pol_magnitude.value(&event), 0.38562805);
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
        let (reaction, _, _, _) = reaction();
        let mut polarization = reaction.polarization("pol_magnitude", "pol_angle");
        polarization.pol_angle.bind(dataset.metadata()).unwrap();
        polarization.pol_magnitude.bind(dataset.metadata()).unwrap();
        let event = dataset.event_view(0);
        assert_relative_eq!(polarization.pol_angle.value(&event), 1.935929887818673);
        assert_relative_eq!(polarization.pol_magnitude.value(&event), 0.38562805);
    }

    #[test]
    fn test_mandelstam() {
        let dataset = test_dataset();
        let metadata = dataset.metadata();
        let (reaction, _, _, _) = reaction();
        let mut s = reaction.mandelstam(Channel::S);
        let mut t = reaction.mandelstam(Channel::T);
        let mut u = reaction.mandelstam(Channel::U);
        for variable in [&mut s, &mut t, &mut u] {
            variable.bind(metadata).unwrap();
        }
        let event = dataset.event_view(0);
        let resolved = reaction.resolve_two_to_two(&event).unwrap();
        assert_relative_eq!(s.value(&event), resolved.s());
        assert_relative_eq!(t.value(&event), resolved.t());
        assert_relative_eq!(u.value(&event), resolved.u());
    }

    #[test]
    fn test_mandelstam_display() {
        let (reaction, _, _, _) = reaction();
        let s = reaction.mandelstam(Channel::S);
        assert_eq!(s.to_string(), "Mandelstam(channel=s)");
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
