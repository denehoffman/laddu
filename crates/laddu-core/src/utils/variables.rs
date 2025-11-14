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
        vectors::Vec3,
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
    fn new_single<S: Into<String>>(name: S) -> Self {
        Self {
            names: vec![name.into()],
            indices: Vec::new(),
        }
    }

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

    fn index(&self) -> usize {
        self.indices
            .first()
            .copied()
            .expect("P4Selection is expected to contain at least one element")
    }

    fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    fn names(&self) -> &[String] {
        &self.names
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
    constituents: P4Selection,
}
impl Mass {
    /// Create a new [`Mass`] from the sum of the four-momenta identified by `constituents` in the
    /// [`EventData`]'s `p4s` field.
    pub fn new<I, S>(constituents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            constituents: P4Selection::new_many(constituents),
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
        event.get_p4_sum(self.constituents.indices()).m()
    }
}

/// A struct for obtaining the $`\cos\theta`$ (cosine of the polar angle) of a decay product in
/// a given reference frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    beam: P4Selection,
    recoil: P4Selection,
    daughter: P4Selection,
    resonance: P4Selection,
    frame: Frame,
}
impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CosTheta(beam={}, recoil=[{}], daughter=[{}], resonance=[{}], frame={})",
            names_to_string(self.beam.names()),
            names_to_string(self.recoil.names()),
            names_to_string(self.daughter.names()),
            names_to_string(self.resonance.names()),
            self.frame
        )
    }
}
impl CosTheta {
    /// Construct the angle given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<S, R, D, X>(beam: S, recoil: R, daughter: D, resonance: X, frame: Frame) -> Self
    where
        S: Into<String>,
        R: IntoIterator,
        R::Item: Into<String>,
        D: IntoIterator,
        D::Item: Into<String>,
        X: IntoIterator,
        X::Item: Into<String>,
    {
        Self {
            beam: P4Selection::new_single(beam.into()),
            recoil: P4Selection::new_many(recoil),
            daughter: P4Selection::new_many(daughter),
            resonance: P4Selection::new_many(resonance),
            frame,
        }
    }
}
impl Default for CosTheta {
    fn default() -> Self {
        Self {
            beam: P4Selection::new_single("beam"),
            recoil: P4Selection::new_many(["recoil"]),
            daughter: P4Selection::new_many(["daughter"]),
            resonance: P4Selection::new_many(["res_1", "res_2"]),
            frame: Frame::Helicity,
        }
    }
}
#[typetag::serde]
impl Variable for CosTheta {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.beam.bind(metadata)?;
        self.recoil.bind(metadata)?;
        self.daughter.bind(metadata)?;
        self.resonance.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        let beam = event.p4s[self.beam.index()];
        let recoil = event.get_p4_sum(self.recoil.indices());
        let daughter = event.get_p4_sum(self.daughter.indices());
        let resonance = event.get_p4_sum(self.resonance.indices());
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
    beam: P4Selection,
    recoil: P4Selection,
    daughter: P4Selection,
    resonance: P4Selection,
    frame: Frame,
}
impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phi(beam={}, recoil=[{}], daughter=[{}], resonance=[{}], frame={})",
            names_to_string(self.beam.names()),
            names_to_string(self.recoil.names()),
            names_to_string(self.daughter.names()),
            names_to_string(self.resonance.names()),
            self.frame
        )
    }
}
impl Phi {
    /// Construct the angle given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<S, R, D, X>(beam: S, recoil: R, daughter: D, resonance: X, frame: Frame) -> Self
    where
        S: Into<String>,
        R: IntoIterator,
        R::Item: Into<String>,
        D: IntoIterator,
        D::Item: Into<String>,
        X: IntoIterator,
        X::Item: Into<String>,
    {
        Self {
            beam: P4Selection::new_single(beam.into()),
            recoil: P4Selection::new_many(recoil),
            daughter: P4Selection::new_many(daughter),
            resonance: P4Selection::new_many(resonance),
            frame,
        }
    }
}
impl Default for Phi {
    fn default() -> Self {
        Self {
            beam: P4Selection::new_single("beam"),
            recoil: P4Selection::new_many(["recoil"]),
            daughter: P4Selection::new_many(["daughter"]),
            resonance: P4Selection::new_many(["res_1", "res_2"]),
            frame: Frame::Helicity,
        }
    }
}
#[typetag::serde]
impl Variable for Phi {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.beam.bind(metadata)?;
        self.recoil.bind(metadata)?;
        self.daughter.bind(metadata)?;
        self.resonance.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        let beam = event.p4s[self.beam.index()];
        let recoil = event.get_p4_sum(self.recoil.indices());
        let daughter = event.get_p4_sum(self.daughter.indices());
        let resonance = event.get_p4_sum(self.resonance.indices());
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
            "Angles(beam={}, recoil=[{}], daughter=[{}], resonance=[{}], frame={})",
            names_to_string(self.costheta.beam.names()),
            names_to_string(self.costheta.recoil.names()),
            names_to_string(self.costheta.daughter.names()),
            names_to_string(self.costheta.resonance.names()),
            self.costheta.frame
        )
    }
}
impl Angles {
    /// Construct the angles given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<S, R, D, X>(beam: S, recoil: R, daughter: D, resonance: X, frame: Frame) -> Self
    where
        S: Into<String> + Clone,
        R: IntoIterator + Clone,
        R::Item: Into<String>,
        D: IntoIterator + Clone,
        D::Item: Into<String>,
        X: IntoIterator + Clone,
        X::Item: Into<String>,
    {
        Self {
            costheta: CosTheta::new(
                beam.clone(),
                recoil.clone(),
                daughter.clone(),
                resonance.clone(),
                frame,
            ),
            phi: Phi::new(beam, recoil, daughter, resonance, frame),
        }
    }
}

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolAngle {
    beam: P4Selection,
    recoil: P4Selection,
    angle_aux: AuxSelection,
}
impl Display for PolAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolAngle(beam={}, recoil=[{}], angle_aux={})",
            names_to_string(self.beam.names()),
            names_to_string(self.recoil.names()),
            self.angle_aux.name(),
        )
    }
}
impl PolAngle {
    /// Constructs the polarization angle given the named four-momenta for the beam and target
    /// (recoil) particles.
    pub fn new<S, R, A>(beam: S, recoil: R, angle_aux: A) -> Self
    where
        S: Into<String>,
        R: IntoIterator,
        R::Item: Into<String>,
        A: Into<String>,
    {
        Self {
            beam: P4Selection::new_single(beam.into()),
            recoil: P4Selection::new_many(recoil),
            angle_aux: AuxSelection::new(angle_aux.into()),
        }
    }
}
#[typetag::serde]
impl Variable for PolAngle {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.beam.bind(metadata)?;
        self.recoil.bind(metadata)?;
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
            "Polarization(beam={}, recoil=[{}], magnitude_aux={}, angle_aux={})",
            names_to_string(self.pol_angle.beam.names()),
            names_to_string(self.pol_angle.recoil.names()),
            self.pol_magnitude.magnitude_aux.name(),
            self.pol_angle.angle_aux.name(),
        )
    }
}
impl Polarization {
    /// Constructs the polarization angle and magnitude given the named four-momenta for the beam
    /// and target (recoil) particles and distinct auxiliary columns for magnitude and angle.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude_aux` and `angle_aux` refer to the same auxiliary column name.
    pub fn new<S, R, M, A>(beam: S, recoil: R, magnitude_aux: M, angle_aux: A) -> Self
    where
        S: Into<String>,
        R: IntoIterator,
        R::Item: Into<String>,
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
            pol_angle: PolAngle::new(beam, recoil, angle_aux),
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
    p1: P4Selection,
    p2: P4Selection,
    p3: P4Selection,
    p4: P4Selection,
    missing: Option<u8>,
    channel: Channel,
}
impl Display for Mandelstam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mandelstam(p1=[{}], p2=[{}], p3=[{}], p4=[{}], channel={})",
            names_to_string(self.p1.names()),
            names_to_string(self.p2.names()),
            names_to_string(self.p3.names()),
            names_to_string(self.p4.names()),
            self.channel,
        )
    }
}
impl Mandelstam {
    /// Constructs the Mandelstam variable for the given `channel` and particles.
    /// Fields which can take lists of more than one index will add
    /// the relevant four-momenta to make a new particle from the constituents.
    pub fn new<P1, P2, P3, P4>(
        p1: P1,
        p2: P2,
        p3: P3,
        p4: P4,
        channel: Channel,
    ) -> LadduResult<Self>
    where
        P1: IntoIterator,
        P1::Item: Into<String>,
        P2: IntoIterator,
        P2::Item: Into<String>,
        P3: IntoIterator,
        P3::Item: Into<String>,
        P4: IntoIterator,
        P4::Item: Into<String>,
    {
        let p1 = P4Selection::new_many(p1);
        let p2 = P4Selection::new_many(p2);
        let p3 = P4Selection::new_many(p3);
        let p4 = P4Selection::new_many(p4);
        let mut missing = None;
        if p1.is_empty() {
            missing = Some(1)
        }
        if p2.is_empty() {
            if missing.is_none() {
                missing = Some(2)
            } else {
                return Err(LadduError::Custom("A maximum of one particle may be ommitted while constructing a Mandelstam variable!".to_string()));
            }
        }
        if p3.is_empty() {
            if missing.is_none() {
                missing = Some(3)
            } else {
                return Err(LadduError::Custom("A maximum of one particle may be ommitted while constructing a Mandelstam variable!".to_string()));
            }
        }
        if p4.is_empty() {
            if missing.is_none() {
                missing = Some(4)
            } else {
                return Err(LadduError::Custom("A maximum of one particle may be ommitted while constructing a Mandelstam variable!".to_string()));
            }
        }
        Ok(Self {
            p1,
            p2,
            p3,
            p4,
            missing,
            channel,
        })
    }
}

#[typetag::serde]
impl Variable for Mandelstam {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.p1.bind(metadata)?;
        self.p2.bind(metadata)?;
        self.p3.bind(metadata)?;
        self.p4.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &EventData) -> f64 {
        match self.channel {
            Channel::S => match self.missing {
                None | Some(3) | Some(4) => {
                    let p1 = event.get_p4_sum(self.p1.indices());
                    let p2 = event.get_p4_sum(self.p2.indices());
                    (p1 + p2).mag2()
                }
                Some(1) | Some(2) => {
                    let p3 = event.get_p4_sum(self.p3.indices());
                    let p4 = event.get_p4_sum(self.p4.indices());
                    (p3 + p4).mag2()
                }
                _ => unreachable!(),
            },
            Channel::T => match self.missing {
                None | Some(2) | Some(4) => {
                    let p1 = event.get_p4_sum(self.p1.indices());
                    let p3 = event.get_p4_sum(self.p3.indices());
                    (p1 - p3).mag2()
                }
                Some(1) | Some(3) => {
                    let p2 = event.get_p4_sum(self.p2.indices());
                    let p4 = event.get_p4_sum(self.p4.indices());
                    (p4 - p2).mag2()
                }
                _ => unreachable!(),
            },
            Channel::U => match self.missing {
                None | Some(2) | Some(3) => {
                    let p1 = event.get_p4_sum(self.p1.indices());
                    let p4 = event.get_p4_sum(self.p4.indices());
                    (p1 - p4).mag2()
                }
                Some(1) | Some(4) => {
                    let p2 = event.get_p4_sum(self.p2.indices());
                    let p3 = event.get_p4_sum(self.p3.indices());
                    (p3 - p2).mag2()
                }
                _ => unreachable!(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{test_dataset, test_event};
    use approx::assert_relative_eq;

    #[test]
    fn test_mass_single_particle() {
        let dataset = test_dataset();
        let mut mass = Mass::new(["proton"]);
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
        let mut costheta = CosTheta::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        costheta.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(costheta.value(&dataset[0]), -0.4611175068834238);
    }

    #[test]
    fn test_costheta_display() {
        let costheta = CosTheta::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        assert_eq!(
            costheta.to_string(),
            "CosTheta(beam=beam, recoil=[proton], daughter=[kshort1], resonance=[kshort1, kshort2], frame=Helicity)"
        );
    }

    #[test]
    fn test_phi_helicity() {
        let dataset = test_dataset();
        let mut phi = Phi::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(phi.value(&dataset[0]), -2.657462587335066);
    }

    #[test]
    fn test_phi_display() {
        let phi = Phi::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        assert_eq!(
            phi.to_string(),
            "Phi(beam=beam, recoil=[proton], daughter=[kshort1], resonance=[kshort1, kshort2], frame=Helicity)"
        );
    }

    #[test]
    fn test_costheta_gottfried_jackson() {
        let dataset = test_dataset();
        let mut costheta = CosTheta::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::GottfriedJackson,
        );
        costheta.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(costheta.value(&dataset[0]), 0.09198832278031577);
    }

    #[test]
    fn test_phi_gottfried_jackson() {
        let dataset = test_dataset();
        let mut phi = Phi::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::GottfriedJackson,
        );
        phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(phi.value(&dataset[0]), -2.713913199133907);
    }

    #[test]
    fn test_angles() {
        let dataset = test_dataset();
        let mut angles = Angles::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        angles.costheta.bind(dataset.metadata()).unwrap();
        angles.phi.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(angles.costheta.value(&dataset[0]), -0.4611175068834238);
        assert_relative_eq!(angles.phi.value(&dataset[0]), -2.657462587335066);
    }

    #[test]
    fn test_angles_display() {
        let angles = Angles::new(
            "beam",
            ["proton"],
            ["kshort1"],
            ["kshort1", "kshort2"],
            Frame::Helicity,
        );
        assert_eq!(
            angles.to_string(),
            "Angles(beam=beam, recoil=[proton], daughter=[kshort1], resonance=[kshort1, kshort2], frame=Helicity)"
        );
    }

    #[test]
    fn test_pol_angle() {
        let dataset = test_dataset();
        let mut pol_angle = PolAngle::new("beam", ["proton"], "pol_angle");
        pol_angle.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(pol_angle.value(&dataset[0]), 1.93592989);
    }

    #[test]
    fn test_pol_angle_display() {
        let pol_angle = PolAngle::new("beam", ["proton"], "pol_angle");
        assert_eq!(
            pol_angle.to_string(),
            "PolAngle(beam=beam, recoil=[proton], angle_aux=pol_angle)"
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
        let mut polarization = Polarization::new("beam", ["proton"], "pol_magnitude", "pol_angle");
        polarization.pol_angle.bind(dataset.metadata()).unwrap();
        polarization.pol_magnitude.bind(dataset.metadata()).unwrap();
        assert_relative_eq!(polarization.pol_angle.value(&dataset[0]), 1.93592989);
        assert_relative_eq!(polarization.pol_magnitude.value(&dataset[0]), 0.38562805);
    }

    #[test]
    fn test_polarization_display() {
        let polarization = Polarization::new("beam", ["proton"], "pol_magnitude", "pol_angle");
        assert_eq!(
            polarization.to_string(),
            "Polarization(beam=beam, recoil=[proton], magnitude_aux=pol_magnitude, angle_aux=pol_angle)"
        );
    }

    #[test]
    fn test_mandelstam() {
        let dataset = test_dataset();
        let metadata = dataset.metadata();
        let mut s = Mandelstam::new(
            ["beam"],
            Vec::<&str>::new(),
            ["kshort1", "kshort2"],
            ["proton"],
            Channel::S,
        )
        .unwrap();
        let mut t = Mandelstam::new(
            ["beam"],
            Vec::<&str>::new(),
            ["kshort1", "kshort2"],
            ["proton"],
            Channel::T,
        )
        .unwrap();
        let mut u = Mandelstam::new(
            ["beam"],
            Vec::<&str>::new(),
            ["kshort1", "kshort2"],
            ["proton"],
            Channel::U,
        )
        .unwrap();
        let mut sp = Mandelstam::new(
            Vec::<&str>::new(),
            ["beam"],
            ["proton"],
            ["kshort1", "kshort2"],
            Channel::S,
        )
        .unwrap();
        let mut tp = Mandelstam::new(
            Vec::<&str>::new(),
            ["beam"],
            ["proton"],
            ["kshort1", "kshort2"],
            Channel::T,
        )
        .unwrap();
        let mut up = Mandelstam::new(
            Vec::<&str>::new(),
            ["beam"],
            ["proton"],
            ["kshort1", "kshort2"],
            Channel::U,
        )
        .unwrap();
        for variable in [&mut s, &mut t, &mut u, &mut sp, &mut tp, &mut up] {
            variable.bind(metadata).unwrap();
        }
        let event = &dataset[0];
        assert_relative_eq!(s.value(event), 18.504011052120063);
        assert_relative_eq!(s.value(event), sp.value(event),);
        assert_relative_eq!(t.value(event), -0.19222859969898076);
        assert_relative_eq!(t.value(event), tp.value(event),);
        assert_relative_eq!(u.value(event), -14.404198931464428);
        assert_relative_eq!(u.value(event), up.value(event),);
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
        let s = Mandelstam::new(
            ["beam"],
            Vec::<&str>::new(),
            ["kshort1", "kshort2"],
            ["proton"],
            Channel::S,
        )
        .unwrap();
        assert_eq!(
            s.to_string(),
            "Mandelstam(p1=[beam], p2=[], p3=[kshort1, kshort2], p4=[proton], channel=s)"
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
