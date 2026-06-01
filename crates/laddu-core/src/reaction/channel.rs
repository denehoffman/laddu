use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

use serde::{Deserialize, Serialize};

use super::{MassSampler, MomentumSource, Particle, ParticleGeneration, ParticleSource};
use crate::{
    data::EventLike,
    kinematics::{Axis, AxisSource, Frame, FrameAxes},
    quantum::{ExternalId, Isospin, Statistics},
    variables::{Angles, CosTheta, Phi, PolAngle, Polarization},
    vectors::{Vec3, Vec4},
    Charge, LadduError, LadduResult, Mandelstam, MandelstamChannel, Mass, Parity,
    ParticleProperties, ScalarDistribution, J,
};

/// Stable index for a particle edge in a [`Channel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParticleId(pub usize);

/// Stable index for an interaction vertex in a [`Channel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexId(pub usize);

/// An interaction vertex connecting incoming and outgoing particles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    /// User-facing vertex label.
    pub label: String,
    #[serde(default)]
    generation: Option<VertexGenerator>,
}
impl Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

impl Vertex {
    /// Return the vertex label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the generation annotation for this vertex, if one is configured.
    pub fn generation(&self) -> Option<&VertexGenerator> {
        self.generation.as_ref()
    }

    /// Set the generation annotation for this vertex.
    pub fn with_generation(mut self, generation: VertexGenerator) -> Self {
        self.generation = Some(generation);
        self
    }
}

/// Generation annotations attached to a vertex.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VertexGenerator {
    /// Generate a two-to-two production vertex with a sampled Mandelstam-t value.
    TwoToTwo {
        /// Distribution used to sample Mandelstam t.
        t: ScalarDistribution,
    },
}

/// One endpoint of a particle edge in the channel topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endpoint {
    /// The particle enters from outside the modeled topology.
    ExternalIn,
    /// The particle exits to outside the modeled topology.
    ExternalOut,
    /// The particle is attached to an internal vertex.
    Vertex(VertexId),
}

/// A directed reaction topology with particle properties and generation annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    particles: Vec<Particle>,
    vertices: Vec<Vertex>,
    particle_by_name: HashMap<String, ParticleId>,
    vertex_by_name: HashMap<String, VertexId>,
}

impl Channel {
    /// Construct an empty channel.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            vertices: Vec::new(),
            particle_by_name: HashMap::new(),
            vertex_by_name: HashMap::new(),
        }
    }

    fn validate_label(kind: &str, label: &str) -> LadduResult<()> {
        if label.trim().is_empty() {
            return Err(LadduError::Custom(format!("{kind} label cannot be empty")));
        }
        Ok(())
    }

    fn pid(&self, particle: &str) -> LadduResult<ParticleId> {
        self.particle_by_name
            .get(particle)
            .copied()
            .ok_or_else(|| LadduError::Custom(format!("Unknown particle '{}'", particle)))
    }
    fn vid(&self, vertex: &str) -> LadduResult<VertexId> {
        self.vertex_by_name
            .get(vertex)
            .copied()
            .ok_or_else(|| LadduError::Custom(format!("Unknown vertex '{}'", vertex)))
    }
    fn pids(&self) -> impl Iterator<Item = ParticleId> {
        (0..self.particles.len()).map(ParticleId)
    }
    fn vids(&self) -> impl Iterator<Item = VertexId> {
        (0..self.vertices.len()).map(VertexId)
    }
    fn incoming_to_vertex(&self, vid: VertexId) -> Vec<ParticleId> {
        self.particles
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p.to {
                Endpoint::Vertex(v) if v == vid => Some(ParticleId(i)),
                _ => None,
            })
            .collect()
    }
    fn outgoing_from_vertex(&self, vid: VertexId) -> Vec<ParticleId> {
        self.particles
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p.from {
                Endpoint::Vertex(v) if v == vid => Some(ParticleId(i)),
                _ => None,
            })
            .collect()
    }
    fn ensure_particle(&mut self, particle: &str) -> LadduResult<ParticleId> {
        Self::validate_label("particle", particle)?;
        if let Some(pid) = self.particle_by_name.get(particle) {
            return Ok(*pid);
        }
        let pid = ParticleId(self.particles.len());
        self.particles.push(Particle::new(particle));
        self.particle_by_name.insert(particle.to_string(), pid);
        Ok(pid)
    }
    fn apply_vertex<const I: usize, const O: usize>(
        &mut self,
        label: &str,
        incoming: [&str; I],
        outgoing: [&str; O],
    ) -> LadduResult<VertexId> {
        Self::validate_label("vertex", label)?;
        if self.vertex_by_name.contains_key(label) {
            return Err(LadduError::Custom(format!(
                "channel vertex '{label}' already exists"
            )));
        }
        if I == 0 || O == 0 {
            return Err(LadduError::Custom(
                "channel vertices require at least one incoming and one outgoing particle"
                    .to_string(),
            ));
        }

        let mut members = HashSet::new();
        for particle in incoming.iter().chain(outgoing.iter()) {
            Self::validate_label("particle", particle)?;
            if !members.insert(*particle) {
                return Err(LadduError::Custom(format!(
                    "channel vertex '{label}' repeats particle '{particle}'"
                )));
            }
        }

        let vid = VertexId(self.vertices.len());
        self.vertices.push(Vertex {
            label: label.to_string(),
            generation: None,
        });
        self.vertex_by_name.insert(label.to_string(), vid);

        for particle in incoming {
            let pid = self.ensure_particle(particle)?;
            let current = self.particles[pid.0].to;
            if current != Endpoint::ExternalOut {
                return Err(LadduError::Custom(format!(
                    "channel particle '{particle}' is already consumed by {}",
                    self.endpoint_display(current)
                )));
            }
            self.particles[pid.0].to = Endpoint::Vertex(vid);
        }
        for particle in outgoing {
            let pid = self.ensure_particle(particle)?;
            let current = self.particles[pid.0].from;
            if current != Endpoint::ExternalIn {
                return Err(LadduError::Custom(format!(
                    "channel particle '{particle}' is already produced by {}",
                    self.endpoint_display(current)
                )));
            }
            self.particles[pid.0].from = Endpoint::Vertex(vid);
        }
        for pid in self.pids() {
            let particle = &self.particles[pid.0];
            if particle.is_missing()
                && matches!(particle.from, Endpoint::Vertex(_))
                && matches!(particle.to, Endpoint::Vertex(_))
            {
                return Err(LadduError::Custom(format!(
                    "internal particle '{}' cannot be marked missing",
                    particle.label
                )));
            }
        }
        self.validate_acyclic()?;
        Ok(vid)
    }
    fn endpoint_display(&self, endpoint: Endpoint) -> String {
        match endpoint {
            Endpoint::ExternalIn => "ExternalIn".to_string(),
            Endpoint::ExternalOut => "ExternalOut".to_string(),
            Endpoint::Vertex(vid) => format!("vertex '{}'", self.vertices[vid.0].label),
        }
    }
    fn validate_acyclic(&self) -> LadduResult<()> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Mark {
            Visiting,
            Done,
        }

        fn visit(
            channel: &Channel,
            vid: VertexId,
            marks: &mut HashMap<VertexId, Mark>,
        ) -> LadduResult<()> {
            match marks.get(&vid).copied() {
                Some(Mark::Done) => return Ok(()),
                Some(Mark::Visiting) => {
                    return Err(LadduError::Custom(format!(
                        "channel vertex '{}' introduces a cycle",
                        channel.vertices[vid.0].label
                    )));
                }
                None => {}
            }
            marks.insert(vid, Mark::Visiting);
            for particle in channel.outgoing_from_vertex(vid) {
                if let Endpoint::Vertex(child) = channel.particles[particle.0].to {
                    visit(channel, child, marks)?;
                }
            }
            marks.insert(vid, Mark::Done);
            Ok(())
        }

        let mut marks = HashMap::new();
        for vid in self.vids() {
            visit(self, vid, &mut marks)?;
        }
        Ok(())
    }
    fn p4_at_vertex_by_id<E: EventLike + ?Sized>(
        &self,
        pid: ParticleId,
        vid: VertexId,
        event: &E,
    ) -> LadduResult<Vec4> {
        let path = self.boost_path(vid)?;
        self.p4_along_vertex_path(pid, &path, event)
    }
    fn axis_vector<E: EventLike + ?Sized>(&self, axis: &Axis, event: &E) -> LadduResult<Vec3> {
        if axis.frame().trim().is_empty() {
            return Err(LadduError::Custom(
                "axis evaluation frame cannot be empty".to_string(),
            ));
        }
        let vid = self.vid(axis.frame())?;
        let vector = match axis.source() {
            AxisSource::Particle(particle) => self
                .p4_at_vertex_by_id(self.pid(particle)?, vid, event)?
                .vec3(),
            AxisSource::Normal { a, b } => {
                let a = self.p4_at_vertex_by_id(self.pid(a)?, vid, event)?.vec3();
                let b = self.p4_at_vertex_by_id(self.pid(b)?, vid, event)?.vec3();
                a.cross(&b)
            }
        };
        Ok(axis.sign().apply(vector))
    }
    fn frame_axes<E: EventLike + ?Sized>(
        &self,
        frame: &Frame,
        event: &E,
    ) -> LadduResult<FrameAxes> {
        FrameAxes::from_y_z(
            self.axis_vector(frame.axes().y(), event)?,
            self.axis_vector(frame.axes().z(), event)?,
        )
    }
    fn parent_vertices(&self, vertex: VertexId) -> Vec<VertexId> {
        self.particles
            .iter()
            .filter_map(|p| match (p.from, p.to) {
                (Endpoint::Vertex(parent), Endpoint::Vertex(child)) if child == vertex => {
                    Some(parent)
                }
                _ => None,
            })
            .collect()
    }
    fn boost_path(&self, vid: VertexId) -> LadduResult<Vec<VertexId>> {
        let mut reversed = Vec::new();
        let mut current = vid;
        loop {
            reversed.push(current);
            let parents = self.parent_vertices(current);
            match parents.len() {
                0 => break,
                1 => {
                    current = parents[0];
                }
                _ => {
                    return Err(LadduError::Custom(format!(
                        "Ambiguous vertex ancestry for vertex ({:?}) -> {}",
                        parents
                            .into_iter()
                            .map(|vid| self.vertices[vid.0].clone())
                            .collect::<Vec<_>>(),
                        self.vertices[current.0]
                    )));
                }
            }
        }
        reversed.reverse();
        Ok(reversed)
    }
    fn p4_along_vertex_path<E: EventLike + ?Sized>(
        &self,
        pid: ParticleId,
        path: &[VertexId],
        event: &E,
    ) -> LadduResult<Vec4> {
        let mut p = self.p4_by_pid(pid, event)?;
        let mut frames = Vec::with_capacity(path.len());
        for &vid in path {
            frames.push(self.vertex_p4_lab(vid, event)?);
        }
        for i in 0..frames.len() {
            let frame = frames[i];
            p = p.boost(&-frame.beta());
            for later in frames.iter_mut().skip(i + 1) {
                *later = later.boost(&-frame.beta());
            }
        }
        Ok(p)
    }
    fn vertex_p4_lab<E: EventLike + ?Sized>(&self, vid: VertexId, event: &E) -> LadduResult<Vec4> {
        let mut total = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for pid in self.incoming_to_vertex(vid) {
            total = total + self.p4_by_pid(pid, event)?;
        }
        Ok(total)
    }
    fn p4_by_pid<E: EventLike + ?Sized>(&self, pid: ParticleId, event: &E) -> LadduResult<Vec4> {
        let particle = &self.particles[pid.0];
        Ok(match &particle.source {
            ParticleSource::Inferred => {
                if let Endpoint::Vertex(decay) = particle.to {
                    if self.incoming_to_vertex(decay).len() == 1 {
                        let mut sum = Vec4::new(0.0, 0.0, 0.0, 0.0);
                        for daughter in self.outgoing_from_vertex(decay) {
                            sum = sum + self.p4_by_pid(daughter, event)?;
                        }
                        sum
                    } else {
                        event.p4(&particle.label).ok_or_else(|| {
                            LadduError::Custom(format!(
                                "Particle labeled '{}' missing in dataset",
                                particle.label
                            ))
                        })?
                    }
                } else {
                    event.p4(&particle.label).ok_or_else(|| {
                        LadduError::Custom(format!(
                            "Particle labeled '{}' missing in dataset",
                            particle.label
                        ))
                    })?
                }
            }
            ParticleSource::Stored => event.p4(&particle.label).ok_or_else(|| {
                LadduError::Custom(format!(
                    "Particle labeled '{}' missing in dataset",
                    particle.label
                ))
            })?,
            ParticleSource::Missing => self.infer_missing(pid, event)?,
        })
    }
    fn infer_missing<E: EventLike + ?Sized>(
        &self,
        missing_pid: ParticleId,
        event: &E,
    ) -> LadduResult<Vec4> {
        let mut incoming = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let mut outgoing = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let missing = &self.particles[missing_pid.0];
        for pid in self.pids() {
            if pid == missing_pid {
                continue;
            }
            let p = &self.particles[pid.0];
            match (p.from, p.to) {
                (Endpoint::ExternalIn, _) => {
                    incoming = incoming + self.p4_by_pid(pid, event)?;
                }
                (_, Endpoint::ExternalOut) => {
                    outgoing = outgoing + self.p4_by_pid(pid, event)?;
                }
                _ => {
                    // internal particles cannot be marked missing
                }
            }
        }
        match (missing.from, missing.to) {
            (Endpoint::ExternalIn, _) => Ok(outgoing - incoming),
            (_, Endpoint::ExternalOut) => Ok(incoming - outgoing),
            _ => Err(LadduError::Custom(format!(
                "Internal particle '{}' cannot be inferred as missing",
                missing.label()
            ))),
        }
    }
}
impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}
impl Channel {
    /// Edit an existing particle annotation.
    pub fn edit_particle(&mut self, particle: &str) -> LadduResult<ParticleEdit<'_>> {
        let pid = self.pid(particle)?;
        Ok(ParticleEdit {
            particle: &mut self.particles[pid.0],
        })
    }

    /// Edit an existing vertex annotation.
    pub fn edit_vertex(&mut self, vertex: &str) -> LadduResult<VertexEdit<'_>> {
        let vid = self.vid(vertex)?;
        Ok(VertexEdit {
            vertex: &mut self.vertices[vid.0],
        })
    }

    /// Declare a vertex and create or update the particle edges attached to it.
    pub fn create_vertex<const I: usize, const O: usize>(
        &mut self,
        label: &str,
        incoming: [&str; I],
        outgoing: [&str; O],
    ) -> LadduResult<VertexEdit<'_>> {
        let mut candidate = self.clone();
        let vid = candidate.apply_vertex(label, incoming, outgoing)?;
        *self = candidate;
        Ok(VertexEdit {
            vertex: &mut self.vertices[vid.0],
        })
    }

    /// Declare a one-to-two decay vertex.
    pub fn create_decay(
        &mut self,
        label: &str,
        parent: &str,
        daughters: [&str; 2],
    ) -> LadduResult<VertexEdit<'_>> {
        self.create_vertex(label, [parent], daughters)
    }

    /// Declare a two-to-two production vertex.
    pub fn create_production(
        &mut self,
        label: &str,
        incoming: [&str; 2],
        outgoing: [&str; 2],
    ) -> LadduResult<VertexEdit<'_>> {
        self.create_vertex(label, incoming, outgoing)
    }

    /// Return all particles in this channel.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Return all vertices in this channel.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Return a particle by label.
    pub fn particle(&self, particle: &str) -> LadduResult<&Particle> {
        Ok(&self.particles[self.pid(particle)?.0])
    }

    /// Return a vertex by label.
    pub fn vertex(&self, vertex: &str) -> LadduResult<&Vertex> {
        Ok(&self.vertices[self.vid(vertex)?.0])
    }

    /// Return particles incoming to a vertex.
    pub fn incoming_particles(&self, vertex: &str) -> LadduResult<Vec<&Particle>> {
        Ok(self
            .incoming_to_vertex(self.vid(vertex)?)
            .into_iter()
            .map(|pid| &self.particles[pid.0])
            .collect())
    }

    /// Return particles outgoing from a vertex.
    pub fn outgoing_particles(&self, vertex: &str) -> LadduResult<Vec<&Particle>> {
        Ok(self
            .outgoing_from_vertex(self.vid(vertex)?)
            .into_iter()
            .map(|pid| &self.particles[pid.0])
            .collect())
    }

    /// Return decay vertices for a particle, defined as vertices with that particle as input.
    pub fn decay_vertices(&self, particle: &str) -> LadduResult<Vec<&Vertex>> {
        let pid = self.pid(particle)?;
        Ok(self
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(index, vertex)| {
                let vid = VertexId(index);
                self.incoming_to_vertex(vid)
                    .contains(&pid)
                    .then_some(vertex)
            })
            .collect())
    }

    /// Evaluate the lab-frame four-momentum for a particle.
    pub fn p4<E: EventLike + ?Sized>(&self, particle: &str, event: &E) -> LadduResult<Vec4> {
        self.p4_by_pid(self.pid(particle)?, event)
    }
    /// Evaluate a particle four-momentum in the rest-frame path of a vertex.
    pub fn p4_at_vertex<E: EventLike + ?Sized>(
        &self,
        particle: &str,
        vertex: &str,
        event: &E,
    ) -> LadduResult<Vec4> {
        let pid = self.pid(particle)?;
        let vid = self.vid(vertex)?;
        self.p4_at_vertex_by_id(pid, vid, event)
    }
    /// Build a mass variable for a channel particle.
    pub fn mass(&self, particle: &str) -> LadduResult<Mass> {
        Ok(Mass {
            evaluator: self.mass_evaluator(particle)?,
        })
    }
    fn mass_evaluator(&self, particle: &str) -> LadduResult<MassEvaluator> {
        Ok(MassEvaluator {
            channel: self.clone(),
            pid: self.pid(particle)?,
        })
    }
    /// Build a Mandelstam variable for a two-to-two vertex.
    pub fn mandelstam(
        &self,
        vertex: &str,
        mandelstam_channel: MandelstamChannel,
    ) -> LadduResult<Mandelstam> {
        Ok(Mandelstam {
            evaluator: self.mandelstam_evaluator(vertex)?,
            mandelstam_channel,
        })
    }
    fn mandelstam_evaluator(&self, vertex: &str) -> LadduResult<MandelstamEvaluator> {
        let vid = self.vid(vertex)?;
        let incoming = self.incoming_to_vertex(vid);
        let outgoing = self.outgoing_from_vertex(vid);

        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(LadduError::Custom(format!(
                "Vertex {} is not two-to-two",
                vertex
            )));
        }
        let pid1 = incoming[0];
        let pid2 = incoming[1];
        let pid3 = outgoing[0];
        let pid4 = outgoing[1];

        let mut n_missing = 0;
        let mut missing = None;
        if self.particles[pid1.0].is_missing() {
            n_missing += 1;
            missing = Some(MissingId::P1);
        }
        if self.particles[pid2.0].is_missing() {
            n_missing += 1;
            missing = Some(MissingId::P2);
        }
        if self.particles[pid3.0].is_missing() {
            n_missing += 1;
            missing = Some(MissingId::P3);
        }
        if self.particles[pid4.0].is_missing() {
            n_missing += 1;
            missing = Some(MissingId::P4);
        }
        if n_missing > 1 {
            return Err(LadduError::Custom(format!(
                "Too many missing four-momenta at vertex {}",
                vertex
            )));
        }
        Ok(MandelstamEvaluator {
            channel: self.clone(),
            pid1,
            pid2,
            pid3,
            pid4,
            missing,
        })
    }
    /// Build angular variables for a particle measured in a symbolic frame.
    pub fn angles(&self, particle: &str, frame: Frame) -> LadduResult<Angles> {
        let evaluator = self.angle_evaluator(particle, frame)?;
        Ok(Angles {
            costheta: CosTheta {
                evaluator: evaluator.clone(),
            },
            phi: Phi { evaluator },
        })
    }
    fn angle_evaluator(&self, particle: &str, frame: Frame) -> LadduResult<AngleEvaluator> {
        let _ = self.vid(frame.origin())?;
        Ok(AngleEvaluator {
            channel: self.clone(),
            pid: self.pid(particle)?,
            frame,
        })
    }

    /// Build a beam-polarization angle variable for a two-to-two production vertex.
    pub fn pol_angle<A>(&self, vertex: &str, angle_aux: A) -> LadduResult<PolAngle>
    where
        A: Into<String>,
    {
        Ok(PolAngle::new(
            self.polarization_angle_evaluator(vertex)?,
            angle_aux,
        ))
    }

    /// Build beam-polarization variables for a two-to-two production vertex.
    pub fn polarization<M, A>(
        &self,
        vertex: &str,
        magnitude_aux: M,
        angle_aux: A,
    ) -> LadduResult<Polarization>
    where
        M: Into<String>,
        A: Into<String>,
    {
        Ok(Polarization::new(
            self.polarization_angle_evaluator(vertex)?,
            magnitude_aux,
            angle_aux,
        ))
    }

    fn polarization_angle_evaluator(
        &self,
        vertex: &str,
    ) -> LadduResult<PolarizationAngleEvaluator> {
        let vid = self.vid(vertex)?;
        let incoming = self.incoming_to_vertex(vid);
        let outgoing = self.outgoing_from_vertex(vid);

        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(LadduError::Custom(format!(
                "Vertex {} is not two-to-two",
                vertex
            )));
        }

        Ok(PolarizationAngleEvaluator {
            channel: self.clone(),
            vertex: vid,
            reference: incoming[0],
            spectator: outgoing[1],
        })
    }
}

/// Mutable particle annotation editor.
pub struct ParticleEdit<'a> {
    particle: &'a mut Particle,
}

impl ParticleEdit<'_> {
    /// Read p4 values from a dataset column with the same label.
    pub fn stored(&mut self) -> &mut Self {
        self.particle.source = ParticleSource::Stored;
        self
    }

    /// Infer the p4 source from topology.
    pub fn inferred(&mut self) -> &mut Self {
        self.particle.source = ParticleSource::Inferred;
        self
    }

    /// Solve this external particle as the missing p4.
    pub fn missing(&mut self) -> LadduResult<&mut Self> {
        if matches!(self.particle.from, Endpoint::Vertex(_))
            && matches!(self.particle.to, Endpoint::Vertex(_))
        {
            return Err(LadduError::Custom(format!(
                "internal particle '{}' cannot be marked missing",
                self.particle.label
            )));
        }
        self.particle.source = ParticleSource::Missing;
        Ok(self)
    }

    /// Replace all particle properties.
    pub fn properties(&mut self, properties: ParticleProperties) -> &mut Self {
        self.particle.properties = properties;
        self
    }

    /// Replace all generation annotations.
    pub fn generation(&mut self, generation: ParticleGeneration) -> &mut Self {
        self.particle.generation = generation;
        self
    }

    /// Set the mass sampler.
    pub fn mass_sampler(&mut self, sampler: MassSampler) -> &mut Self {
        self.particle.generation = self.particle.generation.clone().with_mass_sampler(sampler);
        self
    }

    /// Set the momentum source.
    pub fn momentum(&mut self, momentum: MomentumSource) -> &mut Self {
        self.particle.generation = self.particle.generation.clone().with_momentum(momentum);
        self
    }

    /// Set the particle name property.
    pub fn name(&mut self, name: impl Into<String>) -> &mut Self {
        self.particle.properties.name = Some(name.into());
        self
    }

    /// Set the particle species property.
    pub fn species(&mut self, species: impl Into<String>) -> &mut Self {
        self.particle.properties.species = Some(species.into());
        self
    }

    /// Set the antiparticle species property.
    pub fn antiparticle_species(&mut self, species: impl Into<String>) -> &mut Self {
        self.particle.properties.antiparticle_species = Some(species.into());
        self
    }

    /// Set whether the particle is self-conjugate.
    pub fn self_conjugate(&mut self, value: bool) -> &mut Self {
        self.particle.properties.self_conjugate = Some(value);
        self
    }

    /// Set the spin property.
    pub fn spin(&mut self, spin: J) -> &mut Self {
        self.particle.properties.spin = Some(spin);
        self.particle.properties.statistics = Some(Statistics::from_spin(spin));
        self
    }

    /// Set the parity property.
    pub fn parity(&mut self, parity: Parity) -> &mut Self {
        self.particle.properties.parity = Some(parity);
        self
    }

    /// Set the C-parity property.
    pub fn c_parity(&mut self, parity: Parity) -> &mut Self {
        self.particle.properties.c_parity = Some(parity);
        self
    }

    /// Set the G-parity property.
    pub fn g_parity(&mut self, parity: Parity) -> &mut Self {
        self.particle.properties.g_parity = Some(parity);
        self
    }

    /// Set the electric charge property.
    pub fn charge(&mut self, charge: Charge) -> &mut Self {
        self.particle.properties.charge = Some(charge);
        self
    }

    /// Set the isospin property.
    pub fn isospin(&mut self, isospin: Isospin) -> &mut Self {
        self.particle.properties.isospin = Some(isospin);
        self
    }

    /// Set the strangeness property.
    pub fn strangeness(&mut self, value: i32) -> &mut Self {
        self.particle.properties.strangeness = Some(value);
        self
    }

    /// Set the charm property.
    pub fn charm(&mut self, value: i32) -> &mut Self {
        self.particle.properties.charm = Some(value);
        self
    }

    /// Set the bottomness property.
    pub fn bottomness(&mut self, value: i32) -> &mut Self {
        self.particle.properties.bottomness = Some(value);
        self
    }

    /// Set the topness property.
    pub fn topness(&mut self, value: i32) -> &mut Self {
        self.particle.properties.topness = Some(value);
        self
    }

    /// Set the baryon number property.
    pub fn baryon_number(&mut self, value: i32) -> &mut Self {
        self.particle.properties.baryon_number = Some(value);
        self
    }

    /// Set the electron lepton number property.
    pub fn electron_lepton_number(&mut self, value: i32) -> &mut Self {
        self.particle.properties.electron_lepton_number = Some(value);
        self
    }

    /// Set the muon lepton number property.
    pub fn muon_lepton_number(&mut self, value: i32) -> &mut Self {
        self.particle.properties.muon_lepton_number = Some(value);
        self
    }

    /// Set the tau lepton number property.
    pub fn tau_lepton_number(&mut self, value: i32) -> &mut Self {
        self.particle.properties.tau_lepton_number = Some(value);
        self
    }

    /// Set the statistics property after validating consistency with spin if present.
    pub fn statistics(&mut self, statistics: Statistics) -> LadduResult<&mut Self> {
        let properties = self
            .particle
            .properties
            .clone()
            .with_statistics(statistics)?;
        self.particle.properties = properties;
        Ok(self)
    }

    /// Set the mass property.
    pub fn mass(&mut self, mass: f64) -> &mut Self {
        self.particle.properties.mass = Some(mass);
        self
    }

    /// Append an external identifier.
    pub fn id(&mut self, id: ExternalId) -> &mut Self {
        self.particle.properties.ids.push(id);
        self
    }

    /// Replace external identifiers.
    pub fn ids<I>(&mut self, ids: I) -> &mut Self
    where
        I: IntoIterator<Item = ExternalId>,
    {
        self.particle.properties.ids = ids.into_iter().collect();
        self
    }

    /// Set the spin and parity properties.
    pub fn jp(&mut self, spin: J, parity: Parity) -> &mut Self {
        self.spin(spin).parity(parity)
    }

    /// Set the spin, parity, and C-parity properties.
    pub fn jpc(&mut self, spin: J, parity: Parity, c_parity: Parity) -> &mut Self {
        self.spin(spin).parity(parity).c_parity(c_parity)
    }
}

/// Mutable vertex annotation editor.
pub struct VertexEdit<'a> {
    vertex: &'a mut Vertex,
}

impl VertexEdit<'_> {
    /// Set the generation annotation.
    pub fn generate(&mut self, generation: VertexGenerator) -> &mut Self {
        self.vertex.generation = Some(generation);
        self
    }
}

/// Evaluator for the invariant mass of one channel particle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassEvaluator {
    channel: Channel,
    pid: ParticleId,
}

impl MassEvaluator {
    /// Return the channel particle label.
    pub fn particle(&self) -> &str {
        self.channel.particles[self.pid.0].label()
    }

    /// Evaluate the invariant mass for one event.
    pub fn mass<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.channel.p4_by_pid(self.pid, event)?.m())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum MissingId {
    P1,
    P2,
    P3,
    P4,
}

/// Evaluated Mandelstam invariants for a two-to-two vertex.
pub struct MandelstamValues {
    /// Mandelstam s.
    pub s: f64,
    /// Mandelstam t.
    pub t: f64,
    /// Mandelstam u.
    pub u: f64,
}

/// Evaluator for Mandelstam invariants at a two-to-two vertex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MandelstamEvaluator {
    channel: Channel,
    pid1: ParticleId,
    pid2: ParticleId,
    pid3: ParticleId,
    pid4: ParticleId,
    missing: Option<MissingId>,
}

impl MandelstamEvaluator {
    fn p1<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<Vec4> {
        self.channel.p4_by_pid(self.pid1, event)
    }
    fn p2<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<Vec4> {
        self.channel.p4_by_pid(self.pid2, event)
    }
    fn p3<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<Vec4> {
        self.channel.p4_by_pid(self.pid3, event)
    }
    fn p4<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<Vec4> {
        self.channel.p4_by_pid(self.pid4, event)
    }
    /// Evaluate Mandelstam s.
    pub fn s<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.get(event)?.s)
    }
    /// Evaluate Mandelstam t.
    pub fn t<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.get(event)?.t)
    }
    /// Evaluate Mandelstam u.
    pub fn u<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.get(event)?.u)
    }
    /// Evaluate all Mandelstam invariants.
    pub fn get<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<MandelstamValues> {
        if let Some(mid) = &self.missing {
            match mid {
                MissingId::P1 => {
                    let p2 = self.p2(event)?;
                    let p3 = self.p3(event)?;
                    let p4 = self.p4(event)?;
                    Ok(MandelstamValues {
                        s: (p3 + p4).m2(),
                        t: (p4 - p2).m2(),
                        u: (p3 - p2).m2(),
                    })
                }
                MissingId::P2 => {
                    let p1 = self.p1(event)?;
                    let p3 = self.p3(event)?;
                    let p4 = self.p4(event)?;
                    Ok(MandelstamValues {
                        s: (p3 + p4).m2(),
                        t: (p1 - p3).m2(),
                        u: (p1 - p4).m2(),
                    })
                }
                MissingId::P3 => {
                    let p1 = self.p1(event)?;
                    let p2 = self.p2(event)?;
                    let p4 = self.p4(event)?;
                    Ok(MandelstamValues {
                        s: (p1 + p2).m2(),
                        t: (p4 - p2).m2(),
                        u: (p1 - p4).m2(),
                    })
                }
                MissingId::P4 => {
                    let p1 = self.p1(event)?;
                    let p2 = self.p2(event)?;
                    let p3 = self.p3(event)?;
                    Ok(MandelstamValues {
                        s: (p1 + p2).m2(),
                        t: (p1 - p3).m2(),
                        u: (p3 - p2).m2(),
                    })
                }
            }
        } else {
            let p1 = self.p1(event)?;
            let p2 = self.p2(event)?;
            let p3 = self.p3(event)?;
            // No need to waste time calculating p4
            Ok(MandelstamValues {
                s: (p1 + p2).m2(),
                t: (p1 - p3).m2(),
                u: (p3 - p2).m2(),
            })
        }
    }
}

/// Evaluator for angular variables derived from a channel topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleEvaluator {
    channel: Channel,
    pid: ParticleId,
    frame: Frame,
}

/// Evaluated angular variables.
pub struct AngleValues {
    costheta: f64,
    phi: f64,
}

impl AngleEvaluator {
    /// Return the measured particle label.
    pub fn particle(&self) -> &str {
        self.channel.particles[self.pid.0].label()
    }

    /// Return the symbolic frame definition.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Evaluate cos(theta).
    pub fn costheta<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.get(event)?.costheta)
    }
    /// Evaluate phi.
    pub fn phi<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<f64> {
        Ok(self.get(event)?.phi)
    }
    /// Evaluate all angular variables.
    pub fn get<E: EventLike + ?Sized>(&self, event: &E) -> LadduResult<AngleValues> {
        let origin = self.channel.vid(self.frame.origin())?;
        let p = self
            .channel
            .p4_at_vertex_by_id(self.pid, origin, event)?
            .vec3();
        let axes = self.channel.frame_axes(&self.frame, event)?;
        Ok(AngleValues {
            costheta: axes.costheta(&p),
            phi: axes.phi(&p),
        })
    }
}

/// Evaluator for beam polarization angle relative to a two-to-two production plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarizationAngleEvaluator {
    channel: Channel,
    vertex: VertexId,
    reference: ParticleId,
    spectator: ParticleId,
}

impl PolarizationAngleEvaluator {
    /// Return the production vertex label.
    pub fn vertex(&self) -> &str {
        self.channel.vertices[self.vertex.0].label()
    }

    /// Return the reference particle label used to define the production plane.
    pub fn reference(&self) -> &str {
        self.channel.particles[self.reference.0].label()
    }

    /// Return the spectator particle label used to define the production plane.
    pub fn spectator(&self) -> &str {
        self.channel.particles[self.spectator.0].label()
    }

    /// Evaluate the polarization angle for one event.
    pub fn angle<E: EventLike + ?Sized>(&self, event: &E, lab_angle: f64) -> LadduResult<f64> {
        let reference = self.channel.p4_by_pid(self.reference, event)?;
        let spectator = self.channel.p4_by_pid(self.spectator, event)?;
        let polarization = Vec3::new(lab_angle.cos(), lab_angle.sin(), 0.0);
        let y = reference.vec3().cross(&-spectator.vec3()).unit();
        let numerator = y.dot(&polarization);
        let denominator = reference.vec3().unit().dot(&polarization.cross(&y));
        Ok(f64::atan2(numerator, denominator))
    }
}
