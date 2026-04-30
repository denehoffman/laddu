use std::collections::{HashMap, HashSet};

use fastrand::Rng;
use laddu_core::{
    math::{q_m, Histogram, Sheet},
    Dataset, DatasetMetadata, LadduError, LadduResult, Particle, Reaction, Vec3, Vec4, PI,
};
use serde::{Deserialize, Serialize};

use crate::distributions::{
    Distribution, HistogramSampler, LadduGenRngExt, MandelstamTDistribution, SimpleDistribution,
};

/// Selects which generated particle four-momenta are written into generated datasets.
///
/// The generated reaction layout always retains the full generated graph. This policy only controls
/// which generated particle IDs become p4 columns in generated [`Dataset`] values and which
/// particles have a p4 label in [`GeneratedEventLayout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneratedStorage {
    /// Store every generated particle p4.
    All,
    /// Store only the listed generated particle IDs, preserving reaction p4-label order.
    Only(Vec<String>),
}

impl GeneratedStorage {
    /// Store every generated particle p4.
    pub fn all() -> Self {
        Self::All
    }

    /// Store only the listed generated particle IDs.
    pub fn only<I, S>(ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Only(ids.into_iter().map(Into::into).collect())
    }

    /// Return true if `id` is selected for dataset storage.
    pub fn stores(&self, id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(ids) => ids.iter().any(|stored_id| stored_id == id),
        }
    }

    fn validate(&self, available_ids: &[String]) -> LadduResult<()> {
        let available = available_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let Self::Only(ids) = self else {
            return Ok(());
        };
        let mut seen = HashSet::new();
        for id in ids {
            if !seen.insert(id.as_str()) {
                return Err(LadduError::Custom(format!(
                    "generated storage contains duplicate particle ID '{id}'"
                )));
            }
            if !available.contains(id.as_str()) {
                return Err(LadduError::Custom(format!(
                    "generated storage references unknown particle ID '{id}'"
                )));
            }
        }
        Ok(())
    }

    fn stored_labels(&self, all_labels: &[String]) -> Vec<String> {
        all_labels
            .iter()
            .filter(|label| self.stores(label))
            .cloned()
            .collect()
    }
}

/// Experiment-neutral metadata describing a generated particle species.
///
/// Species metadata is intentionally separate from generated particle IDs and reconstructed
/// reaction particles. It is meant for generator/export layers that need an external particle code
/// or label without forcing laddu to adopt an experiment-specific particle table.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleSpecies {
    /// A numeric species code with an optional namespace.
    Code {
        /// Numeric species identifier.
        id: i64,
        /// Optional namespace, such as `"pdg"`.
        namespace: Option<String>,
    },
    /// A free-form species label.
    Label(String),
}

impl ParticleSpecies {
    /// Construct a species from a numeric code with no namespace.
    pub fn code(id: i64) -> Self {
        Self::Code {
            id,
            namespace: None,
        }
    }

    /// Construct a species from a numeric code in an explicit namespace.
    pub fn with_namespace(namespace: impl Into<String>, id: i64) -> Self {
        Self::Code {
            id,
            namespace: Some(namespace.into()),
        }
    }

    /// Construct a species from a free-form label.
    pub fn label(label: impl Into<String>) -> Self {
        Self::Label(label.into())
    }
}

fn basis(z: Vec3) -> (Vec3, Vec3, Vec3) {
    let z = z.unit();
    let ref_axis = if z.z.abs() < 0.9 {
        Vec3::z()
    } else {
        Vec3::y()
    };
    let x = ref_axis.cross(&z).unit();
    let y = z.cross(&x);
    (x, y, z)
}

/// Generator settings for an initial-state particle.
#[derive(Clone, Debug)]
pub struct InitialGenerator {
    mass: f64,
    energy_distribution: SimpleDistribution,
}

impl InitialGenerator {
    /// Construct a beam with fixed energy.
    pub fn beam_with_fixed_energy(mass: f64, energy: f64) -> Self {
        debug_assert!(mass >= 0.0, "Mass cannot be negative!\nMass: {}", mass);
        debug_assert!(energy > 0.0, "Energy must be positive!\nEnergy: {}", energy);
        Self {
            mass,
            energy_distribution: SimpleDistribution::Fixed(energy),
        }
    }

    /// Construct a beam with uniformly sampled energy.
    pub fn beam(mass: f64, min_energy: f64, max_energy: f64) -> Self {
        debug_assert!(mass >= 0.0, "Mass cannot be negative!\nMass: {}", mass);
        debug_assert!(
            min_energy > 0.0,
            "Minimum energy must be positive!\nMinimum Energy: {}",
            min_energy
        );
        debug_assert!(
            max_energy > min_energy,
            "Maximum energy must be greater than minimum energy!"
        );
        Self {
            mass,
            energy_distribution: SimpleDistribution::Uniform {
                min: min_energy,
                max: max_energy,
            },
        }
    }

    /// Construct a beam with histogram-sampled energy.
    pub fn beam_with_energy_histogram(mass: f64, energy: Histogram) -> LadduResult<Self> {
        debug_assert!(
            mass >= 0.0,
            "Mass must be positive and greater than zero!\nMass: {}",
            mass
        );
        let sampler = HistogramSampler::new(energy)?;
        debug_assert!(
            sampler.hist.bin_edges()[0] >= mass,
            "Mass cannot be greater than the minimum allowed energy!\nMass: {}\nMinimum Energy: {}",
            mass,
            sampler.hist.bin_edges()[0]
        );
        Ok(Self {
            mass,
            energy_distribution: SimpleDistribution::Histogram(sampler),
        })
    }

    /// Construct a target at rest.
    pub fn target(mass: f64) -> Self {
        Self {
            mass,
            energy_distribution: SimpleDistribution::Fixed(mass),
        }
    }
}

/// Generator settings for a generated composite particle.
#[derive(Clone, Debug)]
pub struct CompositeGenerator {
    mass_distribution: SimpleDistribution,
}

impl CompositeGenerator {
    /// Construct a composite mass generator with a uniform mass range.
    pub fn new(min_mass: f64, max_mass: f64) -> Self {
        Self {
            mass_distribution: SimpleDistribution::Uniform {
                min: min_mass,
                max: max_mass,
            },
        }
    }

    fn sample_mass(&self, rng: &mut Rng) -> f64 {
        self.mass_distribution.sample(rng)
    }
}

/// Generator settings for a stable generated particle.
#[derive(Clone, Debug)]
pub struct StableGenerator {
    mass_distribution: SimpleDistribution,
}

impl StableGenerator {
    /// Construct a fixed-mass stable-particle generator.
    pub fn new(mass: f64) -> Self {
        debug_assert!(mass >= 0.0, "Mass cannot be negative!\nMass: {}", mass);
        Self {
            mass_distribution: SimpleDistribution::Fixed(mass),
        }
    }

    fn sample_mass(&self, rng: &mut Rng) -> f64 {
        self.mass_distribution.sample(rng)
    }
}

/// Reconstruction interpretation for a generated particle.
#[derive(Clone, Debug, PartialEq)]
pub enum Reconstruction {
    /// The particle p4 is stored in the analysis dataset under the generated particle ID.
    Stored,
    /// The particle p4 is fixed in the reconstructed reaction.
    Fixed(Vec4),
    /// The particle p4 is inferred from reaction-level constraints.
    Missing,
    /// The particle p4 is reconstructed as a composite of its two generated daughters.
    Composite,
}

/// A generated particle with generation and reconstruction metadata.
#[derive(Clone, Debug)]
pub enum GeneratedParticle {
    /// An initial-state generated particle.
    Initial {
        id: String,
        generator: InitialGenerator,
        reconstruction: Reconstruction,
        species: Option<ParticleSpecies>,
    },
    /// A stable generated particle.
    Stable {
        id: String,
        generator: StableGenerator,
        reconstruction: Reconstruction,
        species: Option<ParticleSpecies>,
    },
    /// A generated composite particle with exactly two generated daughters.
    Composite {
        id: String,
        generator: CompositeGenerator,
        daughters: (Box<GeneratedParticle>, Box<GeneratedParticle>),
        reconstruction: Reconstruction,
        species: Option<ParticleSpecies>,
    },
}

impl GeneratedParticle {
    /// Construct a generated initial-state particle.
    pub fn initial(
        id: impl Into<String>,
        generator: InitialGenerator,
        reconstruction: Reconstruction,
    ) -> Self {
        Self::Initial {
            id: id.into(),
            generator,
            reconstruction,
            species: None,
        }
    }

    /// Construct a generated stable particle.
    pub fn stable(
        id: impl Into<String>,
        generator: StableGenerator,
        reconstruction: Reconstruction,
    ) -> Self {
        Self::Stable {
            id: id.into(),
            generator,
            reconstruction,
            species: None,
        }
    }

    /// Construct a generated composite particle from exactly two ordered daughters.
    pub fn composite(
        id: impl Into<String>,
        generator: CompositeGenerator,
        daughters: (&GeneratedParticle, &GeneratedParticle),
        reconstruction: Reconstruction,
    ) -> Self {
        Self::Composite {
            id: id.into(),
            generator,
            daughters: (Box::new(daughters.0.clone()), Box::new(daughters.1.clone())),
            reconstruction,
            species: None,
        }
    }

    /// Return a copy of this generated particle with species metadata attached.
    pub fn with_species(mut self, species: ParticleSpecies) -> Self {
        match &mut self {
            Self::Initial {
                species: particle_species,
                ..
            }
            | Self::Stable {
                species: particle_species,
                ..
            }
            | Self::Composite {
                species: particle_species,
                ..
            } => *particle_species = Some(species),
        }
        self
    }

    /// Return the generated particle ID.
    pub fn id(&self) -> &str {
        match self {
            Self::Initial { id, .. } | Self::Stable { id, .. } | Self::Composite { id, .. } => id,
        }
    }

    /// Return optional species metadata for this generated particle.
    pub fn species(&self) -> Option<&ParticleSpecies> {
        match self {
            Self::Initial { species, .. }
            | Self::Stable { species, .. }
            | Self::Composite { species, .. } => species.as_ref(),
        }
    }

    /// Return this particle's reconstruction interpretation.
    pub fn reconstruction(&self) -> &Reconstruction {
        match self {
            Self::Initial { reconstruction, .. }
            | Self::Stable { reconstruction, .. }
            | Self::Composite { reconstruction, .. } => reconstruction,
        }
    }

    fn p4_labels(&self) -> Vec<String> {
        let mut labels = vec![self.id().to_string()];
        if let Self::Composite { daughters, .. } = self {
            labels.append(&mut daughters.0.p4_labels());
            labels.append(&mut daughters.1.p4_labels());
        }
        labels
    }

    fn append_decay_layout(
        &self,
        parent_id: Option<usize>,
        produced_vertex_id: Option<usize>,
        storage: &GeneratedStorage,
        particles: &mut Vec<GeneratedParticleLayout>,
        vertices: &mut Vec<GeneratedVertexLayout>,
    ) -> usize {
        let product_id = particles.len();
        particles.push(GeneratedParticleLayout {
            id: self.id().to_string(),
            product_id,
            parent_id,
            species: self.species().cloned(),
            p4_label: storage.stores(self.id()).then(|| self.id().to_string()),
            produced_vertex_id,
            decay_vertex_id: None,
        });
        if let Self::Composite { daughters, .. } = self {
            let vertex_id = vertices.len();
            particles[product_id].decay_vertex_id = Some(vertex_id);
            vertices.push(GeneratedVertexLayout {
                vertex_id,
                kind: GeneratedVertexKind::Decay,
                incoming_product_ids: vec![product_id],
                outgoing_product_ids: Vec::new(),
            });
            let daughter_1_id = daughters.0.append_decay_layout(
                Some(product_id),
                Some(vertex_id),
                storage,
                particles,
                vertices,
            );
            let daughter_2_id = daughters.1.append_decay_layout(
                Some(product_id),
                Some(vertex_id),
                storage,
                particles,
                vertices,
            );
            vertices[vertex_id].outgoing_product_ids = vec![daughter_1_id, daughter_2_id];
        }
        product_id
    }

    fn sample_mass(&self, rng: &mut Rng) -> f64 {
        match self {
            Self::Initial { generator, .. } => generator.mass,
            Self::Stable { generator, .. } => generator.sample_mass(rng),
            Self::Composite { generator, .. } => generator.sample_mass(rng),
        }
    }

    fn generated_particle(&self) -> LadduResult<Particle> {
        match self.reconstruction() {
            Reconstruction::Stored => Ok(Particle::stored(self.id())),
            Reconstruction::Fixed(p4) => Ok(Particle::fixed(self.id(), *p4)),
            Reconstruction::Missing => Ok(Particle::missing(self.id())),
            Reconstruction::Composite => {
                let Self::Composite { daughters, .. } = self else {
                    return Err(LadduError::Custom(format!(
                        "particle '{}' cannot use composite reconstruction without daughters",
                        self.id()
                    )));
                };
                let daughter_1 = daughters.0.generated_particle()?;
                let daughter_2 = daughters.1.generated_particle()?;
                Particle::composite(self.id(), (&daughter_1, &daughter_2))
            }
        }
    }

    fn validate_reconstruction(&self) -> LadduResult<()> {
        match (self, self.reconstruction()) {
            (Self::Composite { daughters, .. }, Reconstruction::Composite) => {
                daughters.0.validate_reconstruction()?;
                daughters.1.validate_reconstruction()?;
                Ok(())
            }
            (Self::Composite { .. }, _) => Ok(()),
            (_, Reconstruction::Composite) => Err(LadduError::Custom(format!(
                "particle '{}' cannot use composite reconstruction without daughters",
                self.id()
            ))),
            _ => Ok(()),
        }
    }

    fn collect_ids<'a>(&'a self, seen: &mut HashSet<&'a str>) -> LadduResult<()> {
        if !seen.insert(self.id()) {
            return Err(LadduError::Custom(format!(
                "duplicate generated particle identifier '{}'",
                self.id()
            )));
        }
        if let Self::Composite { daughters, .. } = self {
            daughters.0.collect_ids(seen)?;
            daughters.1.collect_ids(seen)?;
        }
        Ok(())
    }

    fn generate_decay(
        &self,
        rng: &mut Rng,
        p4_cm: Vec4,
        cm_to_lab_boost: &Vec3,
        p4_storage: &mut HashMap<String, Vec<Vec4>>,
    ) {
        let p4_lab = p4_cm.boost(cm_to_lab_boost);
        if let Some(storage) = p4_storage.get_mut(self.id()) {
            storage.push(p4_lab);
        }

        let Self::Composite { daughters, .. } = self else {
            return;
        };
        let d1 = &daughters.0;
        let d2 = &daughters.1;
        let parent_mass = p4_cm.m();
        let m1 = d1.sample_mass(rng);
        let m2 = d2.sample_mass(rng);
        let q = q_m(parent_mass, m1, m2, Sheet::Physical).re;
        let parent_msq = parent_mass * parent_mass;
        let msq1 = m1 * m1;
        let msq2 = m2 * m2;
        let e1 = (parent_msq + msq1 - msq2) / (2.0 * parent_mass);
        let e2 = (parent_msq + msq2 - msq1) / (2.0 * parent_mass);

        let cos_theta = rng.uniform(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = rng.uniform(0.0, 2.0 * PI);
        let (sin_phi, cos_phi) = phi.sin_cos();

        let dir = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let p1_p4_rest = (dir * q).with_energy(e1);
        let p2_p4_rest = (-dir * q).with_energy(e2);
        let parent_to_cm_boost = p4_cm.beta();
        let p1_p4_cm = p1_p4_rest.boost(&parent_to_cm_boost);
        let p2_p4_cm = p2_p4_rest.boost(&parent_to_cm_boost);
        d1.generate_decay(rng, p1_p4_cm, cm_to_lab_boost, p4_storage);
        d2.generate_decay(rng, p2_p4_cm, cm_to_lab_boost, p4_storage);
    }
}

/// A generated two-to-two reaction preserving `p1 + p2 -> p3 + p4` role semantics.
#[derive(Clone, Debug)]
pub struct GeneratedTwoToTwoReaction {
    p1: GeneratedParticle,
    p2: GeneratedParticle,
    p3: GeneratedParticle,
    p4: GeneratedParticle,
    tdist: MandelstamTDistribution,
    p1_p3_lab_dir: Vec3,
    p2_p3_lab_dir: Vec3,
}

impl GeneratedTwoToTwoReaction {
    /// Construct a generated two-to-two reaction.
    pub fn new(
        p1: GeneratedParticle,
        p2: GeneratedParticle,
        p3: GeneratedParticle,
        p4: GeneratedParticle,
        tdist: MandelstamTDistribution,
    ) -> LadduResult<Self> {
        validate_initial_role(&p1, "p1")?;
        validate_initial_role(&p2, "p2")?;
        validate_final_role(&p3, "p3")?;
        validate_final_role(&p4, "p4")?;
        let reaction = Self {
            p1,
            p2,
            p3,
            p4,
            tdist,
            p1_p3_lab_dir: Vec3::z(),
            p2_p3_lab_dir: -Vec3::z(),
        };
        reaction.validate()?;
        Ok(reaction)
    }

    fn validate(&self) -> LadduResult<()> {
        let mut seen = HashSet::new();
        for particle in [&self.p1, &self.p2, &self.p3, &self.p4] {
            particle.collect_ids(&mut seen)?;
            particle.validate_reconstruction()?;
        }
        Ok(())
    }

    fn p4_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();
        for particle in [&self.p1, &self.p2, &self.p3, &self.p4] {
            labels.append(&mut particle.p4_labels());
        }
        labels
    }

    fn layout_components(
        &self,
        storage: &GeneratedStorage,
    ) -> (Vec<GeneratedParticleLayout>, Vec<GeneratedVertexLayout>) {
        let mut particles = Vec::new();
        let mut vertices = vec![GeneratedVertexLayout {
            vertex_id: 0,
            kind: GeneratedVertexKind::Production,
            incoming_product_ids: Vec::new(),
            outgoing_product_ids: Vec::new(),
        }];
        let p1_id = self
            .p1
            .append_decay_layout(None, None, storage, &mut particles, &mut vertices);
        let p2_id = self
            .p2
            .append_decay_layout(None, None, storage, &mut particles, &mut vertices);
        let p3_id =
            self.p3
                .append_decay_layout(None, Some(0), storage, &mut particles, &mut vertices);
        let p4_id =
            self.p4
                .append_decay_layout(None, Some(0), storage, &mut particles, &mut vertices);
        vertices[0].incoming_product_ids = vec![p1_id, p2_id];
        vertices[0].outgoing_product_ids = vec![p3_id, p4_id];
        (particles, vertices)
    }

    fn particle_layouts(&self) -> Vec<GeneratedParticleLayout> {
        self.particle_layouts_with_storage(&GeneratedStorage::All)
    }

    fn particle_layouts_with_storage(
        &self,
        storage: &GeneratedStorage,
    ) -> Vec<GeneratedParticleLayout> {
        self.layout_components(storage).0
    }

    fn vertex_layouts(&self) -> Vec<GeneratedVertexLayout> {
        self.layout_components(&GeneratedStorage::All).1
    }

    fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        Reaction::two_to_two(
            &self.p1.generated_particle()?,
            &self.p2.generated_particle()?,
            &self.p3.generated_particle()?,
            &self.p4.generated_particle()?,
        )
    }

    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, Vec<Vec4>>) {
        let GeneratedParticle::Initial {
            id: p1_id,
            generator: p1_generator,
            ..
        } = &self.p1
        else {
            unreachable!("validated generated two-to-two p1 role")
        };
        let GeneratedParticle::Initial {
            id: p2_id,
            generator: p2_generator,
            ..
        } = &self.p2
        else {
            unreachable!("validated generated two-to-two p2 role")
        };

        let p1_e = p1_generator.energy_distribution.sample(rng);
        let p1_m = p1_generator.mass;
        let p1_msq = p1_m * p1_m;
        let p1_p4_lab = rng.p4(p1_m, p1_e, self.p1_p3_lab_dir);
        if let Some(storage) = p4_storage.get_mut(p1_id) {
            storage.push(p1_p4_lab)
        }

        let p2_e = p2_generator.energy_distribution.sample(rng);
        let p2_m = p2_generator.mass;
        let p2_msq = p2_m * p2_m;
        let p2_p4_lab = rng.p4(p2_m, p2_e, self.p2_p3_lab_dir);
        if let Some(storage) = p4_storage.get_mut(p2_id) {
            storage.push(p2_p4_lab)
        }

        let cm = p1_p4_lab + p2_p4_lab;
        let cm_boost = -cm.beta();
        let s = cm.mag2();
        let sqrt_s = s.sqrt();
        let t = self.tdist.sample(rng);

        let p1_p4_cm = p1_p4_lab.boost(&cm_boost);
        let p3_m = self.p3.sample_mass(rng);
        let p3_msq = p3_m * p3_m;
        let p4_m = self.p4.sample_mass(rng);
        let p4_msq = p4_m * p4_m;
        let p_in_mag = q_m(sqrt_s, p1_m, p2_m, Sheet::Physical).re;
        let p_out_mag = q_m(sqrt_s, p3_m, p4_m, Sheet::Physical).re;
        let p1_e_cm = (s + p1_msq - p2_msq) / (2.0 * sqrt_s);
        let p3_e_cm = (s + p3_msq - p4_msq) / (2.0 * sqrt_s);
        let p4_e_cm = (s + p4_msq - p3_msq) / (2.0 * sqrt_s);
        let costheta =
            (t - p1_msq - p3_msq + 2.0 * p1_e_cm * p3_e_cm) / (2.0 * p_in_mag * p_out_mag);
        let costheta = costheta.clamp(-1.0, 1.0);
        let sintheta = (1.0 - costheta * costheta).sqrt();
        let phi = rng.uniform(0.0, 2.0 * PI);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (x, y, z) = basis(p1_p4_cm.vec3());
        let p3_dir_cm = x * (sintheta * cos_phi) + y * (sintheta * sin_phi) + z * costheta;

        let p3_p4_cm = (p3_dir_cm * p_out_mag).with_energy(p3_e_cm);
        self.p3
            .generate_decay(rng, p3_p4_cm, &-cm_boost, p4_storage);
        let p4_p4_cm = (-p3_dir_cm * p_out_mag).with_energy(p4_e_cm);
        self.p4
            .generate_decay(rng, p4_p4_cm, &-cm_boost, p4_storage);
    }
}

fn validate_initial_role(particle: &GeneratedParticle, role: &str) -> LadduResult<()> {
    if matches!(particle, GeneratedParticle::Initial { .. }) {
        Ok(())
    } else {
        Err(LadduError::Custom(format!(
            "generated two-to-two role '{role}' requires an initial particle"
        )))
    }
}

fn validate_final_role(particle: &GeneratedParticle, role: &str) -> LadduResult<()> {
    if matches!(
        particle,
        GeneratedParticle::Stable { .. } | GeneratedParticle::Composite { .. }
    ) {
        Ok(())
    } else {
        Err(LadduError::Custom(format!(
            "generated two-to-two role '{role}' requires an outgoing particle"
        )))
    }
}

/// A generated reaction topology.
#[derive(Clone, Debug)]
pub enum GeneratedReactionTopology {
    /// A generated two-to-two topology.
    TwoToTwo(GeneratedTwoToTwoReaction),
}

impl GeneratedReactionTopology {
    fn p4_labels(&self) -> Vec<String> {
        match self {
            Self::TwoToTwo(reaction) => reaction.p4_labels(),
        }
    }

    fn particle_layouts(&self) -> Vec<GeneratedParticleLayout> {
        match self {
            Self::TwoToTwo(reaction) => reaction.particle_layouts(),
        }
    }

    fn particle_layouts_with_storage(
        &self,
        storage: &GeneratedStorage,
    ) -> Vec<GeneratedParticleLayout> {
        match self {
            Self::TwoToTwo(reaction) => reaction.particle_layouts_with_storage(storage),
        }
    }

    fn vertex_layouts(&self) -> Vec<GeneratedVertexLayout> {
        match self {
            Self::TwoToTwo(reaction) => reaction.vertex_layouts(),
        }
    }

    fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        match self {
            Self::TwoToTwo(reaction) => reaction.reconstructed_reaction(),
        }
    }

    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, Vec<Vec4>>) {
        match self {
            Self::TwoToTwo(reaction) => reaction.generate_event(rng, p4_storage),
        }
    }
}

/// A generated reaction layout.
#[derive(Clone, Debug)]
pub struct GeneratedReaction {
    topology: GeneratedReactionTopology,
}

impl GeneratedReaction {
    /// Construct a generated two-to-two reaction.
    pub fn two_to_two(
        p1: GeneratedParticle,
        p2: GeneratedParticle,
        p3: GeneratedParticle,
        p4: GeneratedParticle,
        tdist: MandelstamTDistribution,
    ) -> LadduResult<Self> {
        Ok(Self {
            topology: GeneratedReactionTopology::TwoToTwo(GeneratedTwoToTwoReaction::new(
                p1, p2, p3, p4, tdist,
            )?),
        })
    }

    /// Return generated p4 labels.
    pub fn p4_labels(&self) -> Vec<String> {
        self.topology.p4_labels()
    }

    /// Return generated particle layout entries in stable product-ID order.
    pub fn particle_layouts(&self) -> Vec<GeneratedParticleLayout> {
        self.topology.particle_layouts()
    }

    /// Return generated particle layout entries for a dataset storage policy.
    pub fn particle_layouts_with_storage(
        &self,
        storage: &GeneratedStorage,
    ) -> Vec<GeneratedParticleLayout> {
        self.topology.particle_layouts_with_storage(storage)
    }

    /// Return generated vertex layout entries in stable vertex-ID order.
    pub fn vertex_layouts(&self) -> Vec<GeneratedVertexLayout> {
        self.topology.vertex_layouts()
    }

    /// Build the reconstructed reaction corresponding to this generated layout.
    pub fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        self.topology.reconstructed_reaction()
    }

    fn generate(
        &self,
        rng: &mut Rng,
        p4_storage: &mut HashMap<String, Vec<Vec4>>,
        n_events: usize,
    ) {
        for _ in 0..n_events {
            self.topology.generate_event(rng, p4_storage);
        }
    }
}

/// Metadata for one generated particle in a generated event layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedParticleLayout {
    id: String,
    product_id: usize,
    parent_id: Option<usize>,
    species: Option<ParticleSpecies>,
    p4_label: Option<String>,
    produced_vertex_id: Option<usize>,
    decay_vertex_id: Option<usize>,
}

impl GeneratedParticleLayout {
    /// Return the generated particle identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Return the zero-based stable product ID in generated-layout order.
    pub fn product_id(&self) -> usize {
        self.product_id
    }

    /// Return the decay-parent product ID, if this particle is a decay daughter.
    pub fn parent_id(&self) -> Option<usize> {
        self.parent_id
    }

    /// Return optional species metadata associated with this generated particle.
    pub fn species(&self) -> Option<&ParticleSpecies> {
        self.species.as_ref()
    }

    /// Return the dataset p4 label associated with this particle, if stored in the batch.
    pub fn p4_label(&self) -> Option<&str> {
        self.p4_label.as_deref()
    }

    /// Return the vertex ID where this particle was produced, if any.
    pub fn produced_vertex_id(&self) -> Option<usize> {
        self.produced_vertex_id
    }

    /// Return the vertex ID where this particle decays, if it is a generated parent.
    pub fn decay_vertex_id(&self) -> Option<usize> {
        self.decay_vertex_id
    }
}

/// The semantic kind of a generated vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedVertexKind {
    /// A production vertex connecting initial-state particles to outgoing products.
    Production,
    /// A decay vertex connecting one generated parent to generated daughters.
    Decay,
}

/// Metadata for one generated vertex in a generated event layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedVertexLayout {
    vertex_id: usize,
    kind: GeneratedVertexKind,
    incoming_product_ids: Vec<usize>,
    outgoing_product_ids: Vec<usize>,
}

impl GeneratedVertexLayout {
    /// Return the zero-based stable vertex ID in generated-layout order.
    pub fn vertex_id(&self) -> usize {
        self.vertex_id
    }

    /// Return the semantic vertex kind.
    pub fn kind(&self) -> GeneratedVertexKind {
        self.kind
    }

    /// Return product IDs entering this vertex.
    pub fn incoming_product_ids(&self) -> &[usize] {
        &self.incoming_product_ids
    }

    /// Return product IDs leaving this vertex.
    pub fn outgoing_product_ids(&self) -> &[usize] {
        &self.outgoing_product_ids
    }
}

/// Metadata describing the columns and generated particles in a generated event batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedEventLayout {
    p4_labels: Vec<String>,
    aux_labels: Vec<String>,
    particles: Vec<GeneratedParticleLayout>,
    vertices: Vec<GeneratedVertexLayout>,
}

impl GeneratedEventLayout {
    /// Construct generated event layout metadata from p4 and auxiliary labels.
    pub fn new(
        p4_labels: Vec<String>,
        aux_labels: Vec<String>,
        particles: Vec<GeneratedParticleLayout>,
        vertices: Vec<GeneratedVertexLayout>,
    ) -> Self {
        Self {
            p4_labels,
            aux_labels,
            particles,
            vertices,
        }
    }

    /// Return generated p4 column labels in dataset order.
    pub fn p4_labels(&self) -> &[String] {
        &self.p4_labels
    }

    /// Return generated auxiliary column labels in dataset order.
    pub fn aux_labels(&self) -> &[String] {
        &self.aux_labels
    }

    /// Return generated particle layout entries in stable product-ID order.
    pub fn particles(&self) -> &[GeneratedParticleLayout] {
        &self.particles
    }

    /// Return generated vertex layout entries in stable vertex-ID order.
    pub fn vertices(&self) -> &[GeneratedVertexLayout] {
        &self.vertices
    }
}

/// A generated dataset batch plus the metadata needed to interpret it.
#[derive(Clone, Debug)]
pub struct GeneratedBatch {
    dataset: Dataset,
    reaction: GeneratedReaction,
    layout: GeneratedEventLayout,
}

impl GeneratedBatch {
    /// Construct a generated batch.
    pub fn new(
        dataset: Dataset,
        reaction: GeneratedReaction,
        layout: GeneratedEventLayout,
    ) -> Self {
        Self {
            dataset,
            reaction,
            layout,
        }
    }

    /// Borrow the generated dataset.
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    /// Consume this batch and return the generated dataset.
    pub fn into_dataset(self) -> Dataset {
        self.dataset
    }

    /// Borrow the generated reaction metadata.
    pub fn reaction(&self) -> &GeneratedReaction {
        &self.reaction
    }

    /// Borrow the generated event layout metadata.
    pub fn layout(&self) -> &GeneratedEventLayout {
        &self.layout
    }
}

/// Event generator for generated reactions.
#[derive(Clone, Debug)]
pub struct EventGenerator {
    reaction: GeneratedReaction,
    aux_generators: HashMap<String, Distribution>,
    storage: GeneratedStorage,
    seed: u64,
}

/// Finite iterator over generated dataset batches.
#[derive(Clone, Debug)]
pub struct GeneratedBatchIter {
    generator: EventGenerator,
    remaining_events: usize,
    batch_size: usize,
    rng: Rng,
}

impl Iterator for GeneratedBatchIter {
    type Item = LadduResult<GeneratedBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_events == 0 {
            return None;
        }
        let n_events = self.batch_size.min(self.remaining_events);
        self.remaining_events -= n_events;
        Some(
            self.generator
                .generate_batch_with_rng(n_events, &mut self.rng),
        )
    }
}

/// Evaluates unnormalized intensities for generated batches.
pub trait BatchIntensity {
    /// Return one nonnegative finite intensity for each event in `batch`.
    fn evaluate(&mut self, batch: &GeneratedBatch) -> LadduResult<Vec<f64>>;
}

impl<F> BatchIntensity for F
where
    F: FnMut(&GeneratedBatch) -> LadduResult<Vec<f64>>,
{
    fn evaluate(&mut self, batch: &GeneratedBatch) -> LadduResult<Vec<f64>> {
        self(batch)
    }
}

/// Envelope strategy used by rejection sampling.
#[derive(Clone, Debug)]
pub enum RejectionEnvelope {
    /// Use a fixed maximum event weight.
    Fixed {
        /// Maximum event weight used as the rejection envelope.
        max_weight: f64,
    },
}

impl RejectionEnvelope {
    fn max_weight(&self) -> f64 {
        match self {
            Self::Fixed { max_weight } => *max_weight,
        }
    }
}

/// Options for rejection sampling generated events.
#[derive(Clone, Debug)]
pub struct RejectionSamplingOptions {
    /// Number of accepted events to produce.
    pub target_accepted: usize,
    /// Number of raw events to generate per source batch.
    pub generation_batch_size: usize,
    /// Target number of accepted events emitted per output batch.
    pub output_batch_size: usize,
    /// Envelope used by the rejection sampler.
    pub envelope: RejectionEnvelope,
    /// Random seed used for accept/reject decisions.
    pub seed: u64,
}

/// Rejection-sampling diagnostics accumulated while sampling.
#[derive(Clone, Debug, Default)]
pub struct RejectionSamplingDiagnostics {
    /// Number of generated events inspected.
    pub generated_events: usize,
    /// Number of events accepted.
    pub accepted_events: usize,
    /// Number of events rejected.
    pub rejected_events: usize,
    /// Maximum observed event intensity.
    pub max_observed_weight: f64,
    /// Envelope maximum used for rejection sampling.
    pub envelope_max_weight: f64,
    /// Number of fixed-envelope violations observed.
    pub envelope_violations: usize,
}

impl RejectionSamplingDiagnostics {
    /// Fraction of generated events accepted.
    pub fn acceptance_efficiency(&self) -> f64 {
        if self.generated_events == 0 {
            0.0
        } else {
            self.accepted_events as f64 / self.generated_events as f64
        }
    }
}

/// Rejection sampler over generated batches.
#[derive(Clone, Debug)]
pub struct RejectionSampler<I> {
    generator: EventGenerator,
    intensity: I,
    options: RejectionSamplingOptions,
}

impl<I> RejectionSampler<I>
where
    I: BatchIntensity,
{
    /// Construct a rejection sampler.
    pub fn new(
        generator: EventGenerator,
        intensity: I,
        options: RejectionSamplingOptions,
    ) -> LadduResult<Self> {
        if options.generation_batch_size == 0 {
            return Err(LadduError::Custom(
                "generation_batch_size must be greater than zero".to_string(),
            ));
        }
        if options.output_batch_size == 0 {
            return Err(LadduError::Custom(
                "output_batch_size must be greater than zero".to_string(),
            ));
        }
        let max_weight = options.envelope.max_weight();
        if !max_weight.is_finite() || max_weight <= 0.0 {
            return Err(LadduError::Custom(
                "rejection envelope max_weight must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            generator,
            intensity,
            options,
        })
    }

    /// Consume this sampler and return an iterator over accepted generated batches.
    pub fn accepted_batches(self) -> RejectionSampleIter<I> {
        let envelope_max_weight = self.options.envelope.max_weight();
        RejectionSampleIter {
            generation_rng: Rng::with_seed(self.generator.seed),
            rejection_rng: Rng::with_seed(self.options.seed),
            diagnostics: RejectionSamplingDiagnostics {
                envelope_max_weight,
                ..Default::default()
            },
            sampler: self,
            current_batch: None,
            current_intensities: Vec::new(),
            current_index: 0,
        }
    }
}

/// Iterator over accepted generated batches.
#[derive(Clone, Debug)]
pub struct RejectionSampleIter<I> {
    sampler: RejectionSampler<I>,
    generation_rng: Rng,
    rejection_rng: Rng,
    diagnostics: RejectionSamplingDiagnostics,
    current_batch: Option<GeneratedBatch>,
    current_intensities: Vec<f64>,
    current_index: usize,
}

impl<I> RejectionSampleIter<I> {
    /// Borrow rejection-sampling diagnostics accumulated so far.
    pub fn diagnostics(&self) -> &RejectionSamplingDiagnostics {
        &self.diagnostics
    }
}

impl<I> RejectionSampleIter<I>
where
    I: BatchIntensity,
{
    fn load_next_source_batch(&mut self) -> LadduResult<()> {
        let batch = self.sampler.generator.generate_batch_with_rng(
            self.sampler.options.generation_batch_size,
            &mut self.generation_rng,
        )?;
        let intensities = self.sampler.intensity.evaluate(&batch)?;
        if intensities.len() != batch.dataset().n_events() {
            return Err(LadduError::Custom(format!(
                "intensity length mismatch: expected {}, got {}",
                batch.dataset().n_events(),
                intensities.len()
            )));
        }
        self.diagnostics.generated_events += batch.dataset().n_events();
        self.current_batch = Some(batch);
        self.current_intensities = intensities;
        self.current_index = 0;
        Ok(())
    }

    fn empty_output_batch(source: &GeneratedBatch) -> GeneratedBatch {
        GeneratedBatch::new(
            Dataset::empty(source.dataset().metadata().clone()),
            source.reaction().clone(),
            source.layout().clone(),
        )
    }
}

impl<I> Iterator for RejectionSampleIter<I>
where
    I: BatchIntensity,
{
    type Item = LadduResult<GeneratedBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.diagnostics.accepted_events >= self.sampler.options.target_accepted {
            return None;
        }

        let mut output: Option<GeneratedBatch> = None;
        while self.diagnostics.accepted_events < self.sampler.options.target_accepted {
            let needs_batch = self
                .current_batch
                .as_ref()
                .map(|batch| self.current_index >= batch.dataset().n_events())
                .unwrap_or(true);
            if needs_batch {
                if let Err(err) = self.load_next_source_batch() {
                    return Some(Err(err));
                }
            }

            let source = self
                .current_batch
                .as_ref()
                .expect("source batch should be loaded");
            if output.is_none() {
                output = Some(Self::empty_output_batch(source));
            }

            let weight = self.current_intensities[self.current_index];
            if !weight.is_finite() || weight < 0.0 {
                return Some(Err(LadduError::Custom(format!(
                    "intensity at event {} must be finite and nonnegative, got {weight}",
                    self.current_index
                ))));
            }
            self.diagnostics.max_observed_weight = self.diagnostics.max_observed_weight.max(weight);
            let envelope_max = self.sampler.options.envelope.max_weight();
            if weight > envelope_max {
                self.diagnostics.envelope_violations += 1;
                return Some(Err(LadduError::Custom(format!(
                    "rejection envelope violation: observed weight {weight} exceeds max_weight {envelope_max}"
                ))));
            }

            let accepted = self.rejection_rng.f64() * envelope_max < weight;
            if accepted {
                let event = match source.dataset().event(self.current_index) {
                    Ok(event) => event,
                    Err(err) => return Some(Err(err)),
                };
                if let Err(err) = output.as_mut().unwrap().dataset.push_event(
                    event.p4s.clone(),
                    event.aux.clone(),
                    event.weight,
                ) {
                    return Some(Err(err));
                }
                self.diagnostics.accepted_events += 1;
            } else {
                self.diagnostics.rejected_events += 1;
            }
            self.current_index += 1;

            if output.as_ref().unwrap().dataset().n_events()
                >= self.sampler.options.output_batch_size
                || self.diagnostics.accepted_events >= self.sampler.options.target_accepted
            {
                break;
            }
        }

        output
            .filter(|batch| batch.dataset().n_events() > 0)
            .map(Ok)
    }
}

impl EventGenerator {
    /// Construct an event generator.
    pub fn new(
        reaction: GeneratedReaction,
        aux_generators: HashMap<String, Distribution>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            reaction,
            aux_generators,
            storage: GeneratedStorage::All,
            seed: seed.unwrap_or_else(|| fastrand::u64(..)),
        }
    }

    /// Return the generated p4 storage policy.
    pub fn storage(&self) -> &GeneratedStorage {
        &self.storage
    }

    /// Return a copy of this generator with a generated p4 storage policy.
    pub fn with_storage(mut self, storage: GeneratedStorage) -> LadduResult<Self> {
        storage.validate(&self.reaction.p4_labels())?;
        self.storage = storage;
        Ok(self)
    }

    fn aux_entries(&self) -> Vec<(&String, &Distribution)> {
        let mut aux_entries = self.aux_generators.iter().collect::<Vec<_>>();
        aux_entries.sort_by_key(|(label, _)| *label);
        aux_entries
    }

    fn generate_batch_with_rng(
        &self,
        n_events: usize,
        rng: &mut Rng,
    ) -> LadduResult<GeneratedBatch> {
        let all_p4_labels = self.reaction.p4_labels();
        self.storage.validate(&all_p4_labels)?;
        let p4_labels = self.storage.stored_labels(&all_p4_labels);
        let aux_entries = self.aux_entries();
        let aux_labels = aux_entries
            .iter()
            .map(|(label, _)| (*label).clone())
            .collect::<Vec<_>>();
        let mut p4_data: HashMap<String, Vec<Vec4>> = p4_labels
            .iter()
            .map(|label| (label.clone(), Vec::with_capacity(n_events)))
            .collect();
        let metadata = DatasetMetadata::new(p4_labels.clone(), aux_labels.clone())?;
        let mut aux: Vec<Vec<f64>> = aux_entries
            .iter()
            .map(|_| Vec::with_capacity(n_events))
            .collect();
        let weights = vec![1.0; n_events];
        for _ in 0..n_events {
            for ((_, distribution), column) in aux_entries.iter().zip(aux.iter_mut()) {
                column.push(distribution.sample(rng));
            }
            self.reaction.generate(rng, &mut p4_data, 1);
        }
        let p4 = p4_labels
            .iter()
            .filter_map(|label| p4_data.remove(label))
            .collect();
        let dataset = Dataset::from_columns(metadata, p4, aux, weights)?;
        Ok(GeneratedBatch::new(
            dataset,
            self.reaction.clone(),
            GeneratedEventLayout::new(
                p4_labels,
                aux_labels,
                self.reaction.particle_layouts_with_storage(&self.storage),
                self.reaction.vertex_layouts(),
            ),
        ))
    }

    /// Generate one dataset batch with generated layout metadata.
    pub fn generate_batch(&self, n_events: usize) -> LadduResult<GeneratedBatch> {
        let mut rng = Rng::with_seed(self.seed);
        self.generate_batch_with_rng(n_events, &mut rng)
    }

    /// Generate a finite iterator over batches.
    ///
    /// The iterator advances one RNG stream, so concatenating all yielded batches is
    /// deterministic and matches [`EventGenerator::generate_dataset`] for the same total count.
    pub fn generate_batches(
        &self,
        total_events: usize,
        batch_size: usize,
    ) -> LadduResult<GeneratedBatchIter> {
        if batch_size == 0 {
            return Err(LadduError::Custom(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        Ok(GeneratedBatchIter {
            generator: self.clone(),
            remaining_events: total_events,
            batch_size,
            rng: Rng::with_seed(self.seed),
        })
    }

    /// Generate a dataset.
    pub fn generate_dataset(&self, n_events: usize) -> LadduResult<Dataset> {
        Ok(self.generate_batch(n_events)?.into_dataset())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use laddu_core::{traits::Variable, Channel, Frame};

    use super::*;

    fn demo_reaction() -> GeneratedReaction {
        let beam = GeneratedParticle::initial(
            "beam",
            InitialGenerator::beam_with_fixed_energy(0.0, 8.0),
            Reconstruction::Stored,
        );
        let target = GeneratedParticle::initial(
            "target",
            InitialGenerator::target(0.938272),
            Reconstruction::Missing,
        );
        let ks1 = GeneratedParticle::stable(
            "kshort1",
            StableGenerator::new(0.497611),
            Reconstruction::Stored,
        );
        let ks2 = GeneratedParticle::stable(
            "kshort2",
            StableGenerator::new(0.497611),
            Reconstruction::Stored,
        );
        let kk = GeneratedParticle::composite(
            "kk",
            CompositeGenerator::new(1.1, 1.6),
            (&ks1, &ks2),
            Reconstruction::Composite,
        );
        let recoil = GeneratedParticle::stable(
            "recoil",
            StableGenerator::new(0.938272),
            Reconstruction::Stored,
        );
        let tdist = MandelstamTDistribution::Exponential { slope: 0.1 };
        GeneratedReaction::two_to_two(beam, target, kk, recoil, tdist).unwrap()
    }

    #[test]
    fn test_generation() {
        let reaction = demo_reaction();
        let generator = EventGenerator::new(reaction, HashMap::new(), Some(12345));
        let n_events = 1_000;
        let dataset = generator.generate_dataset(n_events).unwrap();
        assert_eq!(dataset.n_events(), n_events);
        let metadata = dataset.metadata();
        assert!(metadata.p4_index("beam").is_some());
        assert!(metadata.p4_index("target").is_some());
        assert!(metadata.p4_index("kk").is_some());
        assert!(metadata.p4_index("kshort1").is_some());
        assert!(metadata.p4_index("kshort2").is_some());
        assert!(metadata.p4_index("recoil").is_some());

        for event in dataset {
            let beam_p4 = event.p4("beam").unwrap();
            let target_p4 = event.p4("target").unwrap();
            let kk_p4 = event.p4("kk").unwrap();
            let kshort1_p4 = event.p4("kshort1").unwrap();
            let kshort2_p4 = event.p4("kshort2").unwrap();
            let recoil_p4 = event.p4("recoil").unwrap();

            assert!(beam_p4.e().is_finite());
            assert!(target_p4.e().is_finite());
            assert!(kk_p4.e().is_finite());
            assert!(kshort1_p4.e().is_finite());
            assert!(kshort2_p4.e().is_finite());
            assert!(recoil_p4.e().is_finite());

            assert_relative_eq!(kk_p4, kshort1_p4 + kshort2_p4, epsilon = 1e-10);
            assert_relative_eq!(beam_p4 + target_p4, kk_p4 + recoil_p4, epsilon = 1e-10);
            assert_relative_eq!(kshort1_p4.m(), 0.497611, epsilon = 1e-10);
            assert_relative_eq!(kshort2_p4.m(), 0.497611, epsilon = 1e-10);
            assert_relative_eq!(recoil_p4.m(), 0.938272, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reconstructed_reaction() {
        let generated = demo_reaction();
        let reaction = generated.reconstructed_reaction().unwrap();
        let dataset = EventGenerator::new(generated, HashMap::new(), Some(12345))
            .generate_dataset(4)
            .unwrap();
        let mass = reaction.mass("kk").value_on(&dataset).unwrap();
        let angles = reaction
            .decay("kk")
            .unwrap()
            .angles("kshort1", Frame::Helicity)
            .unwrap();
        let mandelstam = reaction
            .mandelstam(Channel::S)
            .unwrap()
            .value_on(&dataset)
            .unwrap();

        assert_eq!(mass.len(), 4);
        assert_eq!(
            angles.costheta.to_string(),
            "CosTheta(parent=kk, daughter=kshort1, frame=Helicity)"
        );
        assert_eq!(mandelstam.len(), 4);
    }

    #[test]
    fn test_generated_batch_metadata() {
        let generated = demo_reaction();
        let generator = EventGenerator::new(
            generated,
            HashMap::from([("pol_angle".to_string(), Distribution::Fixed(0.25))]),
            Some(12345),
        );
        let batch = generator.generate_batch(4).unwrap();

        assert_eq!(batch.dataset().n_events(), 4);
        assert_eq!(
            batch.layout().p4_labels(),
            &["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
        );
        assert_eq!(batch.layout().aux_labels(), &["pol_angle"]);
        assert_eq!(
            batch.reaction().p4_labels(),
            vec!["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
        );
        assert_eq!(batch.dataset().p4_names(), batch.layout().p4_labels());
        assert_eq!(batch.dataset().aux_names(), batch.layout().aux_labels());
        let particles = batch.layout().particles();
        assert_eq!(particles.len(), 6);
        assert_eq!(particles[0].id(), "beam");
        assert_eq!(particles[0].product_id(), 0);
        assert_eq!(particles[0].parent_id(), None);
        assert_eq!(particles[0].produced_vertex_id(), None);
        assert_eq!(particles[0].decay_vertex_id(), None);
        assert_eq!(particles[1].id(), "target");
        assert_eq!(particles[1].parent_id(), None);
        assert_eq!(particles[1].produced_vertex_id(), None);
        assert_eq!(particles[1].decay_vertex_id(), None);
        assert_eq!(particles[2].id(), "kk");
        assert_eq!(particles[2].product_id(), 2);
        assert_eq!(particles[2].parent_id(), None);
        assert_eq!(particles[2].produced_vertex_id(), Some(0));
        assert_eq!(particles[2].decay_vertex_id(), Some(1));
        assert_eq!(particles[3].id(), "kshort1");
        assert_eq!(particles[3].parent_id(), Some(2));
        assert_eq!(particles[3].produced_vertex_id(), Some(1));
        assert_eq!(particles[3].decay_vertex_id(), None);
        assert_eq!(particles[4].id(), "kshort2");
        assert_eq!(particles[4].parent_id(), Some(2));
        assert_eq!(particles[4].produced_vertex_id(), Some(1));
        assert_eq!(particles[4].decay_vertex_id(), None);
        assert_eq!(particles[5].id(), "recoil");
        assert_eq!(particles[5].parent_id(), None);
        assert_eq!(particles[5].produced_vertex_id(), Some(0));
        assert_eq!(particles[5].decay_vertex_id(), None);
        for particle in particles {
            assert_eq!(particle.p4_label(), Some(particle.id()));
        }
        let vertices = batch.layout().vertices();
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0].vertex_id(), 0);
        assert_eq!(vertices[0].kind(), GeneratedVertexKind::Production);
        assert_eq!(vertices[0].incoming_product_ids(), &[0, 1]);
        assert_eq!(vertices[0].outgoing_product_ids(), &[2, 5]);
        assert_eq!(vertices[1].vertex_id(), 1);
        assert_eq!(vertices[1].kind(), GeneratedVertexKind::Decay);
        assert_eq!(vertices[1].incoming_product_ids(), &[2]);
        assert_eq!(vertices[1].outgoing_product_ids(), &[3, 4]);
    }

    #[test]
    fn generated_storage_only_projects_dataset_columns() {
        let generated = demo_reaction();
        let generator = EventGenerator::new(generated, HashMap::new(), Some(12345))
            .with_storage(GeneratedStorage::only([
                "beam", "target", "kshort1", "kshort2", "recoil",
            ]))
            .unwrap();
        let batch = generator.generate_batch(4).unwrap();

        assert_eq!(
            batch.reaction().p4_labels(),
            vec!["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
        );
        assert_eq!(
            batch.layout().p4_labels(),
            &["beam", "target", "kshort1", "kshort2", "recoil"]
        );
        assert_eq!(batch.dataset().p4_names(), batch.layout().p4_labels());
        assert!(batch.dataset().metadata().p4_index("kk").is_none());

        let particles = batch.layout().particles();
        assert_eq!(particles.len(), 6);
        assert_eq!(particles[2].id(), "kk");
        assert_eq!(particles[2].p4_label(), None);
        assert_eq!(particles[3].p4_label(), Some("kshort1"));
        assert_eq!(particles[4].p4_label(), Some("kshort2"));

        for index in 0..batch.dataset().n_events() {
            let event = batch.dataset().event(index).unwrap();
            assert_relative_eq!(
                event.p4("beam").unwrap() + event.p4("target").unwrap(),
                event.p4("kshort1").unwrap()
                    + event.p4("kshort2").unwrap()
                    + event.p4("recoil").unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn generated_storage_rejects_unknown_and_duplicate_ids() {
        assert!(
            EventGenerator::new(demo_reaction(), HashMap::new(), Some(12345))
                .with_storage(GeneratedStorage::only(["beam", "does_not_exist"]))
                .is_err()
        );
        assert!(
            EventGenerator::new(demo_reaction(), HashMap::new(), Some(12345))
                .with_storage(GeneratedStorage::only(["beam", "beam"]))
                .is_err()
        );
    }

    #[test]
    fn generated_species_metadata_propagates_to_layout() {
        let beam = GeneratedParticle::initial(
            "beam",
            InitialGenerator::beam_with_fixed_energy(0.0, 8.0),
            Reconstruction::Stored,
        )
        .with_species(ParticleSpecies::code(22));
        let target = GeneratedParticle::initial(
            "target",
            InitialGenerator::target(0.938272),
            Reconstruction::Missing,
        )
        .with_species(ParticleSpecies::with_namespace("pdg", 2212));
        let kshort1 = GeneratedParticle::stable(
            "kshort1",
            StableGenerator::new(0.497611),
            Reconstruction::Stored,
        )
        .with_species(ParticleSpecies::label("KShort"));
        let kshort2 = GeneratedParticle::stable(
            "kshort2",
            StableGenerator::new(0.497611),
            Reconstruction::Stored,
        )
        .with_species(ParticleSpecies::label("KShort"));
        let kk = GeneratedParticle::composite(
            "kk",
            CompositeGenerator::new(1.1, 1.6),
            (&kshort1, &kshort2),
            Reconstruction::Composite,
        )
        .with_species(ParticleSpecies::label("KK"));
        let recoil = GeneratedParticle::stable(
            "recoil",
            StableGenerator::new(0.938272),
            Reconstruction::Stored,
        )
        .with_species(ParticleSpecies::code(2212));
        let reaction = GeneratedReaction::two_to_two(
            beam,
            target,
            kk,
            recoil,
            MandelstamTDistribution::Exponential { slope: 0.1 },
        )
        .unwrap();
        let particles = reaction.particle_layouts();

        assert_eq!(particles[0].species(), Some(&ParticleSpecies::code(22)));
        assert_eq!(
            particles[1].species(),
            Some(&ParticleSpecies::with_namespace("pdg", 2212))
        );
        assert_eq!(particles[2].species(), Some(&ParticleSpecies::label("KK")));
        assert_eq!(
            particles[3].species(),
            Some(&ParticleSpecies::label("KShort"))
        );
        assert_eq!(
            particles[4].species(),
            Some(&ParticleSpecies::label("KShort"))
        );
        assert_eq!(particles[5].species(), Some(&ParticleSpecies::code(2212)));
    }

    #[test]
    fn generated_batches_match_one_shot_generation() {
        let generated = demo_reaction();
        let generator = EventGenerator::new(
            generated,
            HashMap::from([(
                "pol_angle".to_string(),
                Distribution::Uniform { min: 0.0, max: 1.0 },
            )]),
            Some(12345),
        );
        let one_shot = generator.generate_dataset(7).unwrap();
        let batches = generator
            .generate_batches(7, 3)
            .unwrap()
            .collect::<LadduResult<Vec<_>>>()
            .unwrap();
        let batch_sizes = batches
            .iter()
            .map(|batch| batch.dataset().n_events())
            .collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![3, 3, 1]);

        let mut offset = 0;
        for batch in batches {
            for local_index in 0..batch.dataset().n_events() {
                let expected = one_shot.event(offset + local_index).unwrap();
                let actual = batch.dataset().event(local_index).unwrap();
                for name in one_shot.p4_names() {
                    assert_relative_eq!(
                        actual.p4(name).unwrap(),
                        expected.p4(name).unwrap(),
                        epsilon = 1e-10
                    );
                }
                for aux_index in 0..one_shot.aux_names().len() {
                    assert_relative_eq!(actual.aux[aux_index], expected.aux[aux_index]);
                }
                assert_relative_eq!(actual.weight(), expected.weight());
            }
            offset += batch.dataset().n_events();
        }
        assert_eq!(offset, one_shot.n_events());
        assert!(generator.generate_batches(1, 0).is_err());
    }

    #[test]
    fn fixed_envelope_rejection_sampler_streams_accepted_batches() {
        let generator = EventGenerator::new(demo_reaction(), HashMap::new(), Some(12345));
        let sampler = RejectionSampler::new(
            generator,
            |batch: &GeneratedBatch| Ok(vec![1.0; batch.dataset().n_events()]),
            RejectionSamplingOptions {
                target_accepted: 5,
                generation_batch_size: 4,
                output_batch_size: 2,
                envelope: RejectionEnvelope::Fixed { max_weight: 1.0 },
                seed: 67890,
            },
        )
        .unwrap();

        let mut iter = sampler.accepted_batches();
        let mut accepted_batches = Vec::new();
        for batch in iter.by_ref() {
            accepted_batches.push(batch.unwrap());
        }
        assert_eq!(
            accepted_batches
                .iter()
                .map(|batch| batch.dataset().n_events())
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(iter.diagnostics().generated_events, 8);
        assert_eq!(iter.diagnostics().accepted_events, 5);
        assert_eq!(iter.diagnostics().rejected_events, 0);
        assert_relative_eq!(iter.diagnostics().acceptance_efficiency(), 5.0 / 8.0);
        for batch in accepted_batches {
            assert_eq!(
                batch.layout().p4_labels(),
                &["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
            );
        }
    }

    #[test]
    fn fixed_envelope_rejection_sampler_rejects_violations() {
        let generator = EventGenerator::new(demo_reaction(), HashMap::new(), Some(12345));
        let sampler = RejectionSampler::new(
            generator,
            |batch: &GeneratedBatch| Ok(vec![2.0; batch.dataset().n_events()]),
            RejectionSamplingOptions {
                target_accepted: 1,
                generation_batch_size: 1,
                output_batch_size: 1,
                envelope: RejectionEnvelope::Fixed { max_weight: 1.0 },
                seed: 67890,
            },
        )
        .unwrap();

        let mut iter = sampler.accepted_batches();
        let err = iter.next().expect("sampler should produce an error");
        assert!(err.is_err());
        assert_eq!(iter.diagnostics().envelope_violations, 1);
        assert_relative_eq!(iter.diagnostics().max_observed_weight, 2.0);
    }

    #[test]
    fn duplicate_generated_particle_ids_are_rejected() {
        let beam = GeneratedParticle::initial(
            "beam",
            InitialGenerator::beam_with_fixed_energy(0.0, 8.0),
            Reconstruction::Stored,
        );
        let target = GeneratedParticle::initial(
            "target",
            InitialGenerator::target(0.938272),
            Reconstruction::Missing,
        );
        let duplicate = GeneratedParticle::stable(
            "beam",
            StableGenerator::new(0.497611),
            Reconstruction::Stored,
        );
        let recoil = GeneratedParticle::stable(
            "recoil",
            StableGenerator::new(0.938272),
            Reconstruction::Stored,
        );

        assert!(GeneratedReaction::two_to_two(
            beam,
            target,
            duplicate,
            recoil,
            MandelstamTDistribution::Exponential { slope: 0.1 },
        )
        .is_err());
    }
}
