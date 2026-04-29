use std::collections::{HashMap, HashSet};

use fastrand::Rng;
use laddu_core::{
    data::{ColumnarP4Column, DatasetStorage},
    math::{q_m, Histogram, Sheet},
    Dataset, DatasetMetadata, LadduError, LadduResult, Particle, Reaction, Vec3, Vec4, PI,
};

use crate::distributions::{
    Distribution, HistogramSampler, LadduGenRngExt, MandelstamTDistribution, SimpleDistribution,
};

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
    pub fn beam_with_energy_histogram(mass: f64, energy: Histogram) -> Self {
        debug_assert!(
            mass >= 0.0,
            "Mass must be positive and greater than zero!\nMass: {}",
            mass
        );
        let sampler = HistogramSampler::new(energy);
        debug_assert!(
            sampler.hist.bin_edges[0] >= mass,
            "Mass cannot be greater than the minimum allowed energy!\nMass: {}\nMinimum Energy: {}",
            mass,
            sampler.hist.bin_edges[0]
        );
        Self {
            mass,
            energy_distribution: SimpleDistribution::Histogram(sampler),
        }
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
    },
    /// A stable generated particle.
    Stable {
        id: String,
        generator: StableGenerator,
        reconstruction: Reconstruction,
    },
    /// A generated composite particle with exactly two generated daughters.
    Composite {
        id: String,
        generator: CompositeGenerator,
        daughters: (Box<GeneratedParticle>, Box<GeneratedParticle>),
        reconstruction: Reconstruction,
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
        }
    }

    /// Return the generated particle ID.
    pub fn id(&self) -> &str {
        match self {
            Self::Initial { id, .. } | Self::Stable { id, .. } | Self::Composite { id, .. } => id,
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
        p4_storage: &mut HashMap<String, ColumnarP4Column>,
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

    fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        Reaction::two_to_two(
            &self.p1.generated_particle()?,
            &self.p2.generated_particle()?,
            &self.p3.generated_particle()?,
            &self.p4.generated_particle()?,
        )
    }

    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, ColumnarP4Column>) {
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

    fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        match self {
            Self::TwoToTwo(reaction) => reaction.reconstructed_reaction(),
        }
    }

    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, ColumnarP4Column>) {
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

    /// Build the reconstructed reaction corresponding to this generated layout.
    pub fn reconstructed_reaction(&self) -> LadduResult<Reaction> {
        self.topology.reconstructed_reaction()
    }

    fn generate(
        &self,
        rng: &mut Rng,
        p4_storage: &mut HashMap<String, ColumnarP4Column>,
        n_events: usize,
    ) {
        for _ in 0..n_events {
            self.topology.generate_event(rng, p4_storage);
        }
    }
}

/// Event generator for generated reactions.
#[derive(Clone, Debug)]
pub struct EventGenerator {
    reaction: GeneratedReaction,
    aux_generators: HashMap<String, Distribution>,
    seed: u64,
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
            seed: seed.unwrap_or_else(|| fastrand::u64(..)),
        }
    }

    /// Generate a dataset.
    pub fn generate_dataset(&self, n_events: usize) -> LadduResult<Dataset> {
        let p4_labels = self.reaction.p4_labels();
        let mut p4_data: HashMap<String, ColumnarP4Column> = p4_labels
            .iter()
            .map(|label| (label.clone(), ColumnarP4Column::with_capacity(n_events)))
            .collect();
        let metadata = DatasetMetadata::new(
            p4_labels.clone(),
            self.aux_generators.keys().cloned().collect(),
        )?;
        let mut rng = Rng::with_seed(self.seed);
        let aux: Vec<Vec<f64>> = self
            .aux_generators
            .values()
            .map(|d| (0..n_events).map(|_| d.sample(&mut rng)).collect())
            .collect();
        let weights = vec![1.0; n_events];
        self.reaction.generate(&mut rng, &mut p4_data, n_events);
        let p4 = p4_labels
            .iter()
            .filter_map(|label| p4_data.remove(label))
            .collect();
        Ok(DatasetStorage::new(metadata, p4, aux, weights).to_dataset())
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
