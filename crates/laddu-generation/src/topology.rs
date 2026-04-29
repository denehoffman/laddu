use std::collections::HashMap;

use fastrand::Rng;
use laddu_core::{
    data::{ColumnarP4Column, DatasetStorage},
    math::{q_m, Histogram, Sheet},
    Dataset, DatasetMetadata, LadduResult, Vec3, Vec4, PI,
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

#[derive(Clone, Debug)]
pub struct GenInitialState {
    mass: f64,
    energy_distribution: SimpleDistribution,
}

impl GenInitialState {
    pub fn beam_with_fixed_energy(mass: f64, energy: f64) -> Self {
        debug_assert!(mass >= 0.0, "Mass cannot be negative!\nMass: {}", mass);
        debug_assert!(energy > 0.0, "Energy must be positive!\nEnergy: {}", energy);
        Self {
            mass,
            energy_distribution: SimpleDistribution::Fixed(energy),
        }
    }
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
    pub fn target(mass: f64) -> Self {
        Self {
            mass,
            energy_distribution: SimpleDistribution::Fixed(mass),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenComposite {
    mass_distribution: SimpleDistribution,
}
impl GenComposite {
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

#[derive(Clone, Debug)]
pub struct GenFinalState {
    mass_distribution: SimpleDistribution,
}
impl GenFinalState {
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

#[derive(Clone, Debug)]
pub enum Reconstruction {
    Reconstructed { p4_names: Vec<String> },
    Fixed(Vec4),
    Missing,
}

#[derive(Clone, Debug)]
pub struct InitialStateParticle {
    label: String,
    generator: GenInitialState,
    reconstruction: Reconstruction,
}

#[derive(Clone, Debug)]
pub enum FinalStateParticle {
    Composite {
        label: String,
        generator: GenComposite,
        daughters: (Box<FinalStateParticle>, Box<FinalStateParticle>),
    },
    Final {
        label: String,
        generator: GenFinalState,
        reconstruction: Reconstruction,
    },
}

impl InitialStateParticle {
    pub fn new(
        label: impl Into<String>,
        generator: GenInitialState,
        reconstruction: Reconstruction,
    ) -> Self {
        Self {
            label: label.into(),
            generator,
            reconstruction,
        }
    }

    pub fn reconstruction(&self) -> &Reconstruction {
        &self.reconstruction
    }
}
impl FinalStateParticle {
    pub fn new(
        label: impl Into<String>,
        generator: GenFinalState,
        reconstruction: Reconstruction,
    ) -> Self {
        Self::Final {
            label: label.into(),
            generator,
            reconstruction,
        }
    }
    pub fn composite(
        label: impl Into<String>,
        generator: GenComposite,
        daughters: (&FinalStateParticle, &FinalStateParticle),
    ) -> Self {
        Self::Composite {
            label: label.into(),
            generator,
            daughters: (Box::new(daughters.0.clone()), Box::new(daughters.1.clone())),
        }
    }
    fn label(&self) -> &str {
        match self {
            Self::Composite { label, .. } => label,
            Self::Final { label, .. } => label,
        }
    }
    pub fn reconstruction(&self) -> Option<&Reconstruction> {
        match self {
            Self::Composite { .. } => None,
            Self::Final { reconstruction, .. } => Some(reconstruction),
        }
    }
    fn p4_labels(&self) -> Vec<String> {
        match self {
            Self::Composite {
                label, daughters, ..
            } => {
                let mut labels = vec![label.clone()];
                labels.append(&mut daughters.0.p4_labels());
                labels.append(&mut daughters.1.p4_labels());
                labels
            }
            Self::Final { label, .. } => vec![label.clone()],
        }
    }
    fn sample_mass(&self, rng: &mut Rng) -> f64 {
        match self {
            FinalStateParticle::Composite { generator, .. } => generator.sample_mass(rng),
            FinalStateParticle::Final { generator, .. } => generator.sample_mass(rng),
        }
    }
    fn generate_decay(
        &self,
        rng: &mut Rng,
        p4_cm: Vec4,
        cm_to_lab_boost: &Vec3,
        p4_storage: &mut HashMap<String, ColumnarP4Column>,
    ) {
        let p4_lab = p4_cm.boost(cm_to_lab_boost);
        if let Some(storage) = p4_storage.get_mut(self.label()) {
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

pub struct GenTwoToTwoReaction {
    p1: InitialStateParticle,
    p2: InitialStateParticle,
    p3: FinalStateParticle,
    p4: FinalStateParticle,
    tdist: MandelstamTDistribution,
    p1_p3_lab_dir: Vec3,
    p2_p3_lab_dir: Vec3,
}

impl GenTwoToTwoReaction {
    pub fn new(
        p1: InitialStateParticle,
        p2: InitialStateParticle,
        p3: FinalStateParticle,
        p4: FinalStateParticle,
        tdist: MandelstamTDistribution,
    ) -> Self {
        // TODO: checks to ensure no loops
        Self {
            p1,
            p2,
            p3,
            p4,
            tdist,
            p1_p3_lab_dir: Vec3::z(), // TODO: add custom initializers
            p2_p3_lab_dir: -Vec3::z(),
        }
    }
    fn p4_labels(&self) -> Vec<String> {
        let mut labels = vec![];
        labels.push(self.p1.label.clone());
        labels.push(self.p2.label.clone());
        labels.append(&mut self.p3.p4_labels());
        labels.append(&mut self.p4.p4_labels());
        labels
    }
    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, ColumnarP4Column>) {
        // generate p1 (lab)
        let p1_e = self.p1.generator.energy_distribution.sample(rng);
        let p1_m = self.p1.generator.mass;
        let p1_msq = p1_m * p1_m;
        let p1_p4_lab = rng.p4(p1_m, p1_e, self.p1_p3_lab_dir);
        if let Some(storage) = p4_storage.get_mut(&self.p1.label) {
            storage.push(p1_p4_lab)
        }

        // generate p2 (lab)
        let p2_e = self.p2.generator.energy_distribution.sample(rng);
        let p2_m = self.p2.generator.mass;
        let p2_msq = p2_m * p2_m;
        let p2_p4_lab = rng.p4(p2_m, p2_e, self.p2_p3_lab_dir);
        if let Some(storage) = p4_storage.get_mut(&self.p2.label) {
            storage.push(p2_p4_lab)
        }

        // boosts and inavriants
        let cm = p1_p4_lab + p2_p4_lab;
        let cm_boost = -cm.beta();
        let s = cm.mag2();
        let sqrt_s = s.sqrt();
        let t = self.tdist.sample(rng);

        // cm frame p1/p2
        let p1_p4_cm = p1_p4_lab.boost(&cm_boost);
        // let p2_p4_cm = p2_p4_lab.boost(&cm_boost);

        // generate p3/p4 (cm)
        let p3_m = self.p3.sample_mass(rng);
        let p3_msq = p3_m * p3_m;
        let p4_m = self.p4.sample_mass(rng);
        let p4_msq = p4_m * p4_m;
        let p_in_mag = q_m(sqrt_s, p1_m, p2_m, Sheet::Physical).re;
        let p_out_mag = q_m(sqrt_s, p3_m, p4_m, Sheet::Physical).re;
        let p1_e_cm = (s + p1_msq - p2_msq) / (2.0 * sqrt_s);
        let p3_e_cm = (s + p3_msq - p4_msq) / (2.0 * sqrt_s);
        let p4_e_cm = (s + p4_msq - p3_msq) / (2.0 * sqrt_s);
        let cos_theta =
            (t - p1_msq - p3_msq + 2.0 * p1_e_cm * p3_e_cm) / (2.0 * p_in_mag * p_out_mag);
        let cos_theta = cos_theta.clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = rng.uniform(0.0, 2.0 * PI);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (x, y, z) = basis(p1_p4_cm.vec3());
        let p3_dir_cm = x * (sin_theta * cos_phi) + y * (sin_theta * sin_phi) + z * cos_theta;

        let p3_p4_cm = (p3_dir_cm * p_out_mag).with_energy(p3_e_cm);
        self.p3
            .generate_decay(rng, p3_p4_cm, &-cm_boost, p4_storage);
        let p4_p4_cm = (-p3_dir_cm * p_out_mag).with_energy(p4_e_cm);
        self.p4
            .generate_decay(rng, p4_p4_cm, &-cm_boost, p4_storage);
    }
}

pub enum GenReactionTopology {
    TwoToTwo(GenTwoToTwoReaction),
}
impl GenReactionTopology {
    fn p4_labels(&self) -> Vec<String> {
        match self {
            GenReactionTopology::TwoToTwo(reaction) => reaction.p4_labels(),
        }
    }
    fn generate_event(&self, rng: &mut Rng, p4_storage: &mut HashMap<String, ColumnarP4Column>) {
        match self {
            GenReactionTopology::TwoToTwo(reaction) => reaction.generate_event(rng, p4_storage),
        }
    }
}

pub struct GenReaction {
    topology: GenReactionTopology,
}

impl GenReaction {
    pub fn two_to_two(
        p1: InitialStateParticle,
        p2: InitialStateParticle,
        p3: FinalStateParticle,
        p4: FinalStateParticle,
        tdist: MandelstamTDistribution,
    ) -> Self {
        Self {
            topology: GenReactionTopology::TwoToTwo(GenTwoToTwoReaction::new(
                p1, p2, p3, p4, tdist,
            )),
        }
    }
    pub fn p4_labels(&self) -> Vec<String> {
        self.topology.p4_labels()
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

pub struct EventGenerator {
    reaction: GenReaction,
    aux_generators: HashMap<String, Distribution>,
    seed: u64,
}

impl EventGenerator {
    pub fn new(
        reaction: GenReaction,
        aux_generators: HashMap<String, Distribution>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            reaction,
            aux_generators,
            seed: seed.unwrap_or_else(|| fastrand::u64(..)),
        }
    }
    pub fn generate_dataset(&self, n_events: usize) -> LadduResult<Dataset> {
        let p4_labels = self.reaction.p4_labels();
        let mut p4_data: HashMap<String, ColumnarP4Column> = p4_labels
            .iter()
            .map(|label| (label.clone(), ColumnarP4Column::with_capacity(n_events)))
            .collect();
        let metadata = DatasetMetadata::new(
            p4_labels.clone(),
            self.aux_generators.keys().cloned().collect(), // TODO: need to check if the order matches iteration order always
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

    use super::*;
    fn demo_reaction() -> GenReaction {
        let beam = InitialStateParticle::new(
            "beam",
            GenInitialState::beam_with_fixed_energy(0.0, 8.0),
            Reconstruction::Reconstructed {
                p4_names: vec!["beam".to_string()],
            },
        );
        let target = InitialStateParticle::new(
            "target",
            GenInitialState::target(0.938272),
            Reconstruction::Missing,
        );
        let ks1 = FinalStateParticle::new(
            "kshort1",
            GenFinalState::new(0.497611),
            Reconstruction::Reconstructed {
                p4_names: vec!["kshort1".to_string()],
            },
        );
        let ks2 = FinalStateParticle::new(
            "kshort2",
            GenFinalState::new(0.497611),
            Reconstruction::Reconstructed {
                p4_names: vec!["kshort2".to_string()],
            },
        );
        let kk = FinalStateParticle::composite("kk", GenComposite::new(1.1, 1.6), (&ks1, &ks2));
        let recoil = FinalStateParticle::new(
            "recoil",
            GenFinalState::new(0.938272),
            Reconstruction::Reconstructed {
                p4_names: vec!["recoil".to_string()],
            },
        );
        let tdist = MandelstamTDistribution::Exponential { slope: 0.1 };
        GenReaction::two_to_two(beam, target, kk, recoil, tdist)
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
}
