use laddu::prelude::*;
use laddu::utils::functions::{blatt_weisskopf, breakup_momentum};

#[derive(Clone)]
pub struct MyBreitWigner {
    name: String,
    mass: ParameterLike,
    width: ParameterLike,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    l: usize,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
}
impl MyBreitWigner {
    pub fn new(
        name: &str,
        mass: ParameterLike,
        width: ParameterLike,
        l: usize,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> Box<Self> {
        Self {
            name: name.to_string(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            l,
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
        }
        .into()
    }
}

impl Amplitude for MyBreitWigner {
    fn register(&mut self, resources: &mut Resources) -> AmplitudeID {
        self.pid_mass = resources.register_parameter(&self.mass);
        self.pid_width = resources.register_parameter(&self.width);
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, event: &Event, _cache: &Cache) -> Complex<Float> {
        let mass = self.resonance_mass.value(event);
        let mass0 = parameters.get(self.pid_mass);
        let width0 = parameters.get(self.pid_width);
        let mass1 = self.daughter_1_mass.value(event);
        let mass2 = self.daughter_2_mass.value(event);
        let q0 = breakup_momentum(mass0, mass1, mass2);
        let q = breakup_momentum(mass, mass1, mass2);
        let f0 = blatt_weisskopf(mass0, mass1, mass2, self.l);
        let f = blatt_weisskopf(mass, mass1, mass2, self.l);
        let width = width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2);
        let n = Float::sqrt(mass0 * width0 / PI);
        let d = Complex::new(mass0.powi(2) - mass.powi(2), -(mass0 * width));
        Complex::from(f * n) / d
    }
}

fn main() {
    let ds_data = open("test_data/data.parquet").unwrap();
    let ds_mc = open("test_data/mc.parquet").unwrap();

    let resonance_mass = Mass::new(&[2, 3]);
    let p1_mass = Mass::new(&[2]);
    let p2_mass = Mass::new(&[3]);
    let mut manager = Manager::default();
    let bw = manager.register(MyBreitWigner::new(
        "bw",
        parameter("mass"),
        parameter("width"),
        2,
        &p1_mass,
        &p2_mass,
        &resonance_mass,
    ));
    let mag = manager.register(Scalar::new("mag", parameter("magnitude")));
    let model = (mag * bw).norm_sqr();

    let nll = NLL::new(&manager, &ds_data, &ds_mc);
    println!("Parameters: {:?}", nll.parameters());
    let result = nll.evaluate(&[1.27, 0.120, 100.0], &model);
    println!("The extended negative log-likelihood is {}", result);
}
