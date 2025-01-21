//! `laddu` (/ˈlʌduː/) is a library for analysis of particle physics data. It is intended to be a simple and efficient alternative to some of the [other tools](#alternatives) out there. `laddu` is written in Rust with bindings to Python via [`PyO3`](https://github.com/PyO3/pyo3) and [`maturin`](https://github.com/PyO3/maturin) and is the spiritual successor to [`rustitude`](https://github.com/denehoffman/rustitude), one of my first Rust projects. The goal of this project is to allow users to perform complex amplitude analyses (like partial-wave analyses) without complex code or configuration files.
//!
//! <div class="warning">
//!
//! This crate is still in an early development phase, and the API is not stable. It can (and likely will) be subject to breaking changes before the 1.0.0 version release (and hopefully not many after that).
//!
//! </div>
//!
//! # Table of Contents
//! - [Key Features](#key-features)
//! - [Installation](#installation)
//! - [Quick Start](#quick-start)
//!   - [Writing a New Amplitude](#writing-a-new-amplitude)
//!   - [Calculating a Likelihood](#calculating-a-likelihood)
//! - [Data Format](#data-format)
//! - [Future Plans](#future-plans)
//! - [Alternatives](#alternatives)
//!
//! # Key Features
//! * A simple interface focused on combining [`Amplitude`](crate::amplitudes::Amplitude)s into models which can be evaluated over [`Dataset`]s.
//! * A single [`Amplitude`](crate::amplitudes::Amplitude) trait which makes it easy to write new amplitudes and integrate them into the library.
//! * Easy interfaces to precompute and cache values before the main calculation to speed up model evaluations.
//! * Efficient parallelism using [`rayon`](https://github.com/rayon-rs/rayon).
//! * Python bindings to allow users to write quick, easy-to-read code that just works.
//!
//! # Installation
//! `laddu` can be added to a Rust project with `cargo`:
//! ```shell
//! cargo add laddu
//! ```
//!
//! The library's Python bindings are located in a library by the same name, which can be installed simply with your favorite Python package manager:
//! ```shell
//! pip install laddu
//! ```
//!
//! # Quick Start
//! ## Rust
//! ### Writing a New Amplitude
//! At the time of writing, Rust is not a common language used by particle physics, but this tutorial should hopefully convince the reader that they don't have to know the intricacies of Rust to write performant amplitudes. As an example, here is how one might write a Breit-Wigner, parameterized as follows:
//! ```math
//! I_{\ell}(m; m_0, \Gamma_0, m_1, m_2) =  \frac{1}{\pi}\frac{m_0 \Gamma_0 B_{\ell}(m, m_1, m_2)}{(m_0^2 - m^2) - \imath m_0 \Gamma}
//! ```
//! where
//! ```math
//! \Gamma = \Gamma_0 \frac{m_0}{m} \frac{q(m, m_1, m_2)}{q(m_0, m_1, m_2)} \left(\frac{B_{\ell}(m, m_1, m_2)}{B_{\ell}(m_0, m_1, m_2)}\right)^2
//! ```
//! is the relativistic width correction, $`q(m_a, m_b, m_c)`$ is the breakup momentum of a particle with mass $`m_a`$ decaying into two particles with masses $`m_b`$ and $`m_c`$, $`B_{\ell}(m_a, m_b, m_c)`$ is the Blatt-Weisskopf barrier factor for the same decay assuming particle $`a`$ has angular momentum $`\ell`$, $`m_0`$ is the mass of the resonance, $`\Gamma_0`$ is the nominal width of the resonance, $`m_1`$ and $`m_2`$ are the masses of the decay products, and $`m`$ is the "input" mass.
//!
//! Although this particular amplitude is already included in `laddu`, let's assume it isn't and imagine how we would write it from scratch:
//!
//! ```rust
//! use laddu::{
//!    ParameterLike, Event, Cache, Resources, Mass,
//!    ParameterID, Parameters, Float, LadduError, PI, AmplitudeID, Complex,
//! };
//! use laddu::traits::*;
//! use laddu::utils::functions::{blatt_weisskopf, breakup_momentum};
//! use laddu::{Deserialize, Serialize, typetag};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! pub struct MyBreitWigner {
//!     name: String,
//!     mass: ParameterLike,
//!     width: ParameterLike,
//!     pid_mass: ParameterID,
//!     pid_width: ParameterID,
//!     l: usize,
//!     daughter_1_mass: Mass,
//!     daughter_2_mass: Mass,
//!     resonance_mass: Mass,
//! }
//! impl MyBreitWigner {
//!     pub fn new(
//!         name: &str,
//!         mass: ParameterLike,
//!         width: ParameterLike,
//!         l: usize,
//!         daughter_1_mass: &Mass,
//!         daughter_2_mass: &Mass,
//!         resonance_mass: &Mass,
//!     ) -> Box<Self> {
//!         Self {
//!             name: name.to_string(),
//!             mass,
//!             width,
//!             pid_mass: ParameterID::default(),
//!             pid_width: ParameterID::default(),
//!             l,
//!             daughter_1_mass: daughter_1_mass.clone(),
//!             daughter_2_mass: daughter_2_mass.clone(),
//!             resonance_mass: resonance_mass.clone(),
//!         }
//!         .into()
//!     }
//! }
//!
//! #[typetag::serde]
//! impl Amplitude for MyBreitWigner {
//!     fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
//!         self.pid_mass = resources.register_parameter(&self.mass);
//!         self.pid_width = resources.register_parameter(&self.width);
//!         resources.register_amplitude(&self.name)
//!     }
//!
//!     fn compute(&self, parameters: &Parameters, event: &Event, _cache: &Cache) -> Complex<Float> {
//!         let mass = self.resonance_mass.value(event);
//!         let mass0 = parameters.get(self.pid_mass);
//!         let width0 = parameters.get(self.pid_width);
//!         let mass1 = self.daughter_1_mass.value(event);
//!         let mass2 = self.daughter_2_mass.value(event);
//!         let q0 = breakup_momentum(mass0, mass1, mass2);
//!         let q = breakup_momentum(mass, mass1, mass2);
//!         let f0 = blatt_weisskopf(mass0, mass1, mass2, self.l);
//!         let f = blatt_weisskopf(mass, mass1, mass2, self.l);
//!         let width = width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2);
//!         let n = Float::sqrt(mass0 * width0 / PI);
//!         let d = Complex::new(mass0.powi(2) - mass.powi(2), -(mass0 * width));
//!         Complex::from(f * n) / d
//!     }
//! }
//! ```
//!
//! While it isn't shown here, we can often be more efficient when implementing
//! [`Amplitude`](crate::amplitudes::Amplitude)s by precomputing values which do not depend on the
//! free parameters. See the [`Amplitude::precompute`](crate::amplitudes::Amplitude::precompute)
//! method for more details.
//!
//! ### Calculating a Likelihood
//! We could then write some code to use this amplitude. For demonstration purposes, let's just calculate an extended unbinned negative log-likelihood, assuming we have some data and Monte Carlo in the proper [parquet format](#data-format):
//! ```rust
//! # use laddu::{
//! #    ParameterLike, Event, Cache, Resources,
//! #    ParameterID, Parameters, Float, LadduError, PI, AmplitudeID, Complex,
//! # };
//! # use laddu::traits::*;
//! # use laddu::utils::functions::{blatt_weisskopf, breakup_momentum};
//! # use laddu::{Deserialize, Serialize, typetag};
//!
//! # #[derive(Clone, Serialize, Deserialize)]
//! # pub struct MyBreitWigner {
//! #     name: String,
//! #     mass: ParameterLike,
//! #     width: ParameterLike,
//! #     pid_mass: ParameterID,
//! #     pid_width: ParameterID,
//! #     l: usize,
//! #     daughter_1_mass: Mass,
//! #     daughter_2_mass: Mass,
//! #     resonance_mass: Mass,
//! # }
//! # impl MyBreitWigner {
//! #     pub fn new(
//! #         name: &str,
//! #         mass: ParameterLike,
//! #         width: ParameterLike,
//! #         l: usize,
//! #         daughter_1_mass: &Mass,
//! #         daughter_2_mass: &Mass,
//! #         resonance_mass: &Mass,
//! #     ) -> Box<Self> {
//! #         Self {
//! #             name: name.to_string(),
//! #             mass,
//! #             width,
//! #             pid_mass: ParameterID::default(),
//! #             pid_width: ParameterID::default(),
//! #             l,
//! #             daughter_1_mass: daughter_1_mass.clone(),
//! #             daughter_2_mass: daughter_2_mass.clone(),
//! #             resonance_mass: resonance_mass.clone(),
//! #         }
//! #         .into()
//! #     }
//! # }
//! #
//! # #[typetag::serde]
//! # impl Amplitude for MyBreitWigner {
//! #     fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
//! #         self.pid_mass = resources.register_parameter(&self.mass);
//! #         self.pid_width = resources.register_parameter(&self.width);
//! #         resources.register_amplitude(&self.name)
//! #     }
//! #
//! #     fn compute(&self, parameters: &Parameters, event: &Event, _cache: &Cache) -> Complex<Float> {
//! #         let mass = self.resonance_mass.value(event);
//! #         let mass0 = parameters.get(self.pid_mass);
//! #         let width0 = parameters.get(self.pid_width);
//! #         let mass1 = self.daughter_1_mass.value(event);
//! #         let mass2 = self.daughter_2_mass.value(event);
//! #         let q0 = breakup_momentum(mass0, mass1, mass2);
//! #         let q = breakup_momentum(mass, mass1, mass2);
//! #         let f0 = blatt_weisskopf(mass0, mass1, mass2, self.l);
//! #         let f = blatt_weisskopf(mass, mass1, mass2, self.l);
//! #         let width = width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2);
//! #         let n = Float::sqrt(mass0 * width0 / PI);
//! #         let d = Complex::new(mass0.powi(2) - mass.powi(2), -(mass0 * width));
//! #         Complex::from(f * n) / d
//! #     }
//! # }
//! use laddu::{Scalar, Mass, Manager, NLL, parameter, open};
//! let ds_data = open("test_data/data.parquet").unwrap();
//! let ds_mc = open("test_data/mc.parquet").unwrap();
//!
//! let resonance_mass = Mass::new([2, 3]);
//! let p1_mass = Mass::new([2]);
//! let p2_mass = Mass::new([3]);
//! let mut manager = Manager::default();
//! let bw = manager.register(MyBreitWigner::new(
//!     "bw",
//!     parameter("mass"),
//!     parameter("width"),
//!     2,
//!     &p1_mass,
//!     &p2_mass,
//!     &resonance_mass,
//! )).unwrap();
//! let mag = manager.register(Scalar::new("mag", parameter("magnitude"))).unwrap();
//! let expr = (mag * bw).norm_sqr();
//! let model = manager.model(&expr);
//!
//! let nll = NLL::new(&model, &ds_data, &ds_mc);
//! println!("Parameters names and order: {:?}", nll.parameters());
//! let result = nll.evaluate(&[1.27, 0.120, 100.0]);
//! println!("The extended negative log-likelihood is {}", result);
//! ```
//! In practice, amplitudes can also be added together, their real and imaginary parts can be taken, and evaluators should mostly take the real part of whatever complex value comes out of the model.
//!
//! # Data Format
//! The data format for `laddu` is a bit different from some of the alternatives like [`AmpTools`](https://github.com/mashephe/AmpTools). Since ROOT doesn't yet have bindings to Rust and projects to read ROOT files are still largely works in progress (although I hope to use [`oxyroot`](https://github.com/m-dupont/oxyroot) in the future when I can figure out a few bugs), the primary interface for data in `laddu` is Parquet files. These are easily accessible from almost any other language and they don't take up much more space than ROOT files. In the interest of future compatibility with any number of experimental setups, the data format consists of an arbitrary number of columns containing the four-momenta of each particle, the polarization vector of each particle (optional) and a single column for the weight. These columns all have standardized names. For example, the following columns would describe a dataset with four particles, the first of which is a polarized photon beam, as in the GlueX experiment:
//! | Column name | Data Type | Interpretation |
//! | ----------- | --------- | -------------- |
//! | `p4_0_E`    | `Float32` | Beam Energy    |
//! | `p4_0_Px`    | `Float32` | Beam Momentum (x-component) |
//! | `p4_0_Py`    | `Float32` | Beam Momentum (y-component) |
//! | `p4_0_Pz`    | `Float32` | Beam Momentum (z-component) |
//! | `eps_0_x`    | `Float32` | Beam Polarization (x-component) |
//! | `eps_0_y`    | `Float32` | Beam Polarization (y-component) |
//! | `eps_0_z`    | `Float32` | Beam Polarization (z-component) |
//! | `p4_1_E`    | `Float32` | Recoil Proton Energy    |
//! | `p4_1_Px`    | `Float32` | Recoil Proton Momentum (x-component) |
//! | `p4_1_Py`    | `Float32` | Recoil Proton Momentum (y-component) |
//! | `p4_1_Pz`    | `Float32` | Recoil Proton Momentum (z-component) |
//! | `p4_2_E`    | `Float32` | Decay Product 1 Energy    |
//! | `p4_2_Px`    | `Float32` | Decay Product 1 Momentum (x-component) |
//! | `p4_2_Py`    | `Float32` | Decay Product 1 Momentum (y-component) |
//! | `p4_2_Pz`    | `Float32` | Decay Product 1 Momentum (z-component) |
//! | `p4_3_E`    | `Float32` | Decay Product 2 Energy    |
//! | `p4_3_Px`    | `Float32` | Decay Product 2 Momentum (x-component) |
//! | `p4_3_Py`    | `Float32` | Decay Product 2 Momentum (y-component) |
//! | `p4_3_Pz`    | `Float32` | Decay Product 2 Momentum (z-component) |
//! | `weight`    | `Float32` | Event Weight |
//!
//! To make it easier to get started, we can directly convert from the `AmpTools` format using the provided [`amptools-to-laddu`] script (see the `bin` directory of this repository). This is not bundled with the Python library (yet) but may be in the future.
//!
//! # Future Plans
//! * MPI and GPU integration (these are incredibly difficult to do right now, but it's something I'm looking into).
//! * As always, more tests and documentation.
//!
//! # Alternatives
//! While this is likely the first Rust project (aside from my previous attempt, [`rustitude`](https://github.com/denehoffman/rustitude)), there are several other amplitude analysis programs out there at time of writing. This library is a rewrite of `rustitude` which was written when I was just learning Rust and didn't have a firm grasp of a lot of the core concepts that are required to make the analysis pipeline memory- and CPU-efficient. In particular, `rustitude` worked well, but ate up a ton of memory and did not handle precalculation as nicely.
//!
//! ### AmpTools
//! The main inspiration for this project is the library most of my collaboration uses, [`AmpTools`](https://github.com/mashephe/AmpTools). `AmpTools` has several advantages over `laddu`: it's probably faster for almost every use case, but this is mainly because it is fully integrated with MPI and GPU support. I'm not actually sure if there's a fair benchmark between the two libraries, but I'd wager `AmpTools` would still win. `AmpTools` is a much older, more developed project, dating back to 2010. However, it does have its disadvantages. First and foremost, the primary interaction with the library is through configuration files which are not really code and sort of represent a domain specific language. As such, there isn't really a way to check if a particular config will work before running it. Users could technically code up their analyses in C++ as well, but I think this would generally be more work for very little benefit. AmpTools primarily interacts with Minuit, so there aren't simple ways to perform alternative optimization algorithms, and the outputs are a file which must also be parsed by code written by the user. This usually means some boilerplate setup for each analysis, a slew of input and output files, and, since it doesn't ship with any amplitudes, integration with other libraries. The data format is also very rigid, to the point where including beam polarization information feels hacked on (see the Zlm implementation [here](https://github.com/JeffersonLab/halld_sim/blob/6815c979cac4b79a47e5183cf285ce9589fe4c7f/src/libraries/AMPTOOLS_AMPS/Zlm.cc#L26) which requires the event-by-event polarization to be stored in the beam's four-momentum). While there isn't an official Python interface, Lawrence Ng has made some progress porting the code [here](https://github.com/lan13005/PyAmpTools).
//!
//! ### PyPWA
//! [`PyPWA`](https://github.com/JeffersonLab/PyPWA/tree/main) is a library written in pure Python. While this might seem like an issue for performance (and it sort of is), the library has several features which encourage the use of JIT compilers. The upside is that analyses can be quickly prototyped and run with very few dependencies, it can even run on GPUs and use multiprocessing. The downside is that recent development has been slow and the actual implementation of common amplitudes is, in my opinion, [messy](https://pypwa.jlab.org/AmplitudeTWOsim.py). I don't think that's a reason to not use it, but it does make it difficult for new users to get started.
//!
//! ### ComPWA
//! [`ComPWA`](https://compwa.github.io/) is a newcomer to the field. It's also a pure Python implementation and is comprised of three separate libraries. [`QRules`](https://github.com/ComPWA/qrules) can be used to validate and generate particle reaction topologies using conservation rules. [`AmpForm`](https://github.com/ComPWA/ampform) uses `SymPy` to transform these topologies into mathematical expressions, and it can also simplify the mathematical forms through the built-in CAS of `SymPy`. Finally, [`TensorWaves`](https://github.com/ComPWA/tensorwaves) connects `AmpForm` to various fitting methods. In general, these libraries have tons of neat features, are well-documented, and are really quite nice to use. I would like to eventually see `laddu` as a companion to `ComPWA` (rather than direct competition), but I don't really know enough about the libraries to say much more than that.
//!
//! ### Others
//! It could be the case that I am leaving out software with which I am not familiar. If so, I'd love to include it here for reference. I don't think that `laddu` will ever be the end-all-be-all of amplitude analysis, just an alternative that might improve on existing systems. It is important for physicists to be aware of these alternatives. For example, if you really don't want to learn Rust but need to implement an amplitude which isn't already included here, `laddu` isn't for you, and one of these alternatives might be best.

/// Methods for loading and manipulating [`Event`]-based data.
pub mod data {
    pub use laddu_core::data::{open, BinnedDataset, Dataset, Event};
}
/// Module for likelihood-related structures and methods
pub mod extensions {
    pub use laddu_extensions::*;
}
/// Structures for manipulating the cache and free parameters.
pub mod resources {
    pub use laddu_core::resources::*;
}
/// Utility functions, enums, and traits
pub mod utils {
    pub use laddu_core::utils::*;
}
/// Useful traits for all crate structs
pub mod traits {
    pub use laddu_core::amplitudes::Amplitude;
    pub use laddu_core::utils::variables::Variable;
    pub use laddu_core::utils::vectors::{FourMomentum, FourVector, ThreeMomentum, ThreeVector};
    pub use laddu_core::ReadWrite;
    pub use laddu_extensions::likelihoods::LikelihoodTerm;
}
/// [`Amplitude`](crate::amplitudes::Amplitude)s and methods for making and evaluating them.
pub mod amplitudes {
    pub use laddu_amplitudes::*;
    pub use laddu_core::amplitudes::{
        constant, parameter, Amplitude, AmplitudeID, Evaluator, Expression, Manager, Model,
        ParameterLike,
    };
}

pub use laddu_amplitudes::*;
pub use laddu_core::amplitudes::{
    constant, parameter, AmplitudeID, Evaluator, Expression, Manager, Model, ParameterLike,
};
pub use laddu_core::data::{open, BinnedDataset, Dataset, Event};
pub use laddu_core::resources::{Cache, ParameterID, Parameters, Resources};
pub use laddu_core::utils::variables::{
    Angles, CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, Polarization,
};
pub use laddu_core::Complex;
pub use laddu_core::Float;
pub use laddu_core::LadduError;
pub use laddu_core::Status;
pub use laddu_core::PI;
pub use laddu_core::{DVector, Vector3, Vector4};
pub use laddu_extensions::*;
pub use serde::{Deserialize, Serialize};
pub use typetag;
