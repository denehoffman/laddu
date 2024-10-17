/// Useful enumerations for various frames and variables common in particle physics analyses.
pub mod enums;
/// Standard special functions like spherical harmonics and momentum definitions.
pub mod functions;
/// Traits and structs which can be used to extract complex information from
/// [`Event`](crate::data::Event)s.
pub mod variables;
/// Traits to give additional functionality to [`nalgebra::Vector3`] and [`nalgebra::Vector4`] (in
/// particular, to treat the latter as a four-momentum).
pub mod vectors;
