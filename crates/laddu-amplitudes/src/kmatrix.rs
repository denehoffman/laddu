//! K-matrix amplitudes.

/// Fixed-coupling K-matrix implementations from Kopf et al.
pub mod kopf;

pub use kopf::{
    KopfKMatrixA0, KopfKMatrixA0Channel, KopfKMatrixA2, KopfKMatrixA2Channel, KopfKMatrixF0,
    KopfKMatrixF0Channel, KopfKMatrixF2, KopfKMatrixF2Channel, KopfKMatrixPi1,
    KopfKMatrixPi1Channel, KopfKMatrixRho, KopfKMatrixRhoChannel,
};
