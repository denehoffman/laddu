use nalgebra::{SMatrix, SVector};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use super::cache_ids::{
    ComplexMatrixID, ComplexScalarID, ComplexVectorID, MatrixID, ScalarID, VectorID,
};

/// A single cache entry corresponding to precomputed data for a particular
/// [`EventData`](crate::data::EventData) in a [`Dataset`](crate::data::Dataset).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cache(pub(crate) Vec<f64>);
impl Cache {
    pub(crate) fn new(cache_size: usize) -> Self {
        Self(vec![0.0; cache_size])
    }
    /// Store a scalar value with the corresponding [`ScalarID`].
    pub fn store_scalar(&mut self, sid: ScalarID, value: f64) {
        self.0[sid.0] = value;
    }
    /// Store a complex scalar value with the corresponding [`ComplexScalarID`].
    pub fn store_complex_scalar(&mut self, csid: ComplexScalarID, value: Complex64) {
        self.0[csid.0] = value.re;
        self.0[csid.1] = value.im;
    }
    /// Store a vector with the corresponding [`VectorID`].
    pub fn store_vector<const R: usize>(&mut self, vid: VectorID<R>, value: SVector<f64, R>) {
        vid.0
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi]);
    }
    /// Store a complex-valued vector with the corresponding [`ComplexVectorID`].
    pub fn store_complex_vector<const R: usize>(
        &mut self,
        cvid: ComplexVectorID<R>,
        value: SVector<Complex64, R>,
    ) {
        cvid.0
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi].re);
        cvid.1
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi].im);
    }
    /// Store a matrix with the corresponding [`MatrixID`].
    pub fn store_matrix<const R: usize, const C: usize>(
        &mut self,
        mid: MatrixID<R, C>,
        value: SMatrix<f64, R, C>,
    ) {
        mid.0.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)])
        });
    }
    /// Store a complex-valued matrix with the corresponding [`ComplexMatrixID`].
    pub fn store_complex_matrix<const R: usize, const C: usize>(
        &mut self,
        cmid: ComplexMatrixID<R, C>,
        value: SMatrix<Complex64, R, C>,
    ) {
        cmid.0.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)].re)
        });
        cmid.1.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)].im)
        });
    }
    /// Retrieve a scalar value from the [`Cache`].
    pub fn get_scalar(&self, sid: ScalarID) -> f64 {
        self.0[sid.0]
    }
    /// Retrieve a complex scalar value from the [`Cache`].
    pub fn get_complex_scalar(&self, csid: ComplexScalarID) -> Complex64 {
        Complex64::new(self.0[csid.0], self.0[csid.1])
    }
    /// Retrieve a vector from the [`Cache`].
    pub fn get_vector<const R: usize>(&self, vid: VectorID<R>) -> SVector<f64, R> {
        SVector::from_fn(|i, _| self.0[vid.0[i]])
    }
    /// Retrieve a complex-valued vector from the [`Cache`].
    pub fn get_complex_vector<const R: usize>(
        &self,
        cvid: ComplexVectorID<R>,
    ) -> SVector<Complex64, R> {
        SVector::from_fn(|i, _| Complex64::new(self.0[cvid.0[i]], self.0[cvid.1[i]]))
    }
    /// Retrieve a matrix from the [`Cache`].
    pub fn get_matrix<const R: usize, const C: usize>(
        &self,
        mid: MatrixID<R, C>,
    ) -> SMatrix<f64, R, C> {
        SMatrix::from_fn(|i, j| self.0[mid.0[i][j]])
    }
    /// Retrieve a complex-valued matrix from the [`Cache`].
    pub fn get_complex_matrix<const R: usize, const C: usize>(
        &self,
        cmid: ComplexMatrixID<R, C>,
    ) -> SMatrix<Complex64, R, C> {
        SMatrix::from_fn(|i, j| Complex64::new(self.0[cmid.0[i][j]], self.0[cmid.1[i][j]]))
    }
}
