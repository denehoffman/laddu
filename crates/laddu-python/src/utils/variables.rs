use std::str::FromStr;

use laddu_core::{
    utils::{
        enums::Topology,
        variables::{
            angles, costheta, mandelstam, mass, phi, pol_angle, pol_magnitude, polarization,
        },
    },
    Channel, Frame,
};
use pyo3::prelude::*;
use pyo3_polars::PyExpr;

#[pyfunction]
#[pyo3(name="mass", signature=(constituents))]
pub fn py_mass(constituents: Vec<String>) -> PyExpr {
    PyExpr(mass(constituents))
}

#[pyfunction]
#[pyo3(name="costheta", signature=(*, beam, recoil, daughter, resonance, frame))]
pub fn py_costheta(
    beam: String,
    recoil: Vec<String>,
    daughter: Vec<String>,
    resonance: Vec<String>,
    frame: String,
) -> PyResult<PyExpr> {
    Ok(PyExpr(costheta(
        beam,
        recoil,
        daughter,
        resonance,
        Frame::from_str(&frame)?,
    )))
}
#[pyfunction]
#[pyo3(name="phi", signature=(*, beam, recoil, daughter, resonance, frame))]
pub fn py_phi(
    beam: String,
    recoil: Vec<String>,
    daughter: Vec<String>,
    resonance: Vec<String>,
    frame: String,
) -> PyResult<PyExpr> {
    Ok(PyExpr(phi(
        beam,
        recoil,
        daughter,
        resonance,
        Frame::from_str(&frame)?,
    )))
}

#[pyfunction]
#[pyo3(name="angles", signature=(*, beam, recoil, daughter, resonance, frame))]
pub fn py_angles(
    beam: String,
    recoil: Vec<String>,
    daughter: Vec<String>,
    resonance: Vec<String>,
    frame: String,
) -> PyResult<[PyExpr; 2]> {
    let [costheta, phi] = angles(beam, recoil, daughter, resonance, Frame::from_str(&frame)?);
    Ok([PyExpr(costheta), PyExpr(phi)])
}

#[pyfunction]
#[pyo3(name="pol_angle", signature=(*, beam, recoil, phi))]
pub fn py_pol_angle(beam: String, recoil: Vec<String>, phi: String) -> PyExpr {
    PyExpr(pol_angle(beam, recoil, phi))
}

#[pyfunction]
#[pyo3(name="pol_magnitude", signature=(*, p_gamma))]
pub fn py_pol_magnitude(p_gamma: String) -> PyExpr {
    PyExpr(pol_magnitude(p_gamma))
}

#[pyfunction]
#[pyo3(name="polarization", signature=(*, beam, recoil, p_gamma, phi))]
pub fn py_polarization(
    beam: String,
    recoil: Vec<String>,
    p_gamma: String,
    phi: String,
) -> [PyExpr; 2] {
    let [pol_magnitude, pol_angle] = polarization(beam, recoil, p_gamma, phi);
    [PyExpr(pol_magnitude), PyExpr(pol_angle)]
}

#[pyfunction]
#[pyo3(name="mandelstam", signature=(*, p1, p2, p3, p4, channel))]
pub fn py_mandelstam(
    p1: Vec<String>,
    p2: Vec<String>,
    p3: Vec<String>,
    p4: Vec<String>,
    channel: String,
) -> PyResult<PyExpr> {
    let topology = if p1.is_empty() {
        Topology::missing_p1(p2, p3, p4)
    } else if p2.is_empty() {
        Topology::missing_p2(p1, p3, p4)
    } else if p3.is_empty() {
        Topology::missing_p3(p1, p2, p4)
    } else if p4.is_empty() {
        Topology::missing_p4(p1, p2, p3)
    } else {
        Topology::all(p1, p2, p3, p4)
    };
    Ok(PyExpr(mandelstam(topology, Channel::from_str(&channel)?)))
}
