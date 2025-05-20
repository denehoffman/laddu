from typing import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ['Vec3', 'Vec4']

class Vec3:
    """A 3-momentum vector formed from Cartesian components.

    Parameters
    ----------
    px, py, pz : float
        The Cartesian components of the 3-vector

    Attributes
    ----------
    mag : float
        The magnitude of the 3-vector
    mag2 : float
        The squared magnitude of the 3-vector
    """

    mag: float
    """
    The magnitude of the 3-vector

    .. math:: |\vec{p}| = \\sqrt{p_x^2 + p_y^2 + p_z^2}
    """
    mag2: float
    costheta: float
    theta: float
    phi: float
    unit: Vec3
    x: float
    y: float
    z: float
    px: float
    py: float
    pz: float
    def __init__(self, px: float, py: float, pz: float) -> None: ...
    def __add__(self, other: Vec3 | int) -> Vec3: ...
    def __radd__(self, other: Vec3 | int) -> Vec3: ...
    def __sub__(self, other: Vec3 | int) -> Vec3: ...
    def __rsub__(self, other: Vec3 | int) -> Vec3: ...
    def __neg__(self) -> Vec3: ...
    def dot(self, other: Vec3) -> float:
        """Calculate the dot product of two vectors.

        Parameters
        ----------
        other : Vec3
            A vector input with which the dot product is taken

        Returns
        -------
        float
            The dot product of this vector and `other`
        """
    def cross(self, other: Vec3) -> Vec3:
        """
        Calculate the cross product of two vectors.

        Parameters
        ----------
        other : Vec3
            A vector input with which the cross product is taken

        Returns
        -------
        Vec3
            The cross product of this vector and `other`
        """
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_array(array: Sequence) -> Vec3: ...
    def with_mass(self, mass: float) -> Vec4: ...
    def with_energy(self, mass: float) -> Vec4: ...

class Vec4:
    mag: float
    mag2: float
    vec3: Vec3
    t: float
    x: float
    y: float
    z: float
    e: float
    px: float
    py: float
    pz: float
    momentum: Vec3
    gamma: float
    beta: Vec3
    m: float
    m2: float
    def __init__(self, px: float, py: float, pz: float, e: float) -> None: ...
    def __add__(self, other: Vec4) -> Vec4: ...
    def __sub__(self, other: Vec4 | int) -> Vec4: ...
    def __rsub__(self, other: Vec4 | int) -> Vec4: ...
    def __neg__(self) -> Vec4: ...
    def boost(self, beta: Vec3) -> Vec4: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_array(array: Sequence) -> Vec4: ...
