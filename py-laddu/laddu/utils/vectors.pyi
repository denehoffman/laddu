from typing import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ['Vector3', 'Vector4']

class Vector3:
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
    unit: Vector3
    px: float
    py: float
    pz: float
    def __init__(self, px: float, py: float, pz: float) -> None: ...
    def __add__(self, other: Vector3 | int) -> Vector3: ...
    def __radd__(self, other: Vector3 | int) -> Vector3: ...
    def __sub__(self, other: Vector3 | int) -> Vector3: ...
    def __rsub__(self, other: Vector3 | int) -> Vector3: ...
    def __neg__(self) -> Vector3: ...
    def dot(self, other: Vector3) -> float:
        """Calculate the dot product of two vectors.

        Parameters
        ----------
        other : Vector3
            A vector input with which the dot product is taken

        Returns
        -------
        float
            The dot product of this vector and `other`
        """
    def cross(self, other: Vector3) -> Vector3:
        """
        Calculate the cross product of two vectors.

        Parameters
        ----------
        other : Vector3
            A vector input with which the cross product is taken

        Returns
        -------
        Vector3
            The cross product of this vector and `other`
        """
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_array(array: Sequence) -> Vector3: ...
    def with_mass(self, mass: float) -> Vector4: ...
    def with_energy(self, mass: float) -> Vector4: ...
    def __getitem__(self, index: int) -> float: ...

class Vector4:
    mag: float
    mag2: float
    vec3: Vector3
    e: float
    px: float
    py: float
    pz: float
    momentum: Vector3
    gamma: float
    beta: Vector3
    m: float
    m2: float
    def __init__(self, px: float, py: float, pz: float, e: float) -> None: ...
    def __add__(self, other: Vector4) -> Vector4: ...
    def __sub__(self, other: Vector4 | int) -> Vector4: ...
    def __rsub__(self, other: Vector4 | int) -> Vector4: ...
    def __neg__(self) -> Vector4: ...
    def boost(self, beta: Vector3) -> Vector4: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_array(array: Sequence) -> Vector4: ...
    def __getitem__(self, index: int) -> float: ...
