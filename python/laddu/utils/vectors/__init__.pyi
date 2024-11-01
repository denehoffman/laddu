import numpy as np
import numpy.typing as npt

class Vector3:
    mag: float
    mag2: float
    costheta: float
    theta: float
    phi: float
    unit: Vector3
    px: float
    py: float
    pz: float
    def __init__(self, px: float, py: float, pz: float): ...
    def __add__(self, other: Vector3) -> Vector3: ...
    def dot(self, other: Vector3) -> float: ...
    def cross(self, other: Vector3) -> Vector3: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...

class Vector4:
    mag: float
    mag2: float
    vec3: Vector3
    e: float
    px: float
    py: float
    pz: float
    momentum: Vector3
    beta: Vector3
    m: float
    m2: float
    def __init__(self, e: float, px: float, py: float, pz: float): ...
    def __add__(self, other: Vector4) -> Vector4: ...
    def boost(self, beta: Vector3) -> Vector4: ...
    def boost_along(self, other: Vector4) -> Vector4: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_momentum(momentum: Vector3, mass: float) -> Vector4: ...
