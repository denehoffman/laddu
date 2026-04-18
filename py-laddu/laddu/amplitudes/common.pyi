from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
)

def Scalar(name: str, value: ParameterLike) -> Expression: ...
def VariableScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
) -> Expression: ...
def ComplexScalar(name: str, re: ParameterLike, im: ParameterLike) -> Expression: ...
def PolarComplexScalar(
    name: str, r: ParameterLike, theta: ParameterLike
) -> Expression: ...
