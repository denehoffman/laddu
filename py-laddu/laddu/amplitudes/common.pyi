from laddu.amplitudes import Expression, Parameter
from laddu.utils.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
)

def Scalar(name: str, value: Parameter | None = None) -> Expression: ...
def VariableScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
) -> Expression: ...
def ComplexScalar(
    name: str, re_im: tuple[Parameter, Parameter] | None = None
) -> Expression: ...
def PolarComplexScalar(
    name: str, r_theta: tuple[Parameter, Parameter] | None = None
) -> Expression: ...
