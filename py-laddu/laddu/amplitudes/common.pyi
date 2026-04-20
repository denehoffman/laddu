from laddu.amplitudes import Expression, Parameter
from laddu.utils.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
)

def Scalar(name: str, value: Parameter) -> Expression: ...
def VariableScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
) -> Expression: ...
def ComplexScalar(name: str, re: Parameter, im: Parameter) -> Expression: ...
def PolarComplexScalar(name: str, r: Parameter, theta: Parameter) -> Expression: ...
