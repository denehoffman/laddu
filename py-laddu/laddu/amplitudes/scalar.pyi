from laddu.amplitude import Expression, Parameter
from laddu.variables import (
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    PolMagnitude,
)

def Scalar(*tags: str, value: Parameter) -> Expression: ...
def VariableScalar(
    *tags: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
) -> Expression: ...
def ComplexScalar(*tags: str, re: Parameter, im: Parameter) -> Expression: ...
def PolarComplexScalar(*tags: str, r: Parameter, theta: Parameter) -> Expression: ...
