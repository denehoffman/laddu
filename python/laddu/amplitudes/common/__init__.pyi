from laddu.amplitudes import Amplitude, ParameterLike

def Scalar(name: str, value: ParameterLike) -> Amplitude: ...  # noqa: N802
def ComplexScalar(name: str, re: ParameterLike, im: ParameterLike) -> Amplitude: ...  # noqa: N802
def PolarComplexScalar(name: str, r: ParameterLike, theta: ParameterLike) -> Amplitude: ...  # noqa: N802
