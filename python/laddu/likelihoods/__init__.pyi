from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitudes import Expression, Manager
from laddu.data import Dataset

class LikelihoodID:
    def __add__(self, other: LikelihoodID | LikelihoodExpression) -> LikelihoodExpression: ...
    def __mul__(self, other: LikelihoodID | LikelihoodExpression) -> LikelihoodExpression: ...

class LikelihoodExpression:
    def __add__(self, other: LikelihoodID | LikelihoodExpression) -> LikelihoodExpression: ...
    def __mul__(self, other: LikelihoodID | LikelihoodExpression) -> LikelihoodExpression: ...

class LikelihoodTerm: ...

class LikelihoodManager:
    def __init__(self) -> None: ...
    def register(self, likelihood_term: LikelihoodTerm) -> LikelihoodID: ...
    def load(self, likelihood_expression: LikelihoodExpression) -> LikelihoodEvaluator: ...

class LikelihoodEvaluator:
    parameters: list[str]
    def evaluate(self, parameters: list[float] | npt.NDArray[np.float64]) -> float: ...
    def minimize(
        self,
        p0: list[float] | npt.NDArray[np.float64],
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal["lbfgsb", "nelder_mead"] = "lbfgsb",
        max_steps: int = 4000,
        debug: bool = False,  # noqa: FBT001, FBT002
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ) -> Status: ...

class NLL:
    parameters: list[str]
    data: Dataset
    accmc: Dataset
    genmc: Dataset
    def __init__(
        self,
        manager: Manager,
        expression: Expression,
        ds_data: Dataset,
        ds_accmc: Dataset,
        ds_genmc: Dataset | None = None,
    ) -> None: ...
    def as_term(self) -> LikelihoodTerm: ...
    def activate(self, name: str | list[str]) -> None: ...
    def activate_all(self) -> None: ...
    def deactivate(self, name: str | list[str]) -> None: ...
    def deactivate_all(self) -> None: ...
    def isolate(self, name: str | list[str]) -> None: ...
    def evaluate(self, parameters: list[float] | npt.NDArray[np.float64]) -> float: ...
    def project(
        self, parameters: list[float] | npt.NDArray[np.float64], *, corrected: bool = False
    ) -> npt.NDArray[np.float64]: ...
    def project_with(
        self, parameters: list[float] | npt.NDArray[np.float64], name: str | list[str], *, corrected: bool = False
    ) -> npt.NDArray[np.float64]: ...
    def minimize(
        self,
        p0: list[float] | npt.NDArray[np.float64],
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal["lbfgsb", "nelder_mead"] = "lbfgsb",
        max_steps: int = 4000,
        debug: bool = False,  # noqa: FBT001, FBT002
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ) -> Status: ...

def LikelihoodScalar(name: str) -> LikelihoodTerm: ...  # noqa: N802

class Status:
    x: npt.NDArray[np.float64]
    err: npt.NDArray[np.float64] | None
    x0: npt.NDArray[np.float64]
    fx: float
    cov: npt.NDArray[np.float64] | None
    hess: npt.NDArray[np.float64] | None
    message: str
    converged: bool
    bounds: list[Bound] | None
    n_f_evals: int
    n_g_evals: int

    def save_as(self, path: str): ...
    @staticmethod
    def load(path: str) -> Status: ...

class Bound:
    lower: float
    upper: float

__all__ = [
    "NLL",
    "Status",
    "Bound",
    "LikelihoodID",
    "LikelihoodExpression",
    "LikelihoodTerm",
    "LikelihoodManager",
    "LikelihoodEvaluator",
    "LikelihoodScalar",
]
