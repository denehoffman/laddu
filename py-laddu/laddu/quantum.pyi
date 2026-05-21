from fractions import Fraction
from typing import Literal, TypeAlias

QuantumNumber: TypeAlias = int | float | Fraction
Sign: TypeAlias = Literal[
    '+',
    '-',
    'plus',
    'minus',
    'pos',
    'neg',
    'positive',
    'negative',
]

class Parity:
    value: int
    def __init__(self, value: Sign) -> None: ...
    @staticmethod
    def positive() -> Parity: ...
    @staticmethod
    def negative() -> Parity: ...

class Statistics:
    Boson: Statistics
    Fermion: Statistics

class Charge:
    value: int | Fraction
    def __init__(self, value: QuantumNumber) -> None: ...

class Isospin:
    isospin: int | Fraction
    projection: int | Fraction
    projection_unchecked: int | Fraction | None
    def __init__(
        self, isospin: QuantumNumber, *, projection: QuantumNumber | None = None
    ) -> None: ...

class ParticleProperties:
    name: str
    name_unchecked: str | None
    species: str
    species_unchecked: str | None
    antiparticle_species: str
    antiparticle_species_unchecked: str | None
    self_conjugate: bool
    self_conjugate_unchecked: bool | None
    spin: int | Fraction
    spin_unchecked: int | Fraction | None
    parity: Parity
    parity_unchecked: Parity | None
    c_parity: Parity
    c_parity_unchecked: Parity | None
    g_parity: Parity
    g_parity_unchecked: Parity | None
    charge: Charge
    charge_unchecked: Charge | None
    isospin: Isospin
    isospin_unchecked: Isospin | None
    strangeness: int
    strangeness_unchecked: int | None
    charm: int
    charm_unchecked: int | None
    bottomness: int
    bottomness_unchecked: int | None
    topness: int
    topness_unchecked: int | None
    baryon_number: int
    baryon_number_unchecked: int | None
    electron_lepton_number: int
    electron_lepton_number_unchecked: int | None
    muon_lepton_number: int
    muon_lepton_number_unchecked: int | None
    tau_lepton_number: int
    tau_lepton_number_unchecked: int | None
    statistics: Statistics
    statistics_unchecked: Statistics | None
    def __init__(
        self,
        name: str | None = None,
        *,
        species: str | None = None,
        antiparticle_species: str | None = None,
        self_conjugate: bool | None = None,
        spin: QuantumNumber | None = None,
        parity: Parity | Sign | None = None,
        c_parity: Parity | Sign | None = None,
        g_parity: Parity | Sign | None = None,
        charge: Charge | QuantumNumber | None = None,
        isospin: Isospin | None = None,
        strangeness: int | None = None,
        charm: int | None = None,
        bottomness: int | None = None,
        topness: int | None = None,
        baryon_number: int | None = None,
        electron_lepton_number: int | None = None,
        muon_lepton_number: int | None = None,
        tau_lepton_number: int | None = None,
        statistics: Statistics | str | None = None,
    ) -> None: ...

class PartialWave:
    j: int | Fraction
    l: int
    s: int | Fraction
    label: str
    def __init__(
        self,
        *,
        j: QuantumNumber,
        l: QuantumNumber,
        s: QuantumNumber,
        label: str | None = None,
    ) -> None: ...

class AllowedPartialWave:
    wave: PartialWave
    parity: Parity | None
    c_parity: Parity | None

class RuleSet:
    def __init__(self) -> None: ...
    @staticmethod
    def angular() -> RuleSet: ...
    @staticmethod
    def strong() -> RuleSet: ...
    @staticmethod
    def electromagnetic() -> RuleSet: ...
    @staticmethod
    def weak() -> RuleSet: ...

class SelectionRules:
    def __init__(self, *, max_l: int = 6, rules: RuleSet | str | None = None) -> None: ...
    @staticmethod
    def coupled_spins(
        spin_1: QuantumNumber, spin_2: QuantumNumber
    ) -> list[int | Fraction]: ...
    def allowed_partial_waves(
        self,
        parent: ParticleProperties,
        daughter_1: ParticleProperties,
        daughter_2: ParticleProperties,
    ) -> list[AllowedPartialWave]: ...

def allowed_projections(spin: QuantumNumber) -> list[int | Fraction]: ...
def coupled_spins(
    spin_1: QuantumNumber, spin_2: QuantumNumber
) -> list[int | Fraction]: ...
def allowed_partial_waves(
    parent: ParticleProperties,
    daughter_1: ParticleProperties,
    daughter_2: ParticleProperties,
    *,
    max_l: int = 6,
    rules: RuleSet | str | None = None,
) -> list[AllowedPartialWave]: ...

__all__ = [
    'AllowedPartialWave',
    'Charge',
    'Isospin',
    'Parity',
    'PartialWave',
    'ParticleProperties',
    'QuantumNumber',
    'RuleSet',
    'SelectionRules',
    'Sign',
    'Statistics',
    'allowed_partial_waves',
    'allowed_projections',
    'coupled_spins',
]
