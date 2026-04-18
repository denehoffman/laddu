from fractions import Fraction

import pytest
from laddu import allowed_projections, helicity_combinations


def test_allowed_projections_returns_physical_values() -> None:
    assert allowed_projections(0) == [0]
    assert allowed_projections(Fraction(1, 2)) == [Fraction(-1, 2), Fraction(1, 2)]
    assert allowed_projections(1.0) == [-1, 0, 1]


def test_helicity_combinations_returns_physical_values() -> None:
    assert helicity_combinations(0, 0) == [(0, 0, 0)]
    assert helicity_combinations(Fraction(1, 2), 0.5) == [
        (Fraction(-1, 2), Fraction(-1, 2), 0),
        (Fraction(-1, 2), Fraction(1, 2), -1),
        (Fraction(1, 2), Fraction(-1, 2), 1),
        (Fraction(1, 2), Fraction(1, 2), 0),
    ]


def test_projection_helpers_reject_invalid_quantum_numbers() -> None:
    with pytest.raises(ValueError, match='integer or half-integer'):
        allowed_projections(0.25)

    with pytest.raises(ValueError, match='integer or half-integer'):
        helicity_combinations(Fraction(1, 3), 0)
