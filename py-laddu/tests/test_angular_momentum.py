from fractions import Fraction
from itertools import product

import pytest
from laddu import allowed_projections


def test_allowed_projections_returns_physical_values() -> None:
    assert allowed_projections(0) == [0]
    assert allowed_projections(Fraction(1, 2)) == [Fraction(-1, 2), Fraction(1, 2)]
    assert allowed_projections(1.0) == [-1, 0, 1]


def test_projection_products_are_explicit_for_helicity_sums() -> None:
    pairs = list(
        product(
            allowed_projections(Fraction(1, 2)),
            allowed_projections(Fraction(1, 2)),
        )
    )
    assert pairs == [
        (Fraction(-1, 2), Fraction(-1, 2)),
        (Fraction(-1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    ]
    assert [lambda_1 - lambda_2 for lambda_1, lambda_2 in pairs] == [0, -1, 1, 0]


def test_projection_helpers_reject_invalid_quantum_numbers() -> None:
    with pytest.raises(RuntimeError, match='integer or half-integer'):
        allowed_projections(0.25)
