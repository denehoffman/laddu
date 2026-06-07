from fractions import Fraction
from itertools import product

import pytest
from laddu import J, L, M, S, allowed_projections


def test_typed_angular_momentum_construction_and_alias() -> None:
    assert S is J
    assert J(1).value == 1
    assert J(1.5).value == Fraction(3, 2)
    assert J(Fraction(3, 2)).value == Fraction(3, 2)
    assert S(Fraction(1, 2)) == J(Fraction(1, 2))
    assert L(2).value == 2
    assert L(2.0).value == 2
    assert L(Fraction(2, 1)).value == 2
    assert M(-1).value == -1
    assert M(-0.5).value == Fraction(-1, 2)
    assert M(Fraction(-1, 2)).value == Fraction(-1, 2)


def test_typed_projection_methods_return_m_values() -> None:
    assert J(3 / 2).projections() == [
        M(-3 / 2),
        M(-1 / 2),
        M(1 / 2),
        M(3 / 2),
    ]
    assert L(1).projections() == [M(-1), M(0), M(1)]


def test_allowed_projections_returns_physical_values() -> None:
    assert allowed_projections(0) == [0]
    assert allowed_projections(Fraction(1, 2)) == [Fraction(-1, 2), Fraction(1, 2)]
    assert allowed_projections(1.0) == [-1, 0, 1]
    assert allowed_projections(J(1 / 2)) == [Fraction(-1, 2), Fraction(1, 2)]


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

    with pytest.raises(RuntimeError, match='orbital angular momentum must be integer'):
        L(Fraction(1, 2))
