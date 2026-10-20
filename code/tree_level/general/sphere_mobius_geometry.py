"""Projective-sphere coordinates and Möbius maps, independent of multiplicity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

INFINITY = None
ProjectivePoint = Optional[complex]


def _homogeneous(point: ProjectivePoint) -> tuple[complex, complex]:
    if point is None:
        return 1.0 + 0.0j, 0.0 + 0.0j
    return complex(point), 1.0 + 0.0j


def _determinant(left: tuple[complex, complex], right: tuple[complex, complex]) -> complex:
    return left[0] * right[1] - left[1] * right[0]


@dataclass(frozen=True)
class MobiusMap:
    """A Mobius transformation ``w=(A z+B)/(C z+D)``."""

    a: complex
    b: complex
    c: complex
    d: complex

    @property
    def determinant(self) -> complex:
        return self.a * self.d - self.b * self.c

    def __post_init__(self) -> None:
        if self.determinant == 0.0:
            raise ValueError("a Mobius map must be nonsingular")

    def __call__(self, point: ProjectivePoint, *, tolerance: float = 1.0e-13) -> ProjectivePoint:
        z0, z1 = _homogeneous(point)
        numerator = self.a * z0 + self.b * z1
        denominator = self.c * z0 + self.d * z1
        scale = max(abs(numerator), abs(denominator), 1.0)
        if abs(denominator) <= float(tolerance) * scale:
            return None
        return complex(numerator / denominator)

    def local_scale(self, point: ProjectivePoint, *, tolerance: float = 1.0e-13) -> complex:
        r"""Return the derivative between canonical local coordinates.

        The canonical coordinate is ``z`` at a finite puncture and ``1/z``
        at infinity; the target convention is identical.  Its absolute value
        gives the primary covariance factor without special-case limits.
        """

        delta = self.determinant
        target = self(point, tolerance=tolerance)
        if point is None:
            if target is None:
                if self.a == 0.0:
                    raise ZeroDivisionError("indeterminate infinity-to-infinity scale")
                return complex(self.d / self.a)
            if self.c == 0.0:
                raise ZeroDivisionError("indeterminate infinity-to-finite scale")
            return complex(-delta / (self.c * self.c))

        z = complex(point)
        if target is None:
            numerator = self.a * z + self.b
            if numerator == 0.0:
                raise ZeroDivisionError("indeterminate finite-to-infinity scale")
            return complex(-delta / (numerator * numerator))
        denominator = self.c * z + self.d
        if denominator == 0.0:
            raise ZeroDivisionError("indeterminate finite Mobius derivative")
        return complex(delta / (denominator * denominator))


def mobius_to_zero_one_infinity(
    zero_point: ProjectivePoint,
    one_point: ProjectivePoint,
    infinity_point: ProjectivePoint,
) -> MobiusMap:
    """Return the unique map sending three distinct points to ``0,1,infinity``."""

    zero_h = _homogeneous(zero_point)
    one_h = _homogeneous(one_point)
    infinity_h = _homogeneous(infinity_point)
    normalization_denominator = _determinant(one_h, zero_h)
    if normalization_denominator == 0.0:
        raise ValueError("zero_point and one_point must be distinct")
    normalization = _determinant(one_h, infinity_h) / normalization_denominator
    # det((z,1),zero) / det((z,1),infinity) times normalization.
    a = normalization * zero_h[1]
    b = -normalization * zero_h[0]
    c = infinity_h[1]
    d = -infinity_h[0]
    return MobiusMap(a, b, c, d)
