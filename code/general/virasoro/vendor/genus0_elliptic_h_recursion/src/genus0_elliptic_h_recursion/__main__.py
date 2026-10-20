"""Run a compact six-point demonstration with ``python -m``."""

from __future__ import annotations

import mpmath as mp

from .block import reconstruct_from_real_moduli
from .recursion import compute_h_recursion


def main() -> None:
    mp.mp.dps = 50
    table = compute_h_recursion(
        central_charge="26.215",
        external_weights=("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
        internal_weights=("0.9371", "1.0837", "1.3321"),
        order=6,
        dps=50,
        pole_tolerance="1e-10",
    )
    result = reconstruct_from_real_moduli(
        table,
        z="0.1075",
        mobile_positions=("0.32", "0.62"),
    )
    print("segment nomes:", tuple(mp.nstr(value, 14) for value in result.nomes.segment_nomes))
    print("H6 order 6:  ", mp.nstr(result.reduced_value, 18))
    print("F6 order 6:  ", mp.nstr(result.value, 18))
    if table.minimum_pole is not None:
        print("minimum |Kac denominator|:", mp.nstr(table.minimum_pole.magnitude, 12))


if __name__ == "__main__":
    main()

