#!/usr/bin/env python3
"""Independent checks of the CCY sphere five-point linear-channel block."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

try:
    from ccy_sphere_five_point import (
        sphere_five_point_c_coefficients,
        sphere_five_point_direct_coefficients,
        sphere_five_point_global_coefficients,
        sphere_five_point_h_c25_limit,
        sphere_five_point_h_coefficients,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from ccy_sphere_five_point import (
        sphere_five_point_c_coefficients,
        sphere_five_point_direct_coefficients,
        sphere_five_point_global_coefficients,
        sphere_five_point_h_c25_limit,
        sphere_five_point_h_coefficients,
    )


def _relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(abs(complex(target)), 1.0e-300)


def _maximum_table_error(
    observed: dict[tuple[int, int], complex],
    target: dict[tuple[int, int], complex],
) -> tuple[float, tuple[int, int]]:
    errors = {
        key: _relative_error(observed[key], target[key])
        for key in target
    }
    key = max(errors, key=errors.get)
    return errors[key], key


def check_global_level_one() -> None:
    external = (0.17, 0.29, 0.43, 0.58, 0.71)
    internal = (0.93, 1.08)
    direct = sphere_five_point_direct_coefficients(
        central_charge=10**9,
        external_weights=external,
        internal_weights=internal,
        order1=1,
        order2=1,
    )
    global_table = sphere_five_point_global_coefficients(
        external_weights=external,
        internal_weights=internal,
        order1=1,
        order2=1,
    )
    # Through one level on either edge, every Virasoro descendant is global.
    error, key = _maximum_table_error(global_table, direct)
    print("global five-point block at level one")
    print(f"  max relative error={error:.3e} at {key}")
    if error > 2.0e-15:
        raise AssertionError("global five-point coefficient has the wrong convention")


def check_h_and_c_recursions_against_direct() -> None:
    cases = (
        (26.215, (0.17, 0.29, 0.43, 0.58, 0.71), (0.93, 1.08)),
        (31.7, (0.21, 0.34, 0.49, 0.63, 0.79), (1.03, 1.19)),
    )
    order = 4
    print("\nfive-point h/c recursions vs descendant definition")
    for index, (central_charge, external, internal) in enumerate(cases, start=1):
        direct = sphere_five_point_direct_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            max_total_order=order,
        )
        h_recursive = sphere_five_point_h_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            max_total_order=order,
        )
        c_recursive = sphere_five_point_c_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            max_total_order=order,
        )
        h_error, h_key = _maximum_table_error(h_recursive, direct)
        c_error, c_key = _maximum_table_error(c_recursive, direct)
        hc_error, hc_key = _maximum_table_error(h_recursive, c_recursive)
        print(
            f"  case {index}: h/direct={h_error:.3e} at {h_key}, "
            f"c/direct={c_error:.3e} at {c_key}, h/c={hc_error:.3e} at {hc_key}"
        )
        if max(h_error, c_error, hc_error) > 5.0e-10:
            raise AssertionError(f"case {index} five-point recursions disagree")


def check_c25_h_regulator() -> None:
    external = (1.17, 1.09, 1.13, 1.21, 1.07)
    internal = (1.31, 1.47)
    order = 4
    exact = sphere_five_point_c_coefficients(
        central_charge=25.0,
        external_weights=external,
        internal_weights=internal,
        order1=order,
        order2=order,
        max_total_order=order,
    )
    regulated, estimates = sphere_five_point_h_c25_limit(
        external_weights=external,
        internal_weights=internal,
        order1=order,
        order2=order,
        max_total_order=order,
    )
    error, key = _maximum_table_error(regulated, exact)
    estimated = max(abs(value) for value in estimates.values())
    print("\nc=25 resonant h-recursion regulator")
    print(f"  max h/c relative error={error:.3e} at {key}")
    print(f"  max absolute fit-shift estimate={estimated:.3e}")
    if error > 2.0e-5:
        raise AssertionError("regulated c=25 h-recursion has not converged")


def run() -> None:
    check_global_level_one()
    check_h_and_c_recursions_against_direct()
    check_c25_h_regulator()
    print("\nall sphere five-point CCY checks passed")


if __name__ == "__main__":
    run()
