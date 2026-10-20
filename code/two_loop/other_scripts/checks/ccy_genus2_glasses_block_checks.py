#!/usr/bin/env python3
"""Checks for the CCY genus-two glasses-frame recursion."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

try:
    from ccy_genus2_block import b_from_c_rs_h, dc_rs_dh, format_complex
    from ccy_genus2_glasses_block import (
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        genus2_global_glasses_sl2_block_resummed,
        handle_residue_prefactor,
    )
    from virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        momentum_from_weight,
        torus_one_point_c_recursion_residue,
        torus_one_point_residue,
    )
    from virasoro_plumbing_graph import direct_plumbing_graph_block, genus2_glasses_graph
except ImportError:  # pragma: no cover - supports package-style execution
    from ccy_genus2_block import b_from_c_rs_h, dc_rs_dh, format_complex
    from ccy_genus2_glasses_block import (
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        genus2_global_glasses_sl2_block_resummed,
        handle_residue_prefactor,
    )
    from virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        momentum_from_weight,
        torus_one_point_c_recursion_residue,
        torus_one_point_residue,
    )
    from virasoro_plumbing_graph import direct_plumbing_graph_block, genus2_glasses_graph


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_order_zero_and_one() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    order0 = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=0,
        include_vacuum_seed=False,
    ).value
    order1 = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=1,
        include_vacuum_seed=False,
    ).value
    global1 = genus2_global_glasses_sl2_block(h_left, h_right, h_bridge, q_left, q_right, q_bridge, 1)
    print("order zero/one")
    print(f"  order0={order0!r}")
    print(f"  order1={order1!r}")
    print(f"  global1={global1!r}")
    require(abs(order0 - 1.0) < 1.0e-14, "order-zero block should be one without vacuum seed")
    require(abs(order1 - global1) < 1.0e-14, "order-one block should equal the global seed")


def check_resummed_global_block() -> None:
    h_left, h_right, h_bridge = 1.2, 1.7, 2.1
    q_left, q_right, q_bridge = 0.03 + 0.01j, 0.07 - 0.015j, 0.11 + 0.02j
    resummed = genus2_global_glasses_sl2_block_resummed(
        h_left,
        h_right,
        h_bridge,
        q_left,
        q_right,
        q_bridge,
        tolerance=1.0e-14,
    )
    high_order = genus2_global_glasses_sl2_block(
        h_left,
        h_right,
        h_bridge,
        q_left,
        q_right,
        q_bridge,
        order=20,
    )
    recursion_seed = ccy_genus2_glasses_block(
        c=26.215,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=0,
        include_vacuum_seed=False,
        resum_global_block=True,
        global_block_tolerance=1.0e-14,
    )
    print("\nresummed glasses global block")
    print(f"  resummed={resummed!r}")
    print(f"  order-20 difference={abs(resummed - high_order):.3e}")
    require(abs(resummed - high_order) < 1.0e-13, "glasses 2F1 factorization changed the global block")
    require(
        abs(recursion_seed.value - resummed) < 1.0e-14,
        "order-zero glasses recursion did not use the all-level global seed",
    )
    require(recursion_seed.global_block_resummed, "resummed glasses metadata was lost")


def check_handle_residue_matches_torus_one_point() -> None:
    r, s = 3, 2
    handle_weight = 1.37
    bridge_weight = 0.82
    b = b_from_c_rs_h(r, s, handle_weight)
    external_lambda = momentum_from_weight(bridge_weight, b)
    expected = -dc_rs_dh(r, s, handle_weight) * torus_one_point_residue(
        r,
        s,
        b,
        external_lambda,
    )
    observed = handle_residue_prefactor(r, s, handle_weight, bridge_weight)
    torus_c_recursion = torus_one_point_c_recursion_residue(
        r,
        s,
        handle_weight,
        bridge_weight,
    )
    print("\nhandle tadpole residue")
    print(f"  observed={observed!r}")
    print(f"  expected={expected!r}")
    require(abs(observed - expected) < 1.0e-10, "handle residue is not the torus one-point residue")
    require(
        abs(torus_c_recursion - observed) < 1.0e-10,
        "genus-one c-recursion residue is not the genus-two handle residue",
    )


def check_separating_limit_matches_two_tori() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 0.002, 0.0017, 1.0e-4
    order = 3
    glasses = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=5,
        vacuum_oscillator_level_max=40,
    ).value
    left = TorusOnePointVirasoroBlock(c, h_left, h_bridge).chiral_block(
        q_left,
        order,
        include_prefactor=False,
    )
    right = TorusOnePointVirasoroBlock(c, h_right, h_bridge).chiral_block(
        q_right,
        order,
        include_prefactor=False,
    )
    expected = left * right
    rel_error = abs(glasses - expected) / max(1.0e-30, abs(expected))
    print("\nseparating degeneration")
    print(f"  glasses={format_complex(glasses)}")
    print(f"  two tori={format_complex(expected)}")
    print(f"  rel error={rel_error:.6e}")
    require(rel_error < 1.0e-4, "glasses block does not approach the product of two torus blocks")


def check_order_three_finite() -> None:
    value = ccy_genus2_glasses_block(
        c=26.215,
        h_left=0.91,
        h_right=0.97,
        h_bridge=1.03,
        q_left=0.003 + 0.001j,
        q_right=0.0025 - 0.0007j,
        q_bridge=0.0012 + 0.0003j,
        order=3,
        include_vacuum_seed=True,
        vacuum_word_len=4,
        vacuum_oscillator_level_max=30,
    ).value
    print("\norder three finite")
    print(f"  value={value!r}")
    require(abs(value) > 0.0, "order-three value vanished unexpectedly")
    require(abs(value) < 10.0, "order-three value is implausibly large for tiny q")


def direct_glasses_descendant_block(
    *,
    c: complex,
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    order: int,
) -> complex:
    """Direct total-degree descendant sum for the glasses graph."""

    return direct_plumbing_graph_block(
        genus2_glasses_graph(),
        central_charge=c,
        edge_weights=(h_left, h_right, h_bridge),
        q_values=(q_left, q_right, q_bridge),
        max_total_level=order,
    ).value


def check_against_direct_descendant_sum() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 1.1e-4, 1.3e-4, 0.9e-4
    order = 3
    direct = direct_glasses_descendant_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
    )
    recursive = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=6,
        vacuum_oscillator_level_max=20,
    ).value
    relative = abs(recursive - direct) / abs(direct)
    print("\ndirect descendant-sum comparison")
    print(f"  direct={direct!r}")
    print(f"  recursive={recursive!r}")
    print(f"  relative difference={relative:.6e}")
    require(relative < 1.0e-10, "glasses recursion disagrees with the direct level-three descendant sum")


def run() -> None:
    check_order_zero_and_one()
    check_resummed_global_block()
    check_handle_residue_matches_torus_one_point()
    check_separating_limit_matches_two_tori()
    check_order_three_finite()
    check_against_direct_descendant_sum()
    print("\nall CCY glasses-frame checks passed")


if __name__ == "__main__":
    run()
