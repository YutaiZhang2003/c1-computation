# Genus-zero elliptic h-recursion

Standalone high-precision Python implementation of the proposed fixed-difference
elliptic `h`-recursion for a chiral Virasoro sphere `n`-point block in the comb
(open-necklace) channel.

The package has one runtime dependency: `mpmath`. It does not import anything
from the project in which it was developed.

## Conventions

Place the operators at

```text
(0, z, t1, ..., t_(n-4), 1, infinity),   0 < z < t1 < ... < 1
```

and supply external weights in exactly this order:

```text
(d_0, d_z, mu_1, ..., mu_(n-4), d_1, d_infinity).
```

There are `r=n-3` internal weights, ordered from the `(0,z)` end toward the
`(1,infinity)` end. The reduced block is expanded in raw segment nomes
`(p_1,...,p_r)`, with `q=product(p_i)`.

The effective propagation variables are

```text
rho_i = 4^(delta_(i,1)+delta_(i,r)) p_i.
```

Thus `rho=(16q,)` at four points, `(4p1,4p2)` at five points, and
`(4p1,p2,...,p_(r-1),4p_r)` for `r>2`. Interior segments do **not** receive
an extra factor of four.

## Installation

From this directory:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

## Minimal six-point example

```python
import mpmath as mp

from genus0_elliptic_h_recursion import (
    compute_h_recursion,
    reconstruct_from_real_moduli,
)

mp.mp.dps = 50

table = compute_h_recursion(
    central_charge="26.215",
    external_weights=(
        "0.17", "0.29",       # at 0,z
        "0.43", "0.58",       # at t1,t2
        "0.71", "0.86",       # at 1,infinity
    ),
    internal_weights=("0.9371", "1.0837", "1.3321"),
    order=10,
    dps=50,
    pole_tolerance="1e-10",
)

answer = reconstruct_from_real_moduli(
    table,
    z="0.1075",
    mobile_positions=("0.32", "0.62"),
)

print(answer.nomes.segment_nomes)
print(answer.reduced_value)  # H_6(p1,p2,p3)
print(answer.value)          # plane-normalized chiral sphere block F_6
```

Use decimal strings rather than binary floats when reproducibility at high
precision matters.

The command

```bash
python -m genus0_elliptic_h_recursion
```

runs a shorter order-six version of this example.

## Main API

- `compute_h_recursion(...)` builds the total-degree-truncated coefficient
  table for the reduced block `H_n`.
- `RecursionTable.evaluate(p, order=N)` evaluates a truncation.
- `RecursionTable.shell(p, N)` returns a single homogeneous shell and is useful
  for convergence diagnostics.
- `invert_aligned_coordinates(z, mobile_positions)` performs the exact real
  aligned-cover inversion.
- `coordinates_from_segment_nomes(p)` evaluates the forward product map.
- `reconstruct_sphere_block(...)` restores the full `c-1` conformal prefactor.
- `reconstruct_from_real_moduli(...)` combines inversion and reconstruction.

The returned `RecursionTable.minimum_pole` records the smallest Kac denominator
visited. Set `pole_tolerance` to reject unsafe parameter choices.

## Complex moduli and branch choices

Automatic coordinate inversion is intentionally restricted to the ordered real
cell. For complex moduli, analytically continue or determine the segment nomes
externally and pass them directly to `RecursionTable.evaluate` and
`reconstruct_sphere_block`.

`lambda_prefactor(..., branch="ordered_real")` uses positive `t_j-z` factors.
For a literal holomorphic branch use `branch="holomorphic"`, which keeps
`z-t_j`; the caller is responsible for consistent analytic continuation.

## Tests

```bash
python -m unittest discover -s tests -v
```

The portable tests cover four-, five-, and six-point coefficient snapshots,
the special endpoint factors, exact coordinate inversion, and full six-point
reconstruction. See [VALIDATION.md](VALIDATION.md) for the stronger PBW checks
performed in the development project.

## Scope

This package computes a **chiral Virasoro block**, with primary three-point
constants stripped. A nonchiral contribution is obtained by multiplying by
the corresponding antiholomorphic block. It does not sum over internal states
or supply a CFT spectrum/structure constants.

The general `n`-point recursion is a research proposal. Its current validation
frontier is stated explicitly in [VALIDATION.md](VALIDATION.md).

