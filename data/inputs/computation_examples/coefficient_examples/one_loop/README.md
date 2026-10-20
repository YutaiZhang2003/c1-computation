# Liouville block coefficient examples

These small reference tables use the same conventions and source APIs as the package's computational guide. Parameters, coefficient generator, edge order, precision policy, excluded factors and an example evaluated value are recorded in the JSON beside each CSV. Internal reference momenta are deliberately simple unequal values; the tables are not complete production banks.

## Table inventory

| Table | Rows | Expansion / truncation |
| --- | ---: | --- |
| [torus_comb_t025_C444.csv](torus_comb_t025_C444.csv) · [parameters](torus_comb_t025_C444.json) | 125 | rectangular_per_axis [4, 4, 4] |
| [torus_necklace_t025_C444.csv](torus_necklace_t025_C444.csv) · [parameters](torus_necklace_t025_C444.json) | 125 | rectangular_per_axis [4, 4, 4] |
| [torus_pair_t025_C444.csv](torus_pair_t025_C444.csv) · [parameters](torus_pair_t025_C444.json) | 125 | rectangular_per_axis [4, 4, 4] |

## Reading and using a table

For a q-series, rows contain nonnegative integer powers `n1,n2[,n3]` and the real/imaginary parts of `C[n]`. The constant term is one. Evaluate

```text
F(q) = sum_n C[n] * product_j q[j]^n[j].
```

The JSON's `q_example` is a list of `[real,imaginary]` pairs in the declared variable order. A total-degree cap means `sum(n)<=N`; a rectangular cap means each `n_j<=N_j`. These are different truncations. The JSON provides `value_at_q` and `value_at_conjugate_q`. The continuation keeps the same coefficient table in both chiral factors; do not conjugate externally continued parameters without checking the stated convention.

For a genus-two c-pole table, `inverse_power=0` holds the constant (its pole columns are placeholders). The other rows represent `A/(c-pole)^inverse_power`. Sum them at c=25 using the **fixed q in the JSON**. A c-pole table is not a list of q monomials. Vacuum/global-seed settings are included in its metadata.

The JSON files and `generation_receipt.json` preserve the original generation settings. These reference examples are checksum-verified; the historical standalone exporters are not part of this repository.

## From a block to a correlator

A descendant series by itself is not a Liouville correlator. Restore the channel's primary propagation, multiply the appropriate Yin three-point constants, include the prescribed momentum measure and sum the complete momentum rule. Then add matter/ghost factors, analytic subtractions and moduli weights at the stage specified in the package README. The five-point continued contour also requires its crossed-pole contribution. Native one-loop reference tables do not replace the boundary kernel's projection/elliptic completion. Genus-two q-series and fixed-q c-pole tables must not be multiplied together or given a second vacuum factor.

See the [repository README](../../../../../README.md) for standalone recovery, conventions and production entry points.
