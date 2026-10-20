# Validation status and limitations

## Algebraic checks completed in the development project

- **Four points:** the corrected sphere-to-pillow normalization was checked
  analytically against direct Virasoro PBW data through level four and
  numerically through level six.
- **Five points:** all symbolic coefficients through total degree four agree
  with direct PBW computation. At total degree ten, all 66 coefficients were
  compared at five asymmetric generic weight choices using exact rational PBW
  arithmetic and an independent 80-digit recursion evaluation.
- **Six points:** all 20 coefficients through total degree three agree
  symbolically for generic weights. All 286 coefficients through total degree
  ten were then checked at ten asymmetric generic internal-weight triples.
  Across 2,860 comparisons, the largest relative error was
  `3.44e-69`. Every sample was audited away from every Kac denominator visited
  by the recursion.

## Portable-engine equivalence gate

Before packaging, the arbitrary-`n` engine in this distribution was compared
directly with the original specialized five- and six-point recursion programs
at 80 decimal digits through total degree ten. All 66 five-point coefficients
agreed with maximum absolute discrepancy `2.76e-76`; all 286 six-point
coefficients agreed exactly at the working precision. This gate checks that the
general edge and shift logic was transferred without changing the validated
special cases.

## Finite-moduli value check

For the six-point example

```text
c=26.215,
external=(0.17,0.29,0.43,0.58,0.71,0.86),
internal=(0.9371,1.0837,1.3321),
t1=0.32, t2=0.62, 0.015<=z<=0.20,
```

the order-ten elliptic reconstruction was compared at 61 points with an
independent fixed-weight central-charge recursion through total degree 20.
The maximum relative difference was `2.90e-5`, smaller pointwise than the sum
of the observed `h`- and `c`-recursion shifts.

Extending only the elliptic recursion to order 12 changed the order-ten result
by at most `9.09e-7` and by `3.19e-7` at the median point. Thus the value
comparison was limited by the slower central-charge series rather than the
order-ten elliptic truncation.

## What remains a proposal

The implementation supports arbitrary `n`, but direct PBW validation currently
ends at `n=6`. In particular, the assumption that every additional middle
vertex reduces to the identity on leading large-weight oscillator labels, and
therefore produces no new regular character, has not yet been independently
checked for `n>=7`.

Automatic coordinate inversion is validated only on the ordered real aligned
cell. Complex moduli require an external analytic-continuation prescription
for the segment nomes and square-root branches.

These restrictions are surfaced here so that downstream projects do not treat
the arbitrary-`n` API as a theorem beyond its present validation frontier.
