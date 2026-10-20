# Formula reference

## 1. Sphere and comb conventions

Let

```text
V_(d0)(0), V_(dz)(z), V_(mu1)(t1), ..., V_(mu_m)(t_m),
V_(d1)(1), V_(dinf)(infinity),       m=n-4.
```

There are `r=n-3=m+1` internal weights. Write

```text
h_i = H + a_i,   a_1=0.
```

The large-weight seed holds `a_i`, all external weights, `c`, and the segment
nomes fixed while `H` tends to infinity.

## 2. Exact aligned map

Set `q=product_i p_i` and

```text
z = 16 q product_(k>=1) [(1+q^(2k))/(1+q^(2k-1))]^8.
```

For the `j`th mobile position define the aggregate right nome

```text
R_j = product_(i=j+1)^r p_i.
```

Then

```text
t_j = T(R_j,q) = 4 R_j Y(q/R_j,R_j),
```

where

```text
Y(pL,pR) = (1+pL)^2
  product_(k>=1) [(1+q^(2k))/(1+q^(2k-1))]^4
  product_(k>=1) [
    (1+pL^(2k+1)pR^(2k))(1+pL^(2k-1)pR^(2k)) /
    ((1+pL^(2k)pR^(2k-1))(1+pL^(2k-2)pR^(2k-1)))
  ]^2,
q=pL pR.
```

On the ordered real cell, each equation `T(R_j,q)=t_j` is inverted
independently. The individual nomes are

```text
p_1=q/R_1,
p_i=R_(i-1)/R_i       (2<=i<=r-1),
p_r=R_(r-1).
```

## 3. Effective plumbing parameters and seed

Define

```text
rho_i = 4^(delta_(i,1)+delta_(i,r)) p_i,
product_i rho_i = 16q.
```

The pillow matrix element has the fixed-difference asymptotic

```text
M_n ~ [product_i rho_i^(h_i-c/24)] chi_pill(q),
chi_pill(q)=product_(k>=1)(1-q^(2k))^(-1/2).
```

After stripping this seed, the reduced block `H_n` tends to one.

## 4. Degenerate data

Use

```text
c=1+6Q^2,   Q=b+b^(-1),
h_(alpha,beta)=[Q^2-(alpha b+beta/b)^2]/4.
```

The implementation uses

```text
A_(alpha,beta) = 1/2 product'_(p,l) (p b+l/b)^(-1),
```

with `p=1-alpha,...,alpha`, `l=1-beta,...,beta`, excluding `(0,0)` and
`(alpha,beta)`.

For a weight `d=(Q^2-lambda_d^2)/4`, the fusion polynomial is

```text
P_(alpha,beta)[d_top/d_bottom]
 = product_(p=1-alpha step 2)^(alpha-1)
   product_(l=1-beta step 2)^(beta-1)
   [(lambda_top+lambda_bottom+p b+l/b)/2]
   [(lambda_top-lambda_bottom+p b+l/b)/2].
```

## 5. General recursion

At edge `i`, let `a_i` denote the current fixed difference. The adjacent
fusion factors are

```text
L_i = P[d0/dz]                                      i=1,
L_i = P[h_ab+a_(i-1)-a_i / mu_(i-1)]               i>1,

R_i = P[h_ab+a_(i+1)-a_i / mu_i]                   i<r,
R_i = P[dinf/d1]                                    i=r.
```

Then

```text
H_n = 1 + sum_i sum_(alpha,beta>=1)
  rho_i^(alpha beta) A_(alpha,beta) L_i R_i /
  (H+a_i-h_(alpha,beta))
  * H_n(T_(i;alpha,beta)(H,a); p).
```

The shifts are

```text
i=1:
  H -> h_(alpha,beta)+alpha beta,
  a_j -> a_j-alpha beta  for j>1;

i>1:
  H -> h_(alpha,beta)-a_i,
  a_i -> a_i+alpha beta,
```

with the remaining differences fixed. In a coefficient table in raw `p_i`,
the residue multiplier is `16^(alpha beta)` for the unique four-point edge,
`4^(alpha beta)` for either endpoint of a multi-edge block, and one for every
interior edge.

## 6. Sphere reconstruction

For arbitrary `kappa`,

```text
Lambda_n^(kappa) =
 theta3(q)^[kappa/2-4(d0+dz+d1+dinf)-2 sum_j mu_j]
 z^[kappa/24-d0-dz]
 (1-z)^[kappa/24-dz-d1]
 product_j [t_j(1-t_j)(z-t_j)]^(-mu_j/2).
```

The character-absorbed form used by the package is

```text
F_n^sphere = Lambda_n^(c-1)
             product_i rho_i^[h_i-(c-1)/24]
             H_n.
```

On the ordered real cell the implementation replaces `z-t_j` by `t_j-z` to
select the positive real branch. This phase convention must be revisited when
analytically continuing complex moduli.

