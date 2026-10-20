# Observable conventions

Production exports use

\[
\mu^{-1}=2\pi g_s,\qquad
\mathcal A_{1\to n}=\sum_h g_s^{n-1+2h}\mathcal A^{(h)}_{1\to n},\qquad
F=-\frac{\log Z}{2\pi R},\qquad F\big|_{h=2}=g_s^2F_2(R).
\]

`c1_string_conventions.py` distinguishes coefficients in powers of `mu^-1`,
current `g_s`, and the historical Xi coupling. Unknown conventions are rejected.
`c1_observables.py` converts final estimates, component errors, and covariance;
the numerical kernels and reduced CFT integrals retain their original units.
Matrix-model amplitude comparisons are in `c1_matrix_model.py` and are applied
after integration.

| Output | Quantity |
| --- | --- |
| Five-point scan `amplitude_gs` | Complex \(\mathcal A^{(0)}_{1\to4}=2\pi i I_5\), coefficient of \(g_s^3\) |
| Six-point `amplitudes_gs.json`, `.csv` | Complex \(\mathcal A^{(0)}_{1\to5}=-(2\pi)^4 5it^6Q_5\), coefficient of \(g_s^4\) |
| Genus-one summary `amplitude_gs` | Real \(\mathcal A^{(1)}_{1\to2}=-2\pi^2\operatorname{Im}I_{1,3}\), coefficient of \(g_s^3\) |
| Tail assembly `free_energy_F2.csv`, `.npz`, `.json` | Literal thermal \(F_2(R)\), coefficient of \(g_s^2\) |

The genus-one workers' existing statistics and absolute tolerance controls stay
in their recorded native units and are labelled `native_amplitude_convention`.
Read `amplitude_gs` for the current amplitude and its errors. Reduced `Q`, `Q5`,
`I1`, and `I_1_3` fields are not coupling-stripped amplitudes. Frozen sources,
receipts, and earlier exports retain their original conventions.

For genus two the BRY-v7 kernel integrates the coefficient of \(g_s^2\) in
\(\log Z\). Its conversion to \(F_2\) is \(-1/(2\pi R)\). The older
`g_convention_audit_20260911` instead stores the coefficient of \(\mu^{-2}\)
in \(\log Z\); its conversion is \(-2\pi/R\). Standard errors use the absolute
factor and cross-radius covariance uses \(C'_{ij}=s_iC_{ij}s_j\). The existing
automorphism and one-sided Fermi-sea factors are already included.

The matrix-model predictions are
\[
F_1=\frac{1+R^{-2}}{48\pi}\log\mu,\qquad
F_2=-\frac{\pi}{2880}(7+10R^{-2}+7R^{-4}).
\]
Thus \(-[7+10R^{-2}+7R^{-4}]/(11520\pi\mu^2)=g_s^2F_2\).
No extra \(g_s^2\) belongs on an axis labelled \(F_2(R)\).

Recover the current numerical results with `python -B reproduce.py` from the repository root.
