# Handoff manifest

Package: `genus0-elliptic-h-recursion`  
Version: `0.1.0`  
Python: `>=3.9`  
Runtime dependency: `mpmath>=1.2`

## Recommended transfer

Send either:

1. `genus0_elliptic_h_recursion-0.1.0-py3-none-any.whl` for direct
   installation; or
2. `genus0_elliptic_h_recursion-0.1.0-source.zip` when the receiving project
   should retain source, formulas, examples, and tests.

Install the wheel with

```bash
python -m pip install genus0_elliptic_h_recursion-0.1.0-py3-none-any.whl
```

or unpack the source ZIP and run

```bash
python -m pip install .
python -m unittest discover -s tests -v
```

## Contents

- `src/genus0_elliptic_h_recursion/recursion.py`: arbitrary-`n` coefficient
  engine and Kac-pole audit.
- `src/genus0_elliptic_h_recursion/geometry.py`: exact aligned pillow map and
  ordered-real inverse.
- `src/genus0_elliptic_h_recursion/block.py`: effective plumbing variables,
  complete `c-1` conformal prefactor, and sphere reconstruction.
- `FORMULAS.md`: self-contained formula dictionary.
- `VALIDATION.md`: validated cases and explicit research limitations.
- `examples/`: executable six-point example.
- `tests/`: repository-independent regression suite.

## Important downstream note

The arbitrary-`n` implementation is a research proposal whose direct PBW
validation currently reaches `n=6`. Do not erase the limitations in
`VALIDATION.md` when incorporating it into another codebase.

