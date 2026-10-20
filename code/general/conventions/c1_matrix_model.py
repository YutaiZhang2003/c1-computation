"""Post-integration matrix-model comparisons in mu^-1=2*pi*g_s units."""
import math


def amplitude_coefficient(genus, outgoing_energies):
    """A^(h)_(1->n), with g_s^(n-1+2h) stripped, for the amplitudes in eq. (1.3)."""
    energies = tuple(complex(w) for w in outgoing_energies)
    if not energies or not all(math.isfinite(w.real) and math.isfinite(w.imag) for w in energies):
        raise ValueError('finite outgoing energies required')
    n = len(energies)
    w = sum(energies)
    if genus == 0 and 2 <= n <= 5:
        return (2*math.pi)**(n-1) * 1j * w * math.prod(energies) * math.prod(k+1j*w for k in range(1,n-1))
    if genus == 1 and n == 1:
        return math.pi**2 / 6 * (1j*w**2 + 2j*w**4 - w**5)
    if genus == 1 and n == 2:
        return -1j*math.pi**3/3 * energies[0]*energies[1]*w*(w-1j)*(w-2j)*(energies[0]**2+energies[1]**2-1j*w+1)
    raise ValueError('matrix coefficient implemented only for tree 1->2..5 and torus 1->1..2')
