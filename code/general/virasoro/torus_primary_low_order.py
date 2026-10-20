"""Vectorized c=25 torus primary elliptic H through level four.

Coefficients follow inverse Gram matrices and primary Ward identities.
The generator and independent reference checks accompany the tail audit.
"""
import numpy as np

def torus_primary_h(q, h, e, order=4):
    if order not in range(5): raise ValueError('order must be 0..4')
    q,h,e=np.broadcast_arrays(np.asarray(q,complex),np.asarray(h,float),np.asarray(e,float))
    if np.any(h<=0): raise ValueError('internal weights must be positive')
    result=np.ones(q.shape,dtype=complex)
    if order >= 1:
        result += ((1/2)*e*(e - 1)/h) * q**1
    if order >= 2:
        result += ((1/4)*e*(e - 1)*(8*e**2*h + 25*e**2 - 56*e*h - 25*e + 96*h**2 + 120*h)/(h*(4*h + 5)**2)) * q**2
    if order >= 3:
        result += ((1/24)*e*(e - 1)*(8*e**4*h**2 + 83*e**4*h + 275*e**4 - 160*e**3*h**2 - 1254*e**3*h - 950*e**3 + 288*e**2*h**3 + 3200*e**2*h**2 + 5971*e**2*h + 2475*e**2 - 1824*e*h**3 - 6888*e*h**2 - 7200*e*h - 1800*e + 768*h**4 + 4224*h**3 + 6960*h**2 + 3600*h)/(h*(h + 3)**2*(4*h + 5)**2)) * q**3
    if order >= 4:
        result += ((1/384)*e*(e - 1)*(256*e**6*h**4 + 6592*e**6*h**3 + 69964*e**6*h**2 + 360483*e**6*h + 672525*e**6 - 9984*e**5*h**4 - 233920*e**5*h**3 - 2008196*e**5*h**2 - 6136185*e**5*h - 5016375*e**5 + 18432*e**4*h**5 + 563712*e**4*h**4 + 6374336*e**4*h**3 + 27766804*e**4*h**2 + 47505921*e**4*h + 26272575*e**4 - 344064*e**3*h**5 - 7428352*e**3*h**4 - 48438080*e**3*h**3 - 131634156*e**3*h**2 - 150131763*e**3*h - 54474525*e**3 + 208896*e**2*h**6 + 6031872*e**2*h**5 + 51250944*e**2*h**4 + 191886624*e**2*h**3 + 341581680*e**2*h**2 + 265619304*e**2*h + 61122600*e**2 - 1314816*e*h**6 - 18424320*e*h**5 - 100847616*e*h**4 - 269656992*e*h**3 - 357081696*e*h**2 - 203873760*e*h - 28576800*e + 344064*h**7 + 5984256*h**6 + 41382912*h**5 + 145112832*h**4 + 270573696*h**3 + 252771840*h**2 + 92534400*h)/(h*(h + 3)**3*(4*h + 5)**2*(4*h + 21)**2)) * q**4
    return result
