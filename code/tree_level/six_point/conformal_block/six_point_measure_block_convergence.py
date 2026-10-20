"""Same-node finite-prefix diagnostics for the native six-point expansion.

Absolute node increments expose cancellations in the spectral sum. Neither a
small observed increment nor this utility certifies an omitted block tail or
the momentum rule. A caller must separately require enough prefixes and decide
its numerical qualification policy.
"""
from __future__ import annotations

import operator
import numpy as np

from six_point_batch_block_contraction import SphereBatchEvaluator

VERSION = 'sphere-six-point-native-prefix-diagnostics-v1'


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(f'{name} must be an integer')
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc


def evaluate_native_prefixes(evaluator, q_values, *, base_logarithms=0j,
                             orders=None, batch_size=16):
    """Return matched values and absolute nonchiral increments at every point.

    ``values[N]`` has the same meaning as ``evaluator.evaluate(...)[N]``.
    For each consecutive requested pair (L,N), ``absolute_weighted_increments``
    is sum_p exp(Re(log_weight_p)) * abs(C_N(q) C_N(qbar) -
    C_L(q) C_L(qbar)). Both chiral factors use the SAME coefficient table,
    including complex coefficients. The signed increment is the difference
    between the complete spectral sums. All requested orders share all nodes.

    The relative absolute increment divides by abs(the highest-order full
    spectral sum), never by a subset, previous order or sum of absolute node
    contributions. At an exactly zero final sum the normalization is undefined:
    a positive numerator returns infinity, zero returns NaN, and the explicit
    ``normalization_defined`` mask is false. No point is dropped or promoted to
    a convergence pass. Nonfinite computations fail closed.

    The already-bound SphereBatchEvaluator owns complete-table validation.
    This function supports one or more prefixes; requiring at least three for
    a convergence decision is the caller's responsibility.
    """
    if not isinstance(evaluator,SphereBatchEvaluator):
        raise TypeError('an already-validated SphereBatchEvaluator is required')
    q=np.asarray(q_values,dtype=complex)
    if q.ndim==1:q=q[None,:]
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or np.any(abs(q)==0):
        raise ValueError('finite nonzero native q triples required')
    count=len(q)
    base=np.broadcast_to(np.asarray(base_logarithms,dtype=complex),(count,))
    if not np.isfinite(base).all():raise ValueError('nonfinite geometry factor')
    requested=(evaluator.maximum_order,) if orders is None else orders
    orders=tuple(sorted({_integer(n,'order') for n in requested}))
    if not orders or min(orders)<0 or max(orders)>evaluator.maximum_order:
        raise ValueError('requested order outside the coefficient bank')
    batch_size=_integer(batch_size,'batch size')
    if batch_size<=0:raise ValueError('positive integral batch size required')
    pairs=tuple(zip(orders[:-1],orders[1:]))
    values=np.empty((count,len(orders)),dtype=complex)
    absolute=np.empty((count,len(pairs)),dtype=float)
    underflow_counts=np.zeros(count,dtype=np.int64)
    for start in range(0,count,batch_size):
        stop=min(start+batch_size,count);qs=q[start:stop];width=len(qs)
        # Match the existing batched contraction and shell accumulation order.
        powers=np.prod(qs[:,None,:]**evaluator.keys[None,:,:],axis=2).T
        monomials=np.empty((len(evaluator.keys),2*width),dtype=complex)
        monomials[:,::2]=powers;monomials[:,1::2]=powers.conjugate()
        logarithms=(evaluator.logs[:,None]+base[start:stop][None,:]
                    +np.einsum('pk,qk->pq',evaluator.primary,np.log(abs(qs)),optimize=False))
        with np.errstate(over='raise',invalid='raise'):
            weights=np.exp(logarithms)
            absolute_weights=np.exp(logarithms.real)
        underflow_counts[start:stop]=np.sum(absolute_weights==0,axis=0)
        chiral=np.zeros((len(evaluator.momenta),2*width),dtype=complex)
        previous_end=0;previous_product=None
        for position,order in enumerate(orders):
            end=int(np.searchsorted(evaluator.degrees,order,side='right'))
            matrix=evaluator.coefficients[:,previous_end:end]
            if matrix.shape[1]:
                chiral+=np.einsum('pk,ks->ps',matrix,monomials[previous_end:end],optimize=True)
            product=chiral[:,::2]*chiral[:,1::2]
            # Keep the existing multiplication ordering for returned values.
            value=np.sum(weights*chiral[:,::2]*chiral[:,1::2],axis=0)
            if not np.isfinite(product).all() or not np.isfinite(value).all():
                raise FloatingPointError('nonfinite native finite-prefix contraction')
            values[start:stop,position]=value
            if previous_product is not None:
                increment=np.sum(absolute_weights*abs(product-previous_product),axis=0)
                if not np.isfinite(increment).all():
                    raise FloatingPointError('nonfinite absolute node increment')
                absolute[start:stop,position-1]=increment
            previous_product=product
            previous_end=end
    with np.errstate(over='raise',invalid='raise'):
        signed=np.diff(values,axis=1)
    if not np.isfinite(signed).all():
        raise FloatingPointError('nonfinite signed spectral increment')
    denominator=abs(values[:,-1]);defined=denominator>0
    with np.errstate(divide='ignore',invalid='ignore'):
        relative=absolute/denominator[:,None]
    return dict(version=VERSION,orders=orders,point_count=count,momentum_node_count=len(evaluator.momenta),
                values={order:values[:,i] for i,order in enumerate(orders)},
                signed_increments={pair:signed[:,i] for i,pair in enumerate(pairs)},
                absolute_weighted_increments={pair:absolute[:,i] for i,pair in enumerate(pairs)},
                relative_absolute_weighted_increments={pair:relative[:,i] for i,pair in enumerate(pairs)},
                normalization_denominator=denominator,normalization_defined=defined,
                zero_absolute_weight_node_counts=underflow_counts,
                tail_certified=False,momentum_rule_certified=False,all_points_retained=True)


__all__=['VERSION','evaluate_native_prefixes']
