"""Prepare and preserve two-point Liouville coefficients for thermal reuse."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from genus1_two_point_worldsheet import LiouvilleTorusTwoPoint,MomentumRule


NATIVE_BANK_FORMAT = "torus-two-point-native-q-c-recursion-v1"
NATIVE_BANK_METHOD = dict(
    format=NATIVE_BANK_FORMAT,
    central_charge=25.,
    necklace_backend="c-recursion",
    ope_backend="c-recursion",
    coefficient_variables={"necklace": ["q1", "q2"], "ope": ["q", "delta"]},
)


def bank_stem(x, n, necklace, ope):
    # A distinct filename protects the historical transformed banks.
    return (f"native_q_v1_x{x:.17g}_n{n}_neck{necklace[0]}-{necklace[1]}"
            f"_ope{ope[0]}-{ope[1]}")


def validate_metadata(metadata):
    if any(metadata.get(k) != v for k, v in NATIVE_BANK_METHOD.items()):
        raise ValueError("A native q1,q2 c-recursion bank is required; regenerate legacy coefficients.")


def load_bank(path, expected=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    validate_metadata(metadata)
    if expected and any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Existing coefficient bank has different settings: " + str(path))
    with np.load(path, allow_pickle=False) as saved:
        if "format" not in saved or saved["format"].item() != NATIVE_BANK_FORMAT:
            raise ValueError("The coefficient array has no native-q format marker: " + str(path))
        arrays = {k: saved[k] for k in saved.files if k != "format"}
    count = metadata["momentum_order"] ** 2
    for label, orders in (("neck", metadata["necklace_orders"]), ("ope", metadata["ope_orders"])):
        for key in ("h1", "h2", "weights", "coefficients"):
            array = arrays[label + "_" + key]
            shape = (count, *(n + 1 for n in orders)) if key == "coefficients" else (count,)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError("Invalid coefficient bank array: " + label + "_" + key)
    return metadata, arrays


def prepare(x,n,necklace,ope,output_dir):
    if not 0 < x < 1:
        raise ValueError("The real-contour thermal calculation requires 0 < x < 1.")
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if int(n) != n or n < 1 or any(int(k) != k or k < 0 for k in (*necklace, *ope)):
        raise ValueError("Positive momentum order and nonnegative integer block orders required.")
    stem=bank_stem(x,n,necklace,ope)
    bank=out/(stem+".npz")
    if bank.exists():
        expected=dict(x=x,momentum_order=n,p_max=6.,power=2.,
                      necklace_orders=list(necklace),ope_orders=list(ope))
        load_bank(bank, expected)
        print("Existing coefficient bank:",stem,flush=True)
        return bank
    start=time.monotonic()
    c=LiouvilleTorusTwoPoint(1j*x,
        momentum_rule=MomentumRule.power_legendre(6.,n,2.),
        necklace_orders=necklace,ope_orders=ope,
        necklace_backend="c-recursion",ope_backend="c-recursion",special_dps=28)
    c.correlator_necklace(1.+1j,1.2j)
    print("Necklace coefficients ready",time.monotonic()-start,flush=True)
    c.correlator_ope(.2+.1j,1.2j)
    a={}
    for label,arrays in (("neck",c._necklace_arrays),("ope",c._ope_arrays)):
        for key,value in zip(("h1","h2","weights","coefficients"),arrays):
            a[label+"_"+key]=value
    np.savez_compressed(bank,format=np.asarray(NATIVE_BANK_FORMAT),**a)
    metadata=dict(NATIVE_BANK_METHOD,x=x,momentum_order=n,p_max=6.,power=2.,necklace_orders=necklace,
                  ope_orders=ope,worldsheet_only=True,seconds=time.monotonic()-start,
                  source="code/one_loop/two_point/liouville_integration/genus1_two_point_worldsheet.py")
    (out/(stem+".json")).write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)
    return bank


def main():
    p=argparse.ArgumentParser();p.add_argument("--x",type=float,required=True)
    p.add_argument("--nodes",type=int,default=16)
    p.add_argument("--necklace",type=int,nargs=2,default=(4,4))
    p.add_argument("--ope",type=int,nargs=2,default=(2,6))
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();prepare(a.x,a.nodes,tuple(a.necklace),tuple(a.ope),a.output_dir)


if __name__ == "__main__":
    main()
