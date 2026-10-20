"""Verified saved Liouville inputs and fixed thermal sampling conventions."""
from collections import OrderedDict
import math
from genus1_source_paths import ROOT
from genus1_boundary_spectrum import read
from genus1_boundary_banks import load_native, EllipticGridBank, sha
from genus1_collision_elliptic import CollisionBank
from genus1_three_point_collision import reconstruct_points
from kernels import compact_factor

OUT = ROOT/'output/thermal_one_loop_20260927'
SOURCE = ROOT/'output/diagnostics/genus1_curve10_primary_384_20260923/evidence_snapshot'
OUTPUT = ROOT/'output/thermal_refinement_20260927'
SECTORS = ((0,1,2),(0,2,1),(2,0,1))

class Banks:
    def __init__(self, tag):
        self.root = OUT/"inputs"/tag/"grids"
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        folder = self.root/key
        spec, pin = read(folder/"spec.json"), read(folder/"ready.json")
        assert sha(folder/"spec.json") == pin["spec_sha256"]
        # Every retrieved artifact is checked against its original final pin.
        for name in pin["files"]:
            path = folder/name
            if path.exists():
                assert sha(path) == pin["files"][name], name
        bank = load_native(folder/"input/exports/bank.npz")
        assert bank.metadata["identity"] == spec["grid"]["identity"]
        assert bank.metadata["quadrature"]["rules"] == spec["grid"]["rules"]
        ellipse = None
        if spec["grid"]["elliptic_order"] is not None:
            ellipse = EllipticGridBank.load(folder/"input/elliptic/bank.npz",
                receipt_sha256=pin["files"]["input/elliptic/bank.json"])
        elif spec["grid"].get("collision_elliptic") is not None:
            ellipse = CollisionBank.load(folder/"input/collision/bank.npz", bank,
                spec["case"]["kind"], spec["grid"]["collision_elliptic"],
                receipt_sha256=pin["files"]["input/collision/bank.json"])
        result = spec, pin, bank, ellipse
        self.cache[key] = result
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result


def pair_factor(R, tau, pair, delta):
    w1, w2 = reconstruct_points(tuple(pair), 0j, complex(delta))
    return compact_factor(R, tau, (w1, w2, 0j), [1, 1, -2])
