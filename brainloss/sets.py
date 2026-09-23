"""Stimulus and readout neurons, matched by the cell type annotation of each connectome."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import neurons_path

READ_TYPES = ["MN9", "DNp01", "DNa01", "DNa02", "DNg15", "DNg35", "DNg84", "DNg85", "DNg48"]
HEAD_DN = ["DNg15", "DNg35", "DNg84", "DNg85", "DNg48"]


@lru_cache(None)
def neurons(connectome):
    return pd.read_parquet(neurons_path(connectome))


def _idx(mask):
    return np.flatnonzero(np.asarray(mask, bool)).astype(np.int32)


@lru_cache(None)
def stimuli(connectome):
    n = neurons(connectome)
    t = n["type"].fillna("")
    side = n["side"].fillna("")
    loom = t.isin(["LC4", "LPLC2"])
    if connectome == "flywire":
        sc = n["subclass"].fillna("")
        sugar, bitter, head = sc == "sugar/water", sc == "bitter", sc == "head bristle"
    else:
        sugar = t.isin(["LB3", "LB3a", "LB3b", "LB3c", "LB3d", "LB2d"])
        bitter = t.isin(["LB1a", "LB1b", "LB1c", "LB1d"])
        # head bristle mechanosensory neurons without the eye bristles (BM_InOm); part of them is typed only
        # as "BM" or only by flywireType (BM_dOcci, BM_dPoOr), as in the FlyWire "head bristle" subclass
        ft = n["flywireType"].fillna("")
        head = ((t.str.startswith("BM") | ((t == "") & ft.str.startswith("BM_")))
                & (t != "BM_InOm") & ~ft.str.contains("InOm"))
    return {
        "sugar": _idx(sugar), "bitter": _idx(bitter), "loom": _idx(loom),
        "loomL": _idx(loom & (side == "L")), "loomR": _idx(loom & (side == "R")),
        "lc10L": _idx((t == "LC10a") & (side == "L")), "lc10R": _idx((t == "LC10a") & (side == "R")),
        "head": _idx(head),
    }


@lru_cache(None)
def readouts(connectome):
    """name -> neuron index, e.g. MN9_L, DNp01_R. In FlyWire MN9 is annotated as CB0701."""
    n = neurons(connectome)
    t = n["type"].fillna("")
    if connectome == "flywire":
        t = t.replace({"CB0701": "MN9"})
    out = {}
    for typ in READ_TYPES:
        for s in ("L", "R"):
            rows = np.flatnonzero(((t == typ) & (n["side"] == s)).to_numpy())
            assert len(rows) == 1, (connectome, typ, s, len(rows))
            out[f"{typ}_{s}"] = int(rows[0])
    return out


def by_type(connectome, typ, side=None):
    n = neurons(connectome)
    m = n["type"].fillna("") == typ
    if side:
        m &= n["side"] == side
    return _idx(m)


def random_sensory(connectome, size, seed):
    """Control input: random sensory neurons, as many as in the real stimulus."""
    n = neurons(connectome)
    sc = n["superclass"].fillna("")
    pool = _idx(sc.str.contains("sensory"))
    return np.sort(np.random.default_rng(seed).choice(pool, size, replace=False)).astype(np.int32)
