"""Share of synapses that disappear together with the removed neurons, for every nested mask and fraction.
Used to put neuron removal and the synapse-removal control on one axis.
    python scripts/syn_share.py --graph malecns_all --modes A B --masks 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import runner  # noqa: E402
from brainloss.experiments import FRACS  # noqa: E402
from brainloss.paths import RESULTS, graph_path  # noqa: E402
from evlif import read_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_all")
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--masks", type=int, default=40)
    a = ap.parse_args()
    N, E, ip, post, w = read_graph(graph_path(a.graph), mmap=False)
    pre = np.repeat(np.arange(N, dtype=np.int32), np.diff(ip))
    absw = np.abs(w).astype(np.float64)
    total = absw.sum()
    rows = []
    for mode in a.modes:
        for m in range(a.masks):
            order = runner.order_of(a.graph, mode, m)
            rank = np.full(N, np.iinfo(np.int32).max, np.int32)
            rank[order] = np.arange(len(order), dtype=np.int32)
            first = np.minimum(rank[pre], rank[post])  # an edge is lost once either end is removed
            o = np.argsort(first, kind="stable")
            cum = np.cumsum(absw[o])
            fs = first[o]
            for p in FRACS:
                cut = int(round(p * len(order)))
                k = np.searchsorted(fs, cut)
                rows.append({"graph": a.graph, "mode": mode, "frac": p, "mask": m,
                             "syn_lost": float(cum[k - 1] / total) if k else 0.0})
        print(mode, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "malecns" / f"syn_share_{a.graph}.csv", index=False)
    print(df.groupby(["mode", "frac"]).syn_lost.median().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
