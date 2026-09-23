"""How looming reaches the feeding motor neuron MN9, and what changes without PVLP069.
1) Wiring: shortest paths by input share (cost = -log of the fraction of the target's synaptic input that comes
   from the source) from LC4/LPLC2 through PVLP069 to MN9, and through the inhibitors named by Chen & Xi
   (DNge031, CB0565 = MaleCNS GNG119).
2) Activity: sugar + looming with and without both PVLP069; neurons whose activity changes most.
    python scripts/paths.py [--graph malecns_all]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import sets  # noqa: E402
from brainloss.paths import RESULTS, graph_path  # noqa: E402
from evlif import Engine, read_graph, vth_for_fly  # noqa: E402

OUT = RESULTS / "malecns"


def names(n, idx):
    return [f"{n.type.iat[i] if isinstance(n.type.iat[i], str) else n.flywireType.iat[i]}_{n.side.iat[i]}"
            f"({str(n.nt.iat[i])[:4]})" for i in idx]


def path_to(pred, src_row, dst):
    p = [dst]
    while p[-1] != -9999 and pred[src_row, p[-1]] >= 0:
        p.append(int(pred[src_row, p[-1]]))
    return p[::-1]


def wiring(graph):
    n = sets.neurons("malecns")
    N, E, ip, post, w = read_graph(graph_path(graph), mmap=False)
    W = sp.csr_matrix((w.astype(np.float64), post, ip), shape=(N, N))
    inabs = np.asarray(abs(W).sum(0)).ravel()
    A = abs(W).tocoo()
    frac = A.data / np.maximum(inabs[A.col], 1)
    C = sp.csr_matrix((-np.log(np.clip(frac, 1e-9, 1)) + 1e-6, (A.row, A.col)), shape=(N, N))
    ro = sets.readouts("malecns")
    pv = list(sets.by_type("malecns", "PVLP069"))
    inh = list(sets.by_type("malecns", "DNge031")) + list(sets.by_type("malecns", "GNG119"))
    loom = sets.stimuli("malecns")["loom"]
    dst = ro["MN9_L"]
    rows = []
    src = pv + inh
    dist, pred = dijkstra(C, directed=True, indices=src, return_predecessors=True)
    for r, s in enumerate(src):
        p = path_to(pred, r, dst)
        sg = [float(np.sign(W[p[k], p[k + 1]])) for k in range(len(p) - 1)]
        rows.append({"from": names(n, [s])[0], "to": "MN9_L", "cost": round(float(dist[r, dst]), 2),
                     "hops": len(p) - 1, "path": " > ".join(names(n, p)), "signs": "".join("+" if x > 0 else "-" for x in sg),
                     "net_sign": "+" if np.prod(sg) > 0 else "-"})
    # PVLP069 -> the preprint inhibitors
    for r, s in enumerate(pv):
        for t in inh:
            p = path_to(pred, r, t)
            sg = [float(np.sign(W[p[k], p[k + 1]])) for k in range(len(p) - 1)]
            rows.append({"from": names(n, [s])[0], "to": names(n, [t])[0], "cost": round(float(dist[r, t]), 2),
                         "hops": len(p) - 1, "path": " > ".join(names(n, p)),
                         "signs": "".join("+" if x > 0 else "-" for x in sg), "net_sign": "+" if np.prod(sg) > 0 else "-"})
    # share of the looming input to PVLP069 and to the inhibitors
    extra = {}
    for t in pv + inh:
        col = W[:, t].toarray().ravel()
        exc = col[col > 0].sum()
        extra[names(n, [t])[0]] = {"exc_input_syn": int(exc), "from_LC4_LPLC2": round(float(col[loom][col[loom] > 0].sum() / max(exc, 1)), 3)}
    # strongest direct inputs of MN9_L
    col = W[:, dst].toarray().ravel()
    top = np.argsort(-np.abs(col))[:15]
    mn9_in = [{"pre": names(n, [i])[0], "syn": int(col[i])} for i in top if col[i] != 0]
    return pd.DataFrame(rows), extra, mn9_in


def activity(graph, sugar=80.0, loom=100.0, flies=(1, 2, 3), trials=5):
    n = sets.neurons("malecns")
    S = sets.stimuli("malecns")
    pv = sets.by_type("malecns", "PVLP069")
    eng = Engine(graph_path(graph), 1 / 1.81)
    G = [(S["sugar"], sugar), (S["loom"], loom)]
    res = {}
    for label, rem in (("intact", None), ("no_PVLP069", pv)):
        tot = np.zeros(eng.N)
        for fly in flies:
            eng.set_removed(rem)
            eng.set_vth(vth_for_fly(eng.N, fly, 0.5))
            for tr in range(trials):
                tot += eng.run(G, dt=0.2, max_ms=500.0, seed=fly * 100 + tr, full=True)["spike_count"]
        res[label] = tot / (len(flies) * trials * 0.5)
    d = res["no_PVLP069"] - res["intact"]
    rows = []
    chain = []
    for t in ("CL213", "GNG119", "DNge031", "DNg74_a", "GNG108", "AVLP535", "MN9"):
        chain += sets.by_type("malecns", t).tolist()
    for i in chain + np.argsort(d)[:25].tolist() + np.argsort(-d)[:15].tolist():
        rows.append({"neuron": names(n, [i])[0], "intact_hz": round(res["intact"][i], 1),
                     "without_PVLP069_hz": round(res["no_PVLP069"][i], 1), "change_hz": round(d[i], 1),
                     "sign": int(n.sign.iat[i])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wt, extra, mn9_in = wiring(a.graph)
    wt.to_csv(OUT / f"paths_{a.graph}.csv", index=False)
    json.dump({"input_shares": extra, "MN9_L_top_inputs": mn9_in}, open(OUT / f"paths_{a.graph}_inputs.json", "w"),
              ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(wt.to_string(index=False))
    print(json.dumps(extra, ensure_ascii=False))
    print(mn9_in)
    act = activity(a.graph)
    act.insert(0, "group", ["chain"] * (len(act) - 40) + ["drop"] * 25 + ["rise"] * 15)
    act.to_csv(OUT / f"paths_{a.graph}_activity.csv", index=False)
    print(act.to_string(index=False))


if __name__ == "__main__":
    main()
