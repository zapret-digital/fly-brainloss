"""For every skill: which share of the readout's excitatory synaptic input comes straight from the stimulated
sensory neurons, and the fewest excitatory links from those neurons to the readout.
    python scripts/direct_inputs.py [--graph malecns_all]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import sets  # noqa: E402
from brainloss.paths import RESULTS, graph_path  # noqa: E402
from evlif import read_graph  # noqa: E402

SKILLS = {
    "eat": ("sugar", ["MN9_L", "MN9_R"]),
    "escape": ("loom", ["DNp01_L", "DNp01_R"]),
    "turn_away": ("loomL", ["DNa01_R", "DNa02_R"]),
    "turn_to": ("lc10L", ["DNa02_L"]),
    "head": ("head", [f"{t}_{s}" for t in sets.HEAD_DN for s in "LR"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_all")
    a = ap.parse_args()
    N, E, ip, post, w = read_graph(graph_path(a.graph), mmap=False)
    W = sp.csr_matrix((w.astype(np.float64), post, ip), shape=(N, N))
    WT = W.T.tocsr()
    Wexc = sp.csr_matrix(((w > 0).astype(np.float64), post, ip), shape=(N, N))
    S, ro = sets.stimuli("malecns"), sets.readouts("malecns")
    out = {}
    for skill, (stim, rs) in SKILLS.items():
        src = S[stim]
        dist = shortest_path(Wexc, directed=True, unweighted=True, indices=src)
        rows = {}
        for r in rs:
            col = WT[ro[r]].toarray().ravel()
            exc = col[col > 0].sum()
            direct = col[src][col[src] > 0].sum()
            d = dist[:, ro[r]]
            rows[r] = {"direct_share_of_excitatory_input": round(float(direct / max(exc, 1)), 3),
                       "fewest_excitatory_links": int(np.min(d)) if np.isfinite(d).any() else None}
        out[skill] = {"stimulus": stim, "readouts": rows}
        print(skill, rows, flush=True)
    path = RESULTS / "malecns" / f"direct_inputs_{a.graph}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
