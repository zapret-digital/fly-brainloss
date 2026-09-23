"""Search for a small set of neurons whose removal lets the fly keep eating under looming (MaleCNS).
Stage 1: remove chunks of 35 neurons from two ranked lists of excitatory neurons (top by out-degree, as in the
FlyWire search that found PVLP069, and top by synapses received from LC4/LPLC2).
Stage 2: every neuron of each chunk that restores MN9, one at a time.
Condition: sugar 80 Hz + looming 100 Hz vs sugar 80 Hz alone; restored = MN9 >= 70 % of the no-looming level.
    python scripts/find_bottleneck.py [--graph malecns_all] [--top 1645] [--workers 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS, graph_path  # noqa: E402
from evlif import read_graph  # noqa: E402

OUT = RESULTS / "malecns"
CONDS = {"s80": [("sugar", 80.0)], "s80_l100": [("sugar", 80.0), ("loom", 100.0)]}
CHUNK = 35


def ranked(graph, top):
    n = sets.neurons("malecns")
    N, E, ip, post, w = read_graph(graph_path(graph), mmap=False)
    pool = runner.pool_of(graph, "A")
    exc = pool[n.sign.to_numpy()[pool] > 0]
    outdeg = np.diff(ip)
    by_deg = exc[np.argsort(-outdeg[exc], kind="stable")][:top]
    loom = sets.stimuli("malecns")["loom"]
    is_loom = np.zeros(N, bool)
    is_loom[loom] = True
    pre = np.repeat(np.arange(N, dtype=np.int32), np.diff(ip))
    m = is_loom[pre] & (w > 0)
    from_loom = np.bincount(post[m], weights=w[m], minlength=N)
    by_loom = exc[np.argsort(-from_loom[exc], kind="stable")][:350]
    return {"hubs": by_deg, "loom_targets": by_loom}, from_loom


def score(path):
    rows = []
    for r in runner.load(path):
        _, _, _, label, _, fly, _ = r["key"].split("|")
        c = r["conds"]
        s = (c["s80"]["hz"]["MN9_L"] + c["s80"]["hz"]["MN9_R"]) / 2
        sl = (c["s80_l100"]["hz"]["MN9_L"] + c["s80_l100"]["hz"]["MN9_R"]) / 2
        rows.append({"label": label, "fly": int(fly), "s80": s, "s80_l100": sl, "restored": sl >= 0.7 * s and s >= 2})
    df = pd.DataFrame(rows)
    return df.groupby("label").agg(s80=("s80", "median"), s80_l100=("s80_l100", "median"),
                                   restored_share=("restored", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_all")
    ap.add_argument("--top", type=int, default=1645)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    lists, from_loom = ranked(a.graph, a.top)
    n = sets.neurons("malecns")
    rem = {"intact": []}
    for name, lst in lists.items():
        for k in range(0, len(lst), CHUNK):
            rem[f"{name}_{k}-{k + CHUNK - 1}"] = lst[k:k + CHUNK]
    tasks = experiments.single_tasks(a.graph, 1 / 1.81, rem, CONDS, flies=(1, 2, 3), sd=0.5, T=500.0, trials=3,
                                     dt=0.2, tag="chunk")
    p1 = OUT / f"bottleneck_chunks_{a.graph}.jsonl"
    runner.run_all(tasks, p1, workers=a.workers, log_every=20)
    sc = score(p1).sort_values("s80_l100", ascending=False)
    sc.to_csv(OUT / f"bottleneck_chunks_{a.graph}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sc.head(15).to_string(index=False))
    base = sc[sc.label == "intact"].iloc[0]
    hits = sc[(sc.restored_share >= 2 / 3) & (sc.label != "intact")]
    print(f"intact: s80 {base.s80:.1f} Hz, s80+loom {base.s80_l100:.1f} Hz; chunks that restore feeding: {len(hits)}")
    singles = {}
    for lab in hits.label:
        name, rng = lab.rsplit("_", 1)
        k0 = int(rng.split("-")[0])
        for i in lists[name][k0:k0 + CHUNK]:
            singles[f"single_{int(i)}"] = [int(i)]
    if singles:
        tasks = experiments.single_tasks(a.graph, 1 / 1.81, singles, CONDS, flies=(1, 2, 3), sd=0.5, T=500.0,
                                         trials=3, dt=0.2, tag="single")
        p2 = OUT / f"bottleneck_singles_{a.graph}.jsonl"
        runner.run_all(tasks, p2, workers=a.workers, log_every=50)
        s2 = score(p2).sort_values("s80_l100", ascending=False)
        s2["neuron"] = [f"{n.type.iat[int(l.split('_')[1])]}_{n.side.iat[int(l.split('_')[1])]}" for l in s2.label]
        s2["from_loom_syn"] = [int(from_loom[int(l.split('_')[1])]) for l in s2.label]
        s2.to_csv(OUT / f"bottleneck_singles_{a.graph}.csv", index=False)
        print(s2.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
