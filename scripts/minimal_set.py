"""How few neurons have to go for looming to stop blocking feeding? Starts from the chunk found by
find_bottleneck.py (the 35 strongest recipients of LC4/LPLC2 input) and halves it while a half still works.
Restored = sugar 80 Hz + looming 100 Hz gives MN9 >= 70 % of sugar alone, in at least 2 of 3 flies.
The criterion is loose: a half can pass it with 2 of 3 flies while its median MN9 comes back only part of the way,
so read the medians in "steps" (s80 is sugar alone, s80_l100 is sugar under looming).
    python scripts/minimal_set.py [--graph malecns_all] [--workers 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402
from find_bottleneck import CONDS, ranked, score  # noqa: E402

OUT = RESULTS / "malecns"
NOTE = ("passed = MN9 under looming >= 70 % of sugar alone in at least 2 of 3 flies; "
        "compare s80_l100 with s80 in steps to see how far feeding comes back")


def test(graph, groups, path, workers):
    tasks = experiments.single_tasks(graph, 1 / 1.81, groups, CONDS, flies=(1, 2, 3), sd=0.5, T=500.0, trials=3,
                                     dt=0.2, tag="minset")
    runner.run_all(tasks, path, workers=workers, log_every=100)
    sc = score(path).set_index("label")
    return {k: sc.loc[k] for k in groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    n = sets.neurons("malecns")
    lists, from_loom = ranked(a.graph, 1645)
    cur = [int(i) for i in lists["loom_targets"][:35]]
    path = OUT / f"minimal_set_{a.graph}.jsonl"
    log = []
    while len(cur) > 1:
        h = len(cur) // 2
        halves = {f"set_{'-'.join(map(str, sorted(cur[:h])))}": cur[:h], f"set_{'-'.join(map(str, sorted(cur[h:])))}": cur[h:]}
        res = test(a.graph, halves, path, a.workers)
        step = {"size": len(cur), "halves": []}
        ok = None
        for k, idx in halves.items():
            r = res[k]
            step["halves"].append({"n": len(idx), "types": [f"{n.type.iat[i]}_{n.side.iat[i]}" for i in idx],
                                   "s80": round(float(r.s80), 1), "s80_l100": round(float(r.s80_l100), 1),
                                   "restored_share": round(float(r.restored_share), 2)})
            if r.restored_share >= 2 / 3 and (ok is None or r.s80_l100 > res[ok].s80_l100):
                ok = k
        log.append(step)
        print(json.dumps(step, ensure_ascii=False), flush=True)
        if ok is None:
            break
        cur = halves[ok]
    result = {"last_passing_set": [f"{n.type.iat[i]}_{n.side.iat[i]}" for i in cur], "size": len(cur),
              "idx": cur, "note": NOTE, "steps": log}
    json.dump(result, open(OUT / f"minimal_set_{a.graph}.json", "w"), ensure_ascii=False, indent=1)
    print("last set that passed the criterion:", result["size"], result["last_passing_set"])


if __name__ == "__main__":
    main()
