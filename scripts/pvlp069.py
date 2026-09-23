"""Does looming stop feeding through PVLP069? Single removals of PVLP069 (left, right, both) on MaleCNS.
Robustness: both graphs, w 0.152 and 0.275 mV, flies with v_th sd 0.5 and 1.0 mV, dt 0.1 and 0.2 ms, 5 trials x 0.5 s.
Criterion: removing <= 3 neurons brings MN9 under looming back to >= 70 % of the no-looming level in >= 80 % of flies.
    python scripts/pvlp069.py [--quick]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402

OUT = RESULTS / "malecns"

CONDS = {
    "s150": [("sugar", 150.0)], "s80": [("sugar", 80.0)], "s40": [("sugar", 40.0)],
    "s150_l100": [("sugar", 150.0), ("loom", 100.0)],
    "s150_l150": [("sugar", 150.0), ("loom", 150.0)],
    "s80_l100": [("sugar", 80.0), ("loom", 100.0)],
    "s40_l100": [("sugar", 40.0), ("loom", 100.0)],
    "s150_lL150": [("sugar", 150.0), ("loomL", 150.0)],
    "s150_lR150": [("sugar", 150.0), ("loomR", 150.0)],
    "loom": [("loom", 150.0)],
}
PAIRS = {"s150_l100": "s150", "s150_l150": "s150", "s80_l100": "s80", "s40_l100": "s40",
         "s150_lL150": "s150", "s150_lR150": "s150"}


def removals(conn="malecns"):
    L = sets.by_type(conn, "PVLP069", "L")
    R = sets.by_type(conn, "PVLP069", "R")
    return {"intact": [], "PVLP069_L": L, "PVLP069_R": R, "PVLP069_both": np.concatenate([L, R])}


def tasks_for(graph, w, dts=(0.2, 0.1), sds=(0.5, 1.0), flies=range(1, 11), conds=CONDS, rem=None):
    rem = rem or removals()
    t = []
    for dt in dts:
        for sd in sds:
            t += experiments.single_tasks(graph, w, rem, conds, flies=tuple(flies), sd=sd, T=500.0, trials=5, dt=dt,
                                          tag="pvlp")
    return t


def table(path):
    rows = []
    for r in runner.load(path):
        g, w, _, label, dt, fly, sd = r["key"].split("|")
        row = {"graph": g, "w": round(float(w) * 0.275, 3), "removed": label, "dt": float(dt), "fly": int(fly),
               "sd": float(sd)}
        for c, v in r["conds"].items():
            hz = v["hz"]
            row[c] = (hz["DNp01_L"] + hz["DNp01_R"]) / 2 if c == "loom" else (hz["MN9_L"] + hz["MN9_R"]) / 2
            row[f"{c}_MN9_L"] = hz["MN9_L"]
            row[f"{c}_active"] = v["active"]
        rows.append(row)
    return pd.DataFrame(rows)


def verdicts(df):
    """Share of flies where MN9 under looming is >= 70 % of the same fly's no-looming level."""
    out = []
    for key, g in df.groupby(["graph", "w", "dt", "sd", "removed"]):
        row = dict(zip(["graph", "w", "dt", "sd", "removed"], key))
        row["flies"] = len(g)
        for c, base in PAIRS.items():
            ok = (g[c] >= 0.7 * g[base]) & (g[base] >= 5)
            row[f"{c}_ok"] = round(float(ok.mean()), 2)
            row[f"{c}_med"] = round(float(g[c].median()), 1)
        for b in ("s150", "s80", "s40", "loom"):
            row[f"{b}_med"] = round(float(g[b].median()), 1)
        out.append(row)
    return pd.DataFrame(out)


def main():
    quick = "--quick" in sys.argv
    tasks = []
    for g in ("malecns_all", "malecns_ge5"):
        for w in (1 / 1.81, 1.0):
            if quick:
                tasks += tasks_for(g, w, dts=(0.2,), sds=(0.5,), flies=range(1, 4))
            elif w == 1.0:
                tasks += tasks_for(g, w, dts=(0.2,), sds=(0.5,), flies=range(1, 6))
            else:
                tasks += tasks_for(g, w)
    runner.run_all(tasks, OUT / "pvlp069.jsonl", log_every=20)
    df = table(OUT / "pvlp069.jsonl")
    df.to_csv(OUT / "pvlp069_raw.csv", index=False)
    v = verdicts(df)
    v.to_csv(OUT / "pvlp069_verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
