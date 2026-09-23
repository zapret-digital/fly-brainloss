"""Intact MaleCNS: which skills exist at all, at which input rates, and how the controls respond.
Graphs >=5 synapses and all synapses, w = 0.275 mV (FlyWire value) and 0.275/1.81 = 0.152 mV (the male brain has
about 1.81 times more synapses per neuron). 6 flies (uniform threshold + 5 with v_th sd 0.5 mV), 3 trials x 0.3 s.
Also the degree-preserving shuffled graphs.
    python scripts/calibrate_malecns.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402
from brainloss.skills import skill_values  # noqa: E402

OUT = RESULTS / "malecns"
W = {"0.152": 1 / 1.81, "0.275": 1.0}
SUGAR = [10, 20, 40, 60, 80, 100, 150]


def conds():
    S = sets.stimuli("malecns")
    c = {f"sugar{hz}": [("sugar", float(hz))] for hz in SUGAR}
    for k in range(3):
        rnd = sets.random_sensory("malecns", len(S["sugar"]), seed=11 + k).tolist()
        for hz in (40, 80, 150):
            c[f"rand{k}_{hz}"] = [(rnd, float(hz))]
    c["bitter150"] = [("bitter", 150.0)]
    for hz in (40, 80, 100):
        c[f"sugar{hz}_bitter"] = [("sugar", float(hz)), ("bitter", 150.0)]
    for k in ("loom", "loomL", "loomR", "lc10L", "lc10R", "head"):
        c[k] = [(k, 150.0)]
    return c


def summary(path):
    rows = []
    for r in runner.load(path):
        g, w, _, _, _, fly, _ = r["key"].split("|")
        for cn, v in r["conds"].items():
            hz = v["hz"]
            rows.append({"graph": g, "w": round(float(w) * 0.275, 3), "fly": int(fly), "cond": cn, "active": v["active"],
                         "MN9_L": hz["MN9_L"], "MN9_R": hz["MN9_R"],
                         "DNp01": (hz["DNp01_L"] + hz["DNp01_R"]) / 2,
                         "DNa_L": (hz["DNa01_L"] + hz["DNa02_L"]) / 2, "DNa_R": (hz["DNa01_R"] + hz["DNa02_R"]) / 2,
                         "DNa02_L": hz["DNa02_L"], "DNa02_R": hz["DNa02_R"],
                         "headDN": np.mean([hz[f"{t}_{s}"] for t in sets.HEAD_DN for s in "LR"]),
                         "wall_ms": v["wall_ms"]})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = conds()
    tasks = []
    for g in ("malecns_ge5", "malecns_all", "malecns_ge5_shuf", "malecns_all_shuf"):
        for wname, w in W.items():
            tasks += experiments.single_tasks(g, w, {"intact": []}, c, flies=(0, 1, 2, 3, 4, 5), sd=0.5,
                                              T=300.0, trials=3, dt=0.2, tag="calib")
    runner.run_all(tasks, OUT / "calibration.jsonl", log_every=8)
    df = summary(OUT / "calibration.jsonl")
    df.to_csv(OUT / "calibration_raw.csv", index=False)
    med = df.groupby(["graph", "w", "cond"]).median(numeric_only=True).drop(columns="fly").round(1)
    med.to_csv(OUT / "calibration.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(med.to_string())


if __name__ == "__main__":
    main()
