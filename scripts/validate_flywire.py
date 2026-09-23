"""Engine check on FlyWire 783 (>=5 synapses, w = 0.275 mV, uniform threshold, dt 0.2 ms, 0.3 s, 3 trials,
40 nested masks per fraction). Compares with the earlier FlyWire numbers: eating fails at ~16 %, turning away at
~31 %, the escape readout keeps >= 0.8 of intact at 95 %, and PVLP069 gates feeding under looming.
    python scripts/validate_flywire.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import analysis, experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402

G, W = "flywire_ge5", 1.0
OUT = RESULTS / "flywire"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conds = {
        "eat": [("sugar", 40.0)],
        "eat_bitter": [("sugar", 40.0), ("bitter", 150.0)],
        "loom": [("loom", 150.0)],
        "loomL": [("loomL", 150.0)],
        "lc10L": [("lc10L", 150.0)],
        "head": [("head", 150.0)],
    }
    tasks = experiments.curve_tasks(G, W, "A", conds, masks=40, flies=(0,), sd=0.0, trials=3)
    runner.run_all(tasks, OUT / "curves_A.jsonl")
    df = analysis.relative(analysis.table(OUT / "curves_A.jsonl"))
    cv = analysis.curves(df)
    cv.to_csv(OUT / "curves_A.csv", index=False)
    pt = analysis.p50_table(cv)
    pt.to_csv(OUT / "p50_A.csv", index=False)
    print(pt.T.to_string())

    # PVLP069: the right cell alone and both cells
    pv = sets.by_type("flywire", "PVLP069")
    pr = sets.by_type("flywire", "PVLP069", "R")
    pl = sets.by_type("flywire", "PVLP069", "L")
    fear = {
        "s150": [("sugar", 150.0)],
        "s150_l100": [("sugar", 150.0), ("loom", 100.0)],
        "s150_l150": [("sugar", 150.0), ("loom", 150.0)],
        "s80_l100": [("sugar", 80.0), ("loom", 100.0)],
        "s40_l100": [("sugar", 40.0), ("loom", 100.0)],
        "loom": [("loom", 150.0)],
    }
    rem = {"intact": [], "PVLP069_R": pr, "PVLP069_L": pl, "PVLP069_both": pv}
    tasks = []
    for dt in (0.2, 0.1):
        tasks += experiments.single_tasks(G, W, rem, fear, flies=(0,), sd=0.0, T=500.0, trials=4, dt=dt, tag="pvlp")
    runner.run_all(tasks, OUT / "pvlp069.jsonl")
    rows = []
    for r in runner.load(OUT / "pvlp069.jsonl"):
        _, _, _, label, dt, fly, _ = r["key"].split("|")
        row = {"removed": label, "dt": float(dt)}
        for c, v in r["conds"].items():
            hz = v["hz"]
            row[c] = round((hz["DNp01_L"] + hz["DNp01_R"]) / 2 if c == "loom" else (hz["MN9_L"] + hz["MN9_R"]) / 2, 1)
        rows.append(row)
    import pandas as pd
    t = pd.DataFrame(rows).sort_values(["dt", "removed"])
    t.to_csv(OUT / "pvlp069.csv", index=False)
    print(t.to_string(index=False))
    json.dump({"pvlp069_idx": {"L": pl.tolist(), "R": pr.tolist()}}, open(OUT / "pvlp069_ids.json", "w"))


if __name__ == "__main__":
    main()
