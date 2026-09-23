"""PVLP069 on FlyWire in the stimulus setup of Chen & Xi (bioRxiv 10.64898/2025.12.14.694122, code
github.com/WJXI/escape-of-Drosophila): 20 sugar GRNs, looming on the right side only, MN9 read out, 1 s x 3 trials.
    python scripts/preprint_setup.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import experiments, runner, sets  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402

# NEU_SUGAR from pdt.py of the preprint code
SUGAR20 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
           720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
           720575940639332736, 720575940616885538, 720575940639198653, 720575940617937543, 720575940632425919,
           720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570]


def main():
    n = sets.neurons("flywire")
    idx = pd.Index(n.id)
    s20 = idx.get_indexer(SUGAR20)
    assert (s20 >= 0).all()
    print("sugar20 subclasses:", n.subclass.iloc[s20].value_counts().to_dict(), "sides:", n.side.iloc[s20].value_counts().to_dict())
    S = sets.stimuli("flywire")
    s20 = s20.astype(np.int32).tolist()
    conds = {"s150": [(s20, 150.0)], "s150_loomR150": [(s20, 150.0), ("loomR", 150.0)],
             "s150_loomR100": [(s20, 150.0), ("loomR", 100.0)], "loomR": [("loomR", 150.0)]}
    rem = {"intact": [], "PVLP069_R": sets.by_type("flywire", "PVLP069", "R"),
           "PVLP069_L": sets.by_type("flywire", "PVLP069", "L"), "PVLP069_both": sets.by_type("flywire", "PVLP069")}
    tasks = experiments.single_tasks("flywire_ge5", 1.0, rem, conds, flies=(0, 1, 2, 3, 4, 5), sd=0.5, T=1000.0,
                                     trials=3, dt=0.2, tag="preprint")
    out = RESULTS / "flywire" / "preprint_setup.jsonl"
    runner.run_all(tasks, out)
    rows = []
    for r in runner.load(out):
        _, _, _, label, _, fly, _ = r["key"].split("|")
        row = {"removed": label, "fly": int(fly)}
        for c, v in r["conds"].items():
            row[c] = v["hz"]["DNp01_R"] if c == "loomR" else v["hz"]["MN9_R"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "flywire" / "preprint_setup.csv", index=False)
    print("MN9_R (Hz), median over 6 flies; loomR column = DNp01_R")
    print(df.groupby("removed").median(numeric_only=True).drop(columns="fly").round(1).to_string())


if __name__ == "__main__":
    main()
