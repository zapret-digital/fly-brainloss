"""Main run: random neuron removal on MaleCNS.
Modes: A = inputs and readouts protected ("brain"), B = only readouts protected ("brain and senses").
Fractions 0-95 %, 20 nested masks x 5 flies (v_th sd 0.5 mV), 3 Poisson trials x 0.3 s, dt 0.2 ms.
Controls: the same fractions of synapses removed at random; the degree-preserving shuffle (intact only,
see calibrate_malecns.py).
    python scripts/curves.py --graph malecns_ge5 --w 0.152 --modes A B syn [--masks 20 --flies 5 --sugar 80]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import analysis, experiments, runner  # noqa: E402
from brainloss.paths import RESULTS  # noqa: E402
from brainloss.skills import conditions  # noqa: E402

OUT = RESULTS / "malecns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="malecns_ge5")
    ap.add_argument("--w", type=float, default=0.152, help="synaptic weight unit in mV")
    ap.add_argument("--modes", nargs="+", default=["A", "B", "syn"])
    ap.add_argument("--masks", type=int, default=20)
    ap.add_argument("--flies", type=int, default=5)
    ap.add_argument("--sugar", type=float, default=40.0, help="sugar input rate, Hz")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    w_scale = 1 / 1.81 if abs(a.w - 0.152) < 1e-3 else a.w / 0.275  # 0.152 stands for 0.275 / 1.81
    conds = conditions(sugar_hz=a.sugar)
    flies = tuple(range(1, a.flies + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.graph}_w{a.w:.3f}_s{int(a.sugar)}"
    for mode in a.modes:
        path = OUT / f"curves_{tag}_{mode}.jsonl"
        if mode == "syn":
            tasks = experiments.synapse_tasks(a.graph, w_scale, conds, masks=a.masks, flies=flies)
        else:
            tasks = experiments.curve_tasks(a.graph, w_scale, mode, conds, masks=a.masks, flies=flies)
        runner.run_all(tasks, path, workers=a.workers, log_every=100)
        df = analysis.relative(analysis.table(path))
        df.to_csv(OUT / f"curves_{tag}_{mode}_runs.csv", index=False)
        cv = analysis.curves(df)
        cv.to_csv(OUT / f"curves_{tag}_{mode}.csv", index=False)
        pt = analysis.p50_table(cv)
        for s in analysis.SKILLS:
            lo, hi, never = analysis.p50_boot(df, s, n=1000)
            pt[f"{s}_ci"] = analysis.ci_text(lo, hi)
            pt[f"{s}_never_share"] = round(never, 3)
        pt.to_csv(OUT / f"p50_{tag}_{mode}.csv", index=False)
        pd.set_option("display.width", 250)
        print(pt.T.to_string())


if __name__ == "__main__":
    main()
