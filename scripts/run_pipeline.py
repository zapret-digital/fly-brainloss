"""Runs the whole experiment step by step. Every step resumes where it stopped, so the script can be restarted.
    python scripts/run_pipeline.py            # everything
    python scripts/run_pipeline.py --from 5   # start from step 5
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [
    ("engine check on FlyWire", ["validate_flywire.py"]),
    ("main curves, MaleCNS full graph, mode A", ["curves.py", "--graph", "malecns_all", "--w", "0.152", "--sugar", "80",
                                                  "--modes", "A", "--masks", "40"]),
    ("search for a small set of neurons that lets the fly eat under looming", ["find_bottleneck.py"]),
    ("PVLP069 on MaleCNS", ["pvlp069.py"]),
    ("curves, MaleCNS >=5 synapses, mode A", ["curves.py", "--graph", "malecns_ge5", "--w", "0.152", "--sugar", "80",
                                              "--modes", "A", "--masks", "40"]),
    ("mode B and synapse control, full graph", ["curves.py", "--graph", "malecns_all", "--w", "0.152", "--sugar", "80",
                                                "--modes", "B", "syn"]),
    ("mode B and synapse control, >=5 synapses", ["curves.py", "--graph", "malecns_ge5", "--w", "0.152", "--sugar",
                                                  "80", "--modes", "B", "syn"]),
    ("check at w = 0.275 mV, >=5 synapses", ["curves.py", "--graph", "malecns_ge5", "--w", "0.275", "--sugar", "40",
                                             "--modes", "A", "--masks", "10", "--flies", "2"]),
    ("check at w = 0.275 mV, full graph", ["curves.py", "--graph", "malecns_all", "--w", "0.275", "--sugar", "40",
                                           "--modes", "A", "--masks", "10", "--flies", "2"]),
    ("intact MaleCNS calibration", ["calibrate_malecns.py"]),
    ("PVLP069 in the preprint setup on FlyWire", ["preprint_setup.py"]),
    ("paths from looming to MN9", ["paths.py", "--graph", "malecns_all"]),
    ("direct sensory input to each readout", ["direct_inputs.py", "--graph", "malecns_all"]),
    ("share of synapses lost with neurons, full graph", ["syn_share.py", "--graph", "malecns_all"]),
    ("share of synapses lost with neurons, >=5 synapses", ["syn_share.py", "--graph", "malecns_ge5"]),
    ("charts for methods/", ["methods_charts.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    a = ap.parse_args()
    for k, (name, cmd) in enumerate(STEPS, 1):
        if k < a.start:
            continue
        t0 = time.time()
        print(f"\n=== step {k}/{len(STEPS)}: {name}", flush=True)
        r = subprocess.run([PY, str(ROOT / "scripts" / cmd[0])] + cmd[1:], cwd=ROOT)
        print(f"=== step {k} finished with code {r.returncode} in {time.time() - t0:.0f} s", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nall steps done", flush=True)


if __name__ == "__main__":
    main()
