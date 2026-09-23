"""Step 0: download the data and build the engine graphs.
    python scripts/prepare.py            # MaleCNS only
    python scripts/prepare.py --flywire  # also FlyWire 783 (for the engine check)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss import download, graphs  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    fw = "--flywire" in sys.argv
    download.download(("malecns", "flywire") if fw else ("malecns",))
    graphs.build_malecns((5, 1))
    graphs.build_cloud()
    for g in ("malecns_ge5", "malecns_all"):
        graphs.build_shuffle(g)
    if fw:
        graphs.build_flywire(5)
        graphs.build_shuffle("flywire_ge5")
    print(f"done in {time.time() - t0:.0f} s")
