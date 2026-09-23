import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BRAINLOSS_DATA", ROOT / "data"))
BUILD = Path(os.environ.get("BRAINLOSS_BUILD", ROOT / "build"))
RESULTS = ROOT / "results"
METHODS = ROOT / "methods"

sys.path.insert(0, str(ROOT / "engine"))

for d in (DATA, BUILD, RESULTS):
    d.mkdir(parents=True, exist_ok=True)


def graph_path(name):
    """name: malecns_ge5, malecns_all, flywire_ge5, ... (+ _shuf for the shuffled control)."""
    return BUILD / f"{name}.bin"


def neurons_path(connectome):
    return BUILD / f"{connectome}_neurons.parquet"
