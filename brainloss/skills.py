"""Skills: a stimulus condition and the readout neurons that show the behaviour."""
import numpy as np

from .sets import HEAD_DN, READ_TYPES

READOUT_NAMES = [f"{t}_{s}" for t in READ_TYPES for s in ("L", "R")]

# plain words for the pictures, neuron names only in the tables
WORDS = {
    "eat": "ест",
    "bitter": "бросает еду, если горько",
    "escape": "уворачивается от тени",
    "turn_away": "поворачивает от угрозы",
    "turn_to": "тянется к движущейся точке",
    "head": "реагирует на касание головы",
}


def conditions(sugar_hz=40.0, stim_hz=150.0, bitter_hz=150.0):
    """Condition name -> list of (stimulus set, rate in Hz)."""
    return {
        "eat": [("sugar", sugar_hz)],
        "eat_bitter": [("sugar", sugar_hz), ("bitter", bitter_hz)],
        "loom": [("loom", stim_hz)],
        "loomL": [("loomL", stim_hz)],
        "lc10L": [("lc10L", stim_hz)],
        "head": [("head", stim_hz)],
    }


def _mean(hz, names):
    return float(np.mean([hz[n] for n in names]))


def skill_values(res):
    """res: condition -> {'hz': {readout: Hz}, ...}. Returns skill -> raw output (Hz, or veto share)."""
    out = {}
    if "eat" in res:
        out["eat"] = _mean(res["eat"]["hz"], ["MN9_L", "MN9_R"])
    if "eat_bitter" in res and "eat" in res:
        base = out["eat"]
        left = _mean(res["eat_bitter"]["hz"], ["MN9_L", "MN9_R"])
        out["bitter"] = 1.0 - left / base if base >= 2.0 else np.nan  # veto is measurable only while eating works
        out["bitter_left_hz"] = left
    if "loom" in res:
        out["escape"] = _mean(res["loom"]["hz"], ["DNp01_L", "DNp01_R"])
        lat = [x for x in (res["loom"]["first_ms"]["DNp01_L"], res["loom"]["first_ms"]["DNp01_R"]) if x >= 0]
        out["escape_latency_ms"] = float(np.mean(lat)) if lat else np.nan
    if "loomL" in res:
        out["turn_away"] = _mean(res["loomL"]["hz"], ["DNa01_R", "DNa02_R"])
        out["turn_wrong_side"] = _mean(res["loomL"]["hz"], ["DNa01_L", "DNa02_L"])
    if "lc10L" in res:
        out["turn_to"] = res["lc10L"]["hz"]["DNa02_L"]
    if "head" in res:
        out["head"] = _mean(res["head"]["hz"], [f"{t}_{s}" for t in HEAD_DN for s in ("L", "R")])
    return out
