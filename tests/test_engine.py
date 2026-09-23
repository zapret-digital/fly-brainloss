import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainloss.paths import graph_path  # noqa: E402
from evlif import Engine, write_graph  # noqa: E402


def chain(tmp_path, weights=(30.0, 200.0)):
    """0 -> 1 -> 2: 30 synapses, then 200 (one spike of 1 is enough for 2)."""
    p = tmp_path / "chain.bin"
    write_graph(p, 3, [0, 1, 2, 2], [1, 2], list(weights))
    return Engine(p, 1.0)


def test_rest_is_silent(tmp_path):
    e = chain(tmp_path)
    r = e.run([], max_ms=200, readouts=[0, 1, 2])
    assert r["total_spikes"] == 0 and r["readout_count"].sum() == 0


def test_chain_propagates(tmp_path):
    e = chain(tmp_path)
    r = e.run([([0], 150.0)], max_ms=1000, seed=1, readouts=[0, 1, 2])
    c = r["readout_count"]
    assert c[0] > 100 and c[1] > 0 and c[2] > 0


def test_removed_neuron_is_gone(tmp_path):
    e = chain(tmp_path)
    e.set_removed([1])
    r = e.run([([0], 150.0)], max_ms=1000, seed=1, readouts=[0, 1, 2], full=True)
    assert r["readout_count"][0] > 100
    assert r["readout_count"][1] == 0 and r["readout_count"][2] == 0
    assert r["spike_count"][1] == 0
    e.set_removed([0])  # a removed stimulated neuron does not fire either
    r = e.run([([0], 150.0)], max_ms=1000, seed=1, readouts=[0, 1, 2])
    assert r["readout_count"].sum() == 0
    e.set_removed(None)
    r = e.run([([0], 150.0)], max_ms=1000, seed=1, readouts=[0, 1, 2])
    assert r["readout_count"][2] > 0


def test_weights_override_and_restore(tmp_path):
    e = chain(tmp_path)
    base = e.run([([0], 150.0)], max_ms=1000, seed=3, readouts=[1, 2])["readout_count"].copy()
    e.set_weights(np.array([30.0, 0.0], np.float32))
    r = e.run([([0], 150.0)], max_ms=1000, seed=3, readouts=[1, 2])["readout_count"]
    assert r[0] == base[0] and r[1] == 0
    e.set_weights(None)
    r = e.run([([0], 150.0)], max_ms=1000, seed=3, readouts=[1, 2])["readout_count"]
    assert (r == base).all()


def test_deterministic(tmp_path):
    e = chain(tmp_path)
    a = e.run([([0], 150.0)], max_ms=500, seed=7, readouts=[0, 1, 2], full=True)
    b = e.run([([0], 150.0)], max_ms=500, seed=7, readouts=[0, 1, 2], full=True)
    assert (a["spike_count"] == b["spike_count"]).all() and (a["first_spike_ms"] == b["first_spike_ms"]).all()


def test_inhibition(tmp_path):
    p = tmp_path / "inh.bin"  # 0 excites 2, 1 inhibits 2
    write_graph(p, 3, [0, 1, 2, 2], [2, 2], [30.0, -60.0])
    e = Engine(p, 1.0)
    alone = e.run([([0], 150.0)], max_ms=1000, seed=2, readouts=[2])["readout_count"][0]
    both = e.run([([0], 150.0), ([1], 150.0)], max_ms=1000, seed=2, readouts=[2])["readout_count"][0]
    assert alone > 0 and both < alone / 4


@pytest.mark.skipif(not graph_path("flywire_ge5").exists(), reason="FlyWire graph not built")
def test_flywire_sugar_reference():
    """Published reference of the Shiu et al. 2024 model: 129 sugar/water GRNs at 150 Hz -> MN9 about
    150 Hz (right) and 111 Hz (left) on FlyWire 783 with >=5 synapses, dt 0.2 ms, 1 s, 10 trials."""
    from brainloss.sets import readouts, stimuli
    e = Engine(graph_path("flywire_ge5"), 1.0)
    ro = readouts("flywire")
    rid = [ro["MN9_R"], ro["MN9_L"]]
    cnt = np.zeros(2)
    for tr in range(10):
        cnt += e.run([(stimuli("flywire")["sugar"], 150.0)], dt=0.2, max_ms=1000, seed=tr, readouts=rid)["readout_count"]
    hz = cnt / 10
    assert abs(hz[0] - 150) / 150 < 0.05 and abs(hz[1] - 111) / 111 < 0.05, hz
