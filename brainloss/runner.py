"""Parallel runs: every task is one (removal mask, fly) pair with all its conditions and trials.

Masks are nested: a mask seed fixes a random order of the removable neurons, and fraction p removes
the first round(p * pool) of them, so 10 % is contained in 20 %, and so on.
Results go to a JSON-lines file line by line; finished tasks are skipped when the run is restarted.
"""
import json
import multiprocessing as mp
import os
import time
from functools import lru_cache

import numpy as np

from . import paths  # noqa: F401  (puts engine/ on sys.path)
from .paths import graph_path
from .sets import readouts, stimuli
from .skills import READOUT_NAMES

from evlif import Engine, read_graph, vth_for_fly

_eng = {}
PROTECTED_INPUTS = ("sugar", "bitter", "loom", "lc10L", "head")  # every input set that is stimulated


def connectome_of(graph):
    return graph.split("_")[0]


@lru_cache(None)
def pool_of(graph, mode):
    """Removable neurons. A: everything except stimulated inputs and readouts. B: everything except readouts."""
    c = connectome_of(graph)
    N = int(np.fromfile(graph_path(graph), np.int64, 1)[0])
    prot = np.zeros(N, bool)
    prot[list(readouts(c).values())] = True
    if mode == "A":
        S = stimuli(c)
        for k in PROTECTED_INPUTS:
            prot[S[k]] = True
    return np.flatnonzero(~prot).astype(np.int32)


@lru_cache(8)
def order_of(graph, mode, mask_seed):
    pool = pool_of(graph, mode)
    rng = np.random.default_rng(np.random.SeedSequence([int(mask_seed), 104729]))
    return pool[rng.permutation(len(pool))]


def removed_for(graph, mode, frac, mask_seed):
    order = order_of(graph, mode, mask_seed)
    return order[: int(round(frac * len(order)))]


@lru_cache(2)
def _graph_counts(graph):
    return read_graph(graph_path(graph), mmap=True)


def thinned_weights(graph, q, seed):
    """Synapse-removal control: every synapse is dropped independently with probability q."""
    N, E, indptr, post, w = _graph_counts(graph)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3571]))
    cnt = np.abs(w).astype(np.int64)
    kept = rng.binomial(cnt, 1.0 - q)
    return (np.sign(w) * kept).astype(np.float32), float(1.0 - kept.sum() / cnt.sum())


def engine(graph, w):
    key = (graph, float(w))
    if key not in _eng:
        for k in list(_eng):
            _eng.pop(k).close()
        _eng[key] = Engine(graph_path(graph), w)
    return _eng[key]


def run_task(t):
    """t: dict(graph, w, dt, T, trials, fly, sd, conds={name: [(set, Hz)]}, removed=[...] or mask=(mode, frac, seed),
    syn=(q, seed) or None, extra_removed=[...], seed, key). Returns a result dict."""
    t0 = time.time()
    g = t["graph"]
    c = connectome_of(g)
    eng = engine(g, t["w"])
    S = stimuli(c)
    ro = readouts(c)
    rid = np.array([ro[n] for n in READOUT_NAMES], np.int32)
    syn_info = None
    if t.get("syn"):
        wts, lost = thinned_weights(g, *t["syn"])
        eng.set_weights(wts)
        syn_info = lost
    elif getattr(eng, "_thinned", False):
        eng.set_weights(None)
    eng._thinned = bool(t.get("syn"))
    if "mask" in t:
        mode, frac, ms = t["mask"]
        rem = removed_for(g, mode, frac, ms)
    else:
        rem = np.asarray(t.get("removed", []), np.int32)
    if t.get("extra_removed"):
        rem = np.union1d(rem, np.asarray(t["extra_removed"], np.int32))
    eng.set_removed(rem if len(rem) else None)
    eng.set_vth(vth_for_fly(eng.N, t["fly"], t.get("sd", 0.5)))
    out = {"key": t["key"], "n_removed": int(len(rem)), "syn_lost": syn_info, "conds": {}}
    for ci, (cname, groups) in enumerate(t["conds"].items()):
        G = [(S[s] if isinstance(s, str) else np.asarray(s, np.int32), hz) for s, hz in groups]
        cnt = np.zeros(len(rid))
        first = np.zeros(len(rid))
        nfirst = np.zeros(len(rid))
        act = 0.0
        wall = 0.0
        for tr in range(t["trials"]):
            seed = (int(t["seed"]) * 1000003 + ci * 7919 + tr) & 0xFFFFFFFFFFFF
            r = eng.run(G, dt=t["dt"], max_ms=t["T"], seed=seed, readouts=rid)
            cnt += r["readout_count"]
            f = r["readout_first_ms"]
            first[f >= 0] += f[f >= 0]
            nfirst[f >= 0] += 1
            act += r["active"]
            wall += r["wall_ms"]
        bio_s = t["trials"] * t["T"] / 1000.0
        out["conds"][cname] = {
            "hz": dict(zip(READOUT_NAMES, np.round(cnt / bio_s, 3).tolist())),
            "first_ms": dict(zip(READOUT_NAMES, np.where(nfirst > 0, np.round(first / np.maximum(nfirst, 1), 2), -1).tolist())),
            "active": act / t["trials"], "wall_ms": wall,
        }
    out["wall_s"] = time.time() - t0
    return out


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_all(tasks, out_path, workers=None, log_every=50):
    """Runs tasks in a process pool, appends results to out_path (JSON lines), skips keys already there."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [t for t in tasks if t["key"] not in done]
    # group tasks of one graph together so each worker loads it once
    todo.sort(key=lambda t: (t["graph"], t["w"], bool(t.get("syn"))))
    if not todo:
        print(f"{out_path}: nothing to do ({len(done)} done)")
        return
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{out_path}: {len(todo)} tasks ({len(done)} already done), {workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool, open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % log_every == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f} s, ~{el / i * (len(todo) - i):.0f} s left", flush=True)


def load(out_path):
    with open(out_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
