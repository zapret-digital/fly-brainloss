"""ctypes wrapper for the evlif engine (evlif.c). One Engine = one graph loaded in memory."""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB_NAME = {"win32": "evlif.dll", "darwin": "libevlif.dylib"}.get(sys.platform, "libevlif.so")
_lib = None


class StimGroup(C.Structure):
    _fields_ = [("idx", C.POINTER(C.c_int32)), ("n", C.c_int32), ("rate_hz", C.c_float),
                ("on_ms", C.c_float), ("off_ms", C.c_float)]


class Result(C.Structure):
    _fields_ = [("spike_count", C.POINTER(C.c_int32)), ("first_spike_ms", C.POINTER(C.c_float)),
                ("readout_first_ms", C.POINTER(C.c_float)), ("readout_count", C.POINTER(C.c_int32)),
                ("bin_ms", C.c_float), ("nbins", C.c_int32), ("bin_spikes", C.POINTER(C.c_int32)),
                ("readout_bins", C.POINTER(C.c_int32)), ("bio_ms", C.c_float), ("wall_ms", C.c_float),
                ("awake_after", C.c_int32), ("active", C.c_int32), ("total_spikes", C.c_int64),
                ("syn_events", C.c_int64), ("mean_awake", C.c_double), ("quiet_exit", C.c_int32),
                ("reserved", C.c_int32)]


def lib():
    global _lib
    if _lib is None:
        path = HERE / LIB_NAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, build it first: python engine/build.py")
        L = C.CDLL(str(path))
        L.ev_load.restype = C.c_void_p
        L.ev_load.argtypes = [C.c_char_p, C.c_float]
        L.ev_free.argtypes = [C.c_void_p]
        L.ev_n.restype = C.c_int32
        L.ev_n.argtypes = [C.c_void_p]
        L.ev_nnz.restype = C.c_int64
        L.ev_nnz.argtypes = [C.c_void_p]
        L.ev_reset.argtypes = [C.c_void_p]
        L.ev_set_vth.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        L.ev_set_removed.argtypes = [C.c_void_p, C.POINTER(C.c_uint8)]
        L.ev_set_weights.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        L.ev_run.restype = C.c_int
        L.ev_run.argtypes = [C.c_void_p, C.POINTER(StimGroup), C.c_int32, C.c_float, C.c_float, C.c_uint64,
                             C.POINTER(C.c_int32), C.c_int32, C.POINTER(Result)]
        _lib = L
    return _lib


def _p(a, t):
    return a.ctypes.data_as(C.POINTER(t)) if a is not None else None


class Engine:
    def __init__(self, graph_path, w_scale=1.0):
        self.path = str(graph_path)
        self.w_scale = float(w_scale)
        self.h = lib().ev_load(self.path.encode(), self.w_scale)
        if not self.h:
            raise IOError(f"cannot load graph {graph_path}")
        self.N = lib().ev_n(self.h)
        self.nnz = lib().ev_nnz(self.h)
        self._cnt = np.zeros(self.N, np.int32)
        self._first = np.zeros(self.N, np.float32)
        self._keep = {}

    def close(self):
        if getattr(self, "h", None):
            lib().ev_free(self.h)
            self.h = None

    __del__ = close

    def reset(self):
        lib().ev_reset(self.h)

    def set_vth(self, vth=None):
        """Per-neuron threshold in mV; None = -45 mV everywhere."""
        if vth is None:
            self._keep.pop("vth", None)
            lib().ev_set_vth(self.h, None)
            return
        a = np.ascontiguousarray(vth, np.float32)
        assert len(a) == self.N
        self._keep["vth"] = a
        lib().ev_set_vth(self.h, _p(a, C.c_float))

    def set_removed(self, removed=None):
        """Boolean mask [N] or index array of neurons to take out of the network; None = intact."""
        if removed is None:
            lib().ev_set_removed(self.h, None)
            return
        r = np.asarray(removed)
        if r.dtype == bool and len(r) == self.N:
            m = r.astype(np.uint8)
        else:
            m = np.zeros(self.N, np.uint8)
            m[r.astype(np.int64)] = 1
        m = np.ascontiguousarray(m)
        lib().ev_set_removed(self.h, _p(m, C.c_uint8))

    def set_weights(self, counts=None):
        """New count*sign per edge (same order as the graph file); None restores the file."""
        if counts is None:
            lib().ev_set_weights(self.h, None)
            return
        a = np.ascontiguousarray(counts, np.float32)
        assert len(a) == self.nnz
        lib().ev_set_weights(self.h, _p(a, C.c_float))

    def run(self, groups, dt=0.2, max_ms=300.0, seed=0, readouts=(), bin_ms=0.0, full=False, reset=True):
        """groups: list of (index_array, rate_hz[, on_ms, off_ms]). Returns a dict with readout counts,
        first-spike times and, with full=True, spike counts and first-spike times of every neuron."""
        if reset:
            self.reset()
        keep = []
        G = (StimGroup * max(1, len(groups)))()
        for k, grp in enumerate(groups):
            idx, rate = grp[0], grp[1]
            on = grp[2] if len(grp) > 2 else 0.0
            off = grp[3] if len(grp) > 3 else 0.0
            a = np.ascontiguousarray(np.asarray(idx, np.int32))
            keep.append(a)
            G[k] = StimGroup(_p(a, C.c_int32), len(a), float(rate), float(on), float(off))
        ro = np.ascontiguousarray(np.asarray(readouts, np.int32))
        assert len(np.unique(ro)) == len(ro), "readouts must be unique neurons"
        nr = len(ro)
        rfirst = np.zeros(max(1, nr), np.float32)
        rcount = np.zeros(max(1, nr), np.int32)
        nb = int(np.ceil(max_ms / bin_ms)) if bin_ms > 0 else 0
        bins = np.zeros(max(1, nb), np.int32)
        rbins = np.zeros(max(1, nb * nr), np.int32)
        r = Result()
        if full:
            r.spike_count = _p(self._cnt, C.c_int32)
            r.first_spike_ms = _p(self._first, C.c_float)
        r.readout_first_ms = _p(rfirst, C.c_float)
        r.readout_count = _p(rcount, C.c_int32)
        r.bin_ms = bin_ms
        r.nbins = nb
        r.bin_spikes = _p(bins, C.c_int32)
        r.readout_bins = _p(rbins, C.c_int32)
        lib().ev_run(self.h, G, len(groups), dt, max_ms, int(seed) & 0xFFFFFFFFFFFFFFFF,
                     _p(ro, C.c_int32) if nr else None, nr, C.byref(r))
        out = dict(bio_ms=r.bio_ms, wall_ms=r.wall_ms, awake_after=r.awake_after, active=r.active,
                   total_spikes=r.total_spikes, syn_events=r.syn_events, mean_awake=r.mean_awake,
                   quiet_exit=r.quiet_exit, readout_first_ms=rfirst[:nr].copy(), readout_count=rcount[:nr].copy())
        if full:
            out["spike_count"] = self._cnt.copy()
            out["first_spike_ms"] = self._first.copy()
        if nb:
            out["bin_spikes"] = bins[:nb].copy()
            out["readout_bins"] = rbins[:nb * nr].reshape(nr, nb).copy()
        return out


def vth_for_fly(n, fly, sd):
    """Per-neuron threshold ~ N(-45, sd) mV fixed by the fly seed; fly 0 or sd 0 = uniform -45 mV."""
    if sd <= 0 or fly == 0:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([int(fly), 7919]))
    return (-45.0 + sd * rng.standard_normal(n)).astype(np.float32)


def read_graph(path, mmap=True):
    """(N, E, indptr, post, count*sign) from a graph file, memory-mapped by default."""
    hdr = np.fromfile(path, np.int64, 2)
    N, E = int(hdr[0]), int(hdr[1])
    o = 16
    mode = "r" if mmap else None

    def arr(dtype, n, off):
        if mmap:
            return np.memmap(path, dtype, mode, offset=off, shape=(n,))
        with open(path, "rb") as f:
            f.seek(off)
            return np.fromfile(f, dtype, n)

    indptr = arr(np.int64, N + 1, o)
    o += (N + 1) * 8
    post = arr(np.int32, E, o)
    o += E * 4
    w = arr(np.float32, E, o)
    return N, E, indptr, post, w


def write_graph(path, N, indptr, post, w):
    with open(path, "wb") as f:
        np.array([N, len(post)], np.int64).tofile(f)
        np.asarray(indptr, np.int64).tofile(f)
        np.asarray(post, np.int32).tofile(f)
        np.asarray(w, np.float32).tofile(f)


def packet(res):
    """Cascade packet: neurons that spiked with their first-spike time (ms) and spike count."""
    cnt = res["spike_count"]
    first = res["first_spike_ms"]
    nz = np.flatnonzero(cnt > 0)
    return {"idx": nz.astype(np.int32), "t_ms": first[nz].astype(np.float32), "count": cnt[nz].astype(np.int32)}
