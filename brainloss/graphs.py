"""Connectome tables -> engine graph files (CSR, rows = presynaptic, data = synapse count * sign).

Neuron tables get the same columns for both connectomes:
    idx, id, type, side (L/R/M), superclass, class, subclass, nt, sign, x, y, z (um), has_pos
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

from .paths import BUILD, DATA, graph_path, neurons_path
from evlif import read_graph, write_graph

MCNS = DATA / "malecns"
ANN = MCNS / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NTF = MCNS / "body-neurotransmitters-male-cns-v1.0.feather"
WTS = MCNS / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
FW = DATA / "flywire"

NEG = {"gaba", "glutamate", "histamine"}
POS = {"acetylcholine", "dopamine", "serotonin", "octopamine"}


def malecns_sign(consensus, celltype):
    """Sign by consensus_nt; if unclear or missing, by celltype_predicted_nt; otherwise +1.
    Histamine counts as inhibitory."""
    def classify(s):
        s = pd.Series(s, dtype="string").str.lower()
        out = np.full(len(s), np.nan, np.float32)
        out[s.isin(NEG).fillna(False).to_numpy()] = -1.0
        out[s.isin(POS).fillna(False).to_numpy()] = 1.0
        return out, s
    sc, c = classify(consensus)
    st, t = classify(celltype)
    sign = np.ones(len(sc), np.float32)
    use_c = ~np.isnan(sc)
    use_t = ~use_c & ~np.isnan(st)
    sign[use_c] = sc[use_c]
    sign[use_t] = st[use_t]
    nt = np.where(use_c, c.fillna("").to_numpy(), np.where(use_t, t.fillna("").to_numpy(), "unclear"))
    branch = np.where(use_c, "consensus", np.where(use_t, "celltype", "default"))
    return sign, nt.astype(str), branch


def _csr(N, pre, post, cnt, sign):
    order = np.lexsort((post, pre))
    pre, post, cnt = pre[order], post[order], cnt[order]
    indptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=N), out=indptr[1:])
    return indptr, post.astype(np.int32), (cnt * sign[pre]).astype(np.float32)


def build_malecns(thresholds=(5, 1)):
    rep = {}
    cols = ["bodyId", "status", "type", "flywireType", "instance", "superclass", "class", "subclass",
            "somaSide", "rootSide", "somaLocation", "tosomaLocation"]
    ann = pf.read_table(ANN, columns=cols)
    ann = ann.filter(pc.equal(ann.column("status"), "Traced")).to_pandas()
    ann = ann.sort_values("bodyId").reset_index(drop=True)
    N = len(ann)
    ids = ann.bodyId.to_numpy(np.int64)
    rep["neurons_traced"] = N

    def xyz(col):
        a = np.full((N, 3), np.nan)
        for i, v in enumerate(ann[col].to_numpy()):
            if v is not None and len(v) == 3:
                a[i] = v
        return a
    s, ts = xyz("somaLocation"), xyz("tosomaLocation")
    has_s = ~np.isnan(s).any(1)
    pos = np.where(has_s[:, None], s, ts) * 0.008  # 8 nm voxels -> um
    has_pos = ~np.isnan(pos).any(1)

    nt = pf.read_table(NTF, columns=["body", "consensus_nt", "celltype_predicted_nt"])
    nt = nt.filter(pc.is_in(nt.column("body"), value_set=pa.array(ids))).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body").reindex(ids)
    sign, ntname, branch = malecns_sign(nt.consensus_nt.to_numpy(), nt.celltype_predicted_nt.to_numpy())
    rep["sign_branches"] = pd.Series([f"{b}:{n}" for b, n in zip(branch, ntname)]).value_counts().to_dict()
    rep["inhibitory_neurons"] = int((sign < 0).sum())

    w = pf.read_table(WTS, columns=["body_pre", "body_post", "weight"])
    vs = pa.array(ids)
    w = w.filter(pc.and_(pc.is_in(w.column("body_pre"), value_set=vs), pc.is_in(w.column("body_post"), value_set=vs)))
    pre = np.searchsorted(ids, w.column("body_pre").to_numpy()).astype(np.int64)
    post = np.searchsorted(ids, w.column("body_post").to_numpy()).astype(np.int64)
    cnt = w.column("weight").to_numpy().astype(np.int64)
    del w
    key = pre * N + post
    rep["duplicate_pairs"] = int(len(key) - len(np.unique(key)))
    assert rep["duplicate_pairs"] == 0
    rep["synapses_total"] = int(cnt.sum())
    for th in thresholds:
        k = cnt >= th
        indptr, p2, data = _csr(N, pre[k], post[k], cnt[k], sign)
        name = f"malecns_ge{th}" if th > 1 else "malecns_all"
        write_graph(graph_path(name), N, indptr, p2, data)
        rep[name] = {"edges": int(k.sum()), "synapses": int(cnt[k].sum()), "inhibitory_edges": int((data < 0).sum())}
        print(name, rep[name], flush=True)

    side = ann.somaSide.fillna(ann.rootSide).astype("string")
    df = pd.DataFrame({
        "idx": np.arange(N, dtype=np.int32), "id": ids, "type": ann["type"].astype("string"),
        "flywireType": ann.flywireType.astype("string"), "instance": ann.instance.astype("string"),
        "side": side, "superclass": ann.superclass.astype("string"), "class": ann["class"].astype("string"),
        "subclass": ann.subclass.astype("string"), "nt": ntname, "sign": sign, "sign_source": branch,
        "x": pos[:, 0].astype(np.float32), "y": pos[:, 1].astype(np.float32), "z": pos[:, 2].astype(np.float32),
        "has_pos": has_pos,
    })
    df.to_parquet(neurons_path("malecns"), index=False)
    rep["with_position"] = int(has_pos.sum())
    json.dump(rep, open(BUILD / "malecns_build.json", "w"), indent=1, ensure_ascii=False)
    return rep


def build_flywire(th=5):
    """FlyWire 783: pairs summed over neuropils, threshold on the sum, sign by top_nt (as in Shiu et al. 2024)."""
    neu = pd.read_csv(FW / "Supplemental_file1_neuron_annotations.tsv", sep="\t", dtype={"root_id": np.int64},
                      low_memory=False)
    N = len(neu)
    t = pf.read_table(FW / "proofread_connections_783.feather",
                      columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]).to_pandas()
    t = t.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False, as_index=False)["syn_count"].sum()
    index = pd.Index(neu.root_id)
    pre = index.get_indexer(t.pre_pt_root_id.to_numpy())
    post = index.get_indexer(t.post_pt_root_id.to_numpy())
    cnt = t.syn_count.to_numpy()
    k = (pre >= 0) & (post >= 0) & (cnt >= th)
    sign = np.where(neu.top_nt.isin(["gaba", "glutamate"]).to_numpy(), -1.0, 1.0).astype(np.float32)
    indptr, p2, data = _csr(N, pre[k].astype(np.int64), post[k].astype(np.int64), cnt[k], sign)
    write_graph(graph_path(f"flywire_ge{th}"), N, indptr, p2, data)
    side = neu.side.map({"left": "L", "right": "R", "center": "M"}).astype("string")
    df = pd.DataFrame({
        "idx": np.arange(N, dtype=np.int32), "id": neu.root_id.to_numpy(np.int64),
        "type": neu.cell_type.astype("string"), "side": side, "superclass": neu.super_class.astype("string"),
        "class": neu.cell_class.astype("string"), "subclass": neu.cell_sub_class.astype("string"),
        "nt": neu.top_nt.astype("string"), "sign": sign,
    })
    df.to_parquet(neurons_path("flywire"), index=False)
    rep = {"neurons": N, "edges": int(k.sum())}
    print(f"flywire_ge{th}", rep)
    return rep


def build_shuffle(name, seed=12345):
    """Degree-preserving control: postsynaptic targets are permuted separately among excitatory and among
    inhibitory edges. Every neuron keeps its outgoing weights and its in-degree per sign."""
    N, E, indptr, post, w = read_graph(graph_path(name), mmap=False)
    rng = np.random.default_rng(seed)
    post2 = post.copy()
    for m in (w > 0, w < 0):
        pos = np.flatnonzero(m)
        post2[pos] = post[pos][rng.permutation(len(pos))]
    for m in (w > 0, w < 0):
        assert (np.bincount(post[m], minlength=N) == np.bincount(post2[m], minlength=N)).all()
    write_graph(graph_path(name + "_shuf"), N, indptr, post2, w)
    pre = np.repeat(np.arange(N), np.diff(indptr))
    print(name + "_shuf", {"edges": int(E), "self_loops": int((post2 == pre).sum())})


def build_cloud():
    """Soma cloud for the pictures: float32 x, y, z (um) and has_pos per neuron, MaleCNS order."""
    n = pd.read_parquet(neurons_path("malecns"))
    xyz = n[["x", "y", "z"]].to_numpy(np.float32)
    np.savez_compressed(BUILD / "malecns_cloud.npz", xyz=xyz, has_pos=n.has_pos.to_numpy(),
                        superclass=n.superclass.fillna("").to_numpy().astype(str))
