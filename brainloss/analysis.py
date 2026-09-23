"""Curves from raw runs: output of a skill relative to the same fly with the intact brain."""
import numpy as np
import pandas as pd

from .runner import load
from .skills import skill_values

SKILLS = ["eat", "bitter", "escape", "turn_away", "turn_to", "head"]


def table(path, meta_keys=("graph", "w", "mode", "frac", "mask", "fly")):
    """One row per task with skill outputs. The task key is 'graph|w|mode|frac|mask|fly|...'."""
    rows = []
    for r in load(path):
        parts = r["key"].split("|")
        d = dict(zip(meta_keys, parts))
        d["frac"] = float(d["frac"])
        d["mask"] = int(d["mask"])
        d["fly"] = int(d["fly"])
        d["n_removed"] = r["n_removed"]
        d["syn_lost"] = r.get("syn_lost")
        d.update(skill_values(r["conds"]))
        for c, v in r["conds"].items():
            d[f"active_{c}"] = v["active"]
            for n, hz in v["hz"].items():
                d[f"{c}:{n}"] = hz
        rows.append(d)
    return pd.DataFrame(rows)


def relative(df, skills=SKILLS, group=("graph", "w", "mode")):
    """Adds <skill>_rel = output / median intact output of the same fly (fraction 0 of the same group).
    The median keeps the intact point at 1 despite Poisson noise; with the mean it sits a few per cent lower."""
    df = df.copy()
    keys = list(group) + ["fly"]
    base = df[df.frac == 0].groupby(keys)[list(skills)].median()
    for s in skills:
        b = df.join(base[[s]].rename(columns={s: "_b"}), on=keys)["_b"]
        df[s + "_rel"] = np.where(b > 0, df[s] / b, np.nan)
        df[s + "_base"] = b
    return df


# a skill that exists only while another one works (the bitter veto needs eating) is measured only where
# the other one still gives a response; its median is used only where most masks can be measured
CONDITIONAL = {"bitter"}
MIN_MEASURABLE = 0.5


def curves(df, skills=SKILLS, group=("graph", "w", "mode")):
    """Median and quartiles of <skill>_rel over masks and flies, per fraction."""
    out = []
    for key, g in df.groupby(list(group) + ["frac"]):
        row = dict(zip(list(group) + ["frac"], key))
        row["n"] = len(g)
        for s in skills:
            v = g[s + "_rel"].dropna()
            row[f"{s}_n"] = len(v)
            row[f"{s}_measurable"] = len(v) / len(g)
            ok = len(v) and (s not in CONDITIONAL or len(v) / len(g) >= MIN_MEASURABLE)
            row[f"{s}_med"] = float(v.median()) if ok else np.nan
            row[f"{s}_q25"] = float(v.quantile(0.25)) if ok else np.nan
            row[f"{s}_q75"] = float(v.quantile(0.75)) if ok else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(list(group) + ["frac"]).reset_index(drop=True)


def p50_status(fracs, med, thr=0.5):
    """(p50, status): status is 'crossed', 'never' (stays above thr up to the last fraction) or
    'unmeasurable' (the curve ends before it falls below thr)."""
    prev = None
    for p, m in zip(fracs, med):
        if np.isnan(m):
            return None, "unmeasurable" if prev is not None else "never"
        if m < thr:
            if prev is None:
                return float(p), "crossed"
            p0, m0 = prev
            return float(p0 + (p - p0) * (m0 - thr) / (m0 - m)), "crossed"
        prev = (p, m)
    return None, "never"


def p50(fracs, med, thr=0.5):
    """First fraction where the median falls below thr, linearly interpolated; None if it does not."""
    return p50_status(fracs, med, thr)[0]


def p50_boot(df, skill, n=1000, seed=0):
    """95 % interval of p50. Whole masks are resampled with all their flies, because the flies of one mask
    share the removed neurons. Draws that never fall below 0.5 count as p50 = +inf.
    Returns (lo, hi, share_never); hi = inf means the upper bound is not reached."""
    rng = np.random.default_rng(seed)
    piv = df.pivot_table(index=["mask", "fly"], columns="frac", values=skill + "_rel", aggfunc="mean", dropna=False)
    fr = piv.columns.to_numpy()
    masks = piv.index.get_level_values("mask").to_numpy()
    ids = np.unique(masks)
    rows = {m: np.flatnonzero(masks == m) for m in ids}
    X = piv.to_numpy()
    vals = []
    for _ in range(n):
        pick = np.concatenate([rows[m] for m in rng.choice(ids, len(ids))])
        s = X[pick]
        with np.errstate(all="ignore"):
            med = np.nanmedian(s, axis=0)
            if skill in CONDITIONAL:
                med[np.mean(~np.isnan(s), axis=0) < MIN_MEASURABLE] = np.nan
        v, st = p50_status(fr, med)
        vals.append(v if st == "crossed" else np.inf)
    vals = np.array(vals)
    lo, hi = np.quantile(vals, [0.025, 0.975], method="nearest")
    return float(lo), float(hi), float(np.mean(np.isinf(vals)))


def ci_text(lo, hi):
    if np.isinf(lo):
        return "не падает вдвое"
    return f"{100 * lo:.0f}-{100 * hi:.0f}" if np.isfinite(hi) else f"{100 * lo:.0f}-(не доходит)"


def p50_table(cv, skills=SKILLS, group=("graph", "w", "mode")):
    rows = []
    for key, g in cv.groupby(list(group)):
        row = dict(zip(group, key))
        for s in skills:
            v, st = p50_status(g.frac.to_numpy(), g[f"{s}_med"].to_numpy())
            row[s] = None if v is None else round(100 * v, 1)
            row[f"{s}_status"] = st
            last = g[g.frac == g.frac.max()][f"{s}_med"]
            row[f"{s}_at_max"] = float(last.iloc[0]) if len(last) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
