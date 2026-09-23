"""Charts for methods/ and the README (matplotlib).
    python scripts/methods_charts.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainloss.analysis import SKILLS  # noqa: E402
from brainloss.paths import METHODS, RESULTS  # noqa: E402

BG, FG, MUTED, SUB, GRID = "#090b10", "#f4f1ec", "#a7a29b", "#6f6a63", "#1a1f2b"
COL = {"eat": "#f66f14", "escape": "#ffb070", "turn_away": (0.957, 0.945, 0.925, 0.55),
       "turn_to": (0.957, 0.945, 0.925, 0.36), "head": (0.957, 0.945, 0.925, 0.22), "bitter": "#ef5f0a"}
NAME = {"eat": "ест (MN9)", "escape": "улетает от тени (DNp01)", "turn_away": "уходит от угрозы (DNa01, DNa02 справа)",
        "turn_to": "следит за движением (DNa02 слева)", "head": "реагирует на касание (DNg15/35/84/85/48)",
        "bitter": "бросает горькое"}
SHORT = {"eat": "ест", "escape": "улетает от тени", "turn_away": "уходит от угрозы",
         "turn_to": "следит за движением", "head": "реагирует на касание"}


def main_curves(cv, p50s, out):
    """One panel for the README: the main run, every skill labelled at its end."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    x = 100 * cv.frac.to_numpy()
    ends = []
    for s in ("head", "turn_to", "turn_away", "escape", "eat"):
        m = cv[f"{s}_med"].to_numpy()
        if s in ("eat", "escape"):
            ax.fill_between(x, cv[f"{s}_q25"], cv[f"{s}_q75"], color=COL[s], alpha=0.14, lw=0)
        ax.plot(x, m, color=COL[s], lw=2.6 if s in ("eat", "escape") else 1.8)
        ends.append([m[-1], s])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.09)
    for y, s in ends:
        ax.text(96.5, y, SHORT[s], color=COL[s] if s in ("eat", "escape") else MUTED, fontsize=10, va="center")
    pe = p50s.get("eat")
    if pe is not None:
        ax.plot([pe], [0.5], "o", ms=8, mfc=BG, mec=COL["eat"], mew=2.4)
        ax.annotate(f"ест вдвое меньше уже при {round(pe)} %", (pe, 0.5), xytext=(10, 12), textcoords="offset points",
                    color=FG, fontsize=10)
    ax.axhline(0.5, color=SUB, lw=0.8, ls=":")
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 1.75)
    ax.set_xticks([0, 20, 40, 60, 80, 95])
    ax.set_xlabel("убрано нейронов, %")
    ax.set_ylabel("выход / целая модель той же мухи")
    ax.set_title("MaleCNS, полный граф, режим A: медиана по 40 маскам и 5 мухам", fontsize=10, loc="left", color=FG)
    style(ax)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.9, bottom=0.12)
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUB, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def curves_panel(ax, cv, skills=SKILLS, title=""):
    x = 100 * cv.frac.to_numpy()
    for s in skills:
        if f"{s}_med" not in cv or cv[f"{s}_med"].isna().all():
            continue
        m = cv[f"{s}_med"].to_numpy()
        ax.fill_between(x, cv[f"{s}_q25"], cv[f"{s}_q75"], color=COL[s], alpha=0.14, lw=0)
        ax.plot(x, m, color=COL[s], lw=2.2 if s in ("eat", "escape") else 1.6, label=NAME[s])
    ax.axhline(0.5, color=SUB, lw=0.8, ls=":")
    ax.set_xlim(0, 95)
    ax.set_ylim(0, max(1.3, np.nanmax([cv[f"{s}_q75"].max() for s in skills if f"{s}_q75" in cv]) * 1.05))
    ax.set_xlabel("убрано нейронов, %")
    ax.set_ylabel("выход / целый мозг той же мухи")
    ax.set_title(title, fontsize=10, loc="left", color=FG)
    style(ax)


def main():
    METHODS.mkdir(exist_ok=True)
    mc = RESULTS / "malecns"
    main = mc / "curves_malecns_all_w0.152_s80_A.csv"
    if main.exists():
        p = pd.read_csv(mc / "p50_malecns_all_w0.152_s80_A.csv").iloc[0]
        main_curves(pd.read_csv(main), {"eat": None if pd.isna(p.eat) else float(p.eat)}, METHODS / "main_curves.png")
    files = sorted(mc.glob("curves_malecns_*_[AB].csv")) + sorted(mc.glob("curves_malecns_*_syn.csv"))
    files += [RESULTS / "flywire" / "curves_A.csv"]
    files = [f for f in files if f.exists()]
    n = len(files)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 4.2 * ((n + 1) // 2)), facecolor=BG, squeeze=False)
    for ax, f in zip(axes.ravel(), files):
        cv = pd.read_csv(f)
        curves_panel(ax, cv, title=f"{f.parent.name}: {f.stem.replace('curves_', '')}")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, labelcolor=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(METHODS / "curves_all.png", dpi=130, facecolor=BG)
    plt.close(fig)

    # neurons vs synapses on one axis: share of synapses gone
    for g in ("malecns_all", "malecns_ge5"):
        share = mc / f"syn_share_{g}.csv"
        runsA = mc / f"curves_{g}_w0.152_s80_A_runs.csv"
        syn = mc / f"curves_{g}_w0.152_s80_syn.csv"
        if not (share.exists() and runsA.exists() and syn.exists()):
            continue
        sh = pd.read_csv(share)
        sh = sh[sh["mode"] == "A"].groupby("frac").syn_lost.median()
        cvA = pd.read_csv(mc / f"curves_{g}_w0.152_s80_A.csv")
        cvS = pd.read_csv(syn)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=BG)
        for ax, s in zip(axes, ("eat", "escape", "head")):
            ax.plot(100 * sh.reindex(cvA.frac).to_numpy(), cvA[f"{s}_med"], color=COL[s] if s != "head" else "#ffb070", lw=2, label="убираем нейроны")
            ax.plot(100 * cvS.frac, cvS[f"{s}_med"], color=MUTED, lw=1.6, ls="--", label="убираем синапсы россыпью")
            ax.set_xlabel("ушло синапсов, %")
            ax.set_title(NAME[s], fontsize=10, loc="left", color=FG)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.4)
            style(ax)
            ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)
        fig.tight_layout()
        fig.savefig(METHODS / f"neurons_vs_synapses_{g}.png", dpi=130, facecolor=BG)
        plt.close(fig)

    # activity along the curves: does feeding stay free of the olfactory runaway?
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    for f in sorted(mc.glob("curves_malecns_*_w0.152_s80_[AB]_runs.csv")):
        df = pd.read_csv(f, usecols=lambda c: c in ("frac", "active_eat", "active_loom"))
        m = df.groupby("frac").active_eat.median()
        ax.plot(100 * m.index, m.values, lw=1.6, label=f.stem.replace("curves_", "").replace("_runs", ""))
    ax.set_yscale("log")
    ax.set_xlabel("убрано нейронов, %")
    ax.set_ylabel("активных нейронов при сахаре (медиана)")
    style(ax)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(METHODS / "activity_eat.png", dpi=130, facecolor=BG)
    plt.close(fig)
    print("written to", METHODS)


if __name__ == "__main__":
    main()
