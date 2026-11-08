#Figure 1 (mislabel rate by tissue) and Figure 2 (sex balance by tissue).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS, COLORS, FIG_DPI, MIN_PER_SEX


def set_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",          # keeps text editable in the SVG
    })


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved results/{name}.png and .svg")


def figure1():
    """Mislabel rate (a) and missing-label rate (b) by tissue, with 95% Wilson CIs."""
    t_all = pd.read_csv(RESULTS / "mislabel_final.csv", index_col=0)
    overall = t_all.loc["ALL"]
    t = t_all.drop(index="ALL")
    t = t[t["n_callable"] > 0].sort_values("mislabel_pct")

    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.3 * len(t) + 1.5), sharey=True)
    panels = [
        ("mislabel", "Mislabeled\n(% of lines with confident call)", COLORS["mislabel"]),
        ("missing", "Missing sex label\n(% of all lines)", COLORS["missing"]),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        val = t[f"{key}_pct"]
        err = np.vstack([val - t[f"{key}_CI_low"], t[f"{key}_CI_high"] - val]).clip(min=0)
        ax.barh(y, val, xerr=err, color=color, ecolor=COLORS["accent"],
                capsize=2, error_kw={"lw": 0.8})
        ax.axvline(overall[f"{key}_pct"], ls="--", lw=0.8, color=COLORS["accent"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("%")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{name} (n={int(n)})" for name, n in zip(t.index, t["n"])])
    fig.text(0.01, -0.02, "Bars: 95% Wilson CI. Dashed line: rate across all tissues.", fontsize=7)
    save(fig, "fig1_mislabel_by_tissue")


def figure2():
    """Sex composition of each tissue, with flags for insufficient / imbalanced tissues."""
    b = pd.read_csv(RESULTS / "sex_balance.csv", index_col=0).sort_values("n_total")
    y = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(b) + 1.5))

    ax.barh(y, b["n_female"], color=COLORS["female"], label="Female")
    ax.barh(y, b["n_male"], left=b["n_female"], color=COLORS["male"], label="Male")
    ax.barh(y, b["n_unknown"], left=b["n_female"] + b["n_male"],
            color=COLORS["unknown"], label="Unknown / excluded")

    for i, (_, r) in enumerate(b.iterrows()):
        if r["n_sexed"] > 0:
            ax.text(r["n_total"] + 0.5, i, f"{r['female_frac']:.0%} F", va="center", fontsize=7)

    def label(name, r):
        mark = ("†" if r["insufficient"] else "") + ("*" if r["imbalanced_sig"] else "")
        return f"{name}{mark}"

    ax.set_yticks(y)
    ax.set_yticklabels([label(n, r) for n, r in b.iterrows()])
    for tick, specific in zip(ax.get_yticklabels(), b["sex_specific"]):
        if specific:
            tick.set_fontstyle("italic")

    ax.set_xlabel("Number of cell lines")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(0.01, -0.02,
             f"† fewer than {MIN_PER_SEX} lines of one sex   * significant imbalance (BH q < 0.05)   "
             "italics: sex-specific tissue", fontsize=7)
    save(fig, "fig2_sex_balance_by_tissue")


if __name__ == "__main__":
    set_style()
    figure1()
    figure2()