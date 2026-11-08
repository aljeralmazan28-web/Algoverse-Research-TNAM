#Per-tissue mislabel and missing rates (H1 table + figures).
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, binomtest
from statsmodels.stats.proportion import proportion_confint
from config import PROC, RESULTS, SEX_SPECIFIC, MIN_LINES_PER_TISSUE

STATUS_ORDER = ["concordant", "discordant", "missing", "source_conflict", "uninformative"]


def summarize(g):
    n_disc = (g["status"] == "discordant").sum()
    n_call = g["status"].isin(["concordant", "discordant"]).sum()
    lo, hi = proportion_confint(n_disc, n_call, method="wilson") if n_call else (np.nan, np.nan)
    conf = g[(g["confidence"] == "high") & g["inferred_sex"].isin(["Female", "Male"])]
    return pd.Series({
        "n": len(g),
        "reported_F": (g["reported_sex"] == "Female").sum(),
        "reported_M": (g["reported_sex"] == "Male").sum(),
        "reported_unknown": (g["reported_sex"] == "Unknown").sum(),
        "inferred_F": (g["inferred_sex"] == "Female").sum(),
        "inferred_M": (g["inferred_sex"] == "Male").sum(),
        "ambiguous": g["inferred_sex"].str.startswith("Ambiguous").sum(),
        "discordant_n": n_disc,
        "discordant_pct": 100 * n_disc / n_call if n_call else np.nan,
        "disc_CI_low": 100 * lo if n_call else np.nan,
        "disc_CI_high": 100 * hi if n_call else np.nan,
        "missing_pct": 100 * (g["reported_sex"] == "Unknown").mean(),
        "female_frac": (conf["inferred_sex"] == "Female").mean() if len(conf) else np.nan,
    })


def main():
    df = pd.read_csv(PROC / "sex_audit.csv", index_col="ModelID")
    df["OncotreeLineage"] = df["OncotreeLineage"].fillna("Unknown lineage")
    size = df.groupby("OncotreeLineage")["OncotreeLineage"].transform("size")
    df["lineage"] = df["OncotreeLineage"].where(size >= MIN_LINES_PER_TISSUE, "Other")

    # ---- H1 table ----
    h1 = df.groupby("lineage").apply(summarize).sort_values("n", ascending=False)
    h1["sex_specific"] = h1.index.isin(SEX_SPECIFIC)
    overall = summarize(df).rename("ALL")
    h1 = pd.concat([h1, overall.to_frame().T])
    h1.round(2).to_csv(RESULTS / "H1_table.csv")
    print(h1.round(1).to_string())

    # ---- Statistics ----
    calls = df[df["status"].isin(["concordant", "discordant"])]
    ct = pd.crosstab(calls["lineage"], calls["status"])
    if ct.shape[1] == 2:
        chi2, p, _, expected = chi2_contingency(ct)
        print(f"\nDiscordance differs by tissue: chi2 = {chi2:.2f}, p = {p:.3g}")
        if (expected < 5).any():
            print("  (some expected counts < 5: treat p as approximate; rely on CIs)")
    else:
        print("\nNo discordant lines among confident calls; chi-square skipped")

    ns = df[~df["lineage"].isin(SEX_SPECIFIC) & (df["confidence"] == "high")
            & df["inferred_sex"].isin(["Female", "Male"])]
    if len(ns):
        k = (ns["inferred_sex"] == "Female").sum()
        res = binomtest(k, len(ns), 0.5)
        print(f"Non-sex-specific tissues: female fraction = {k / len(ns):.3f} "
              f"(n = {len(ns)}), binomial p vs 0.5 = {res.pvalue:.3g}")

    # ---- Figure 1: XIST vs Y score ----
    with open(RESULTS / "cutoffs.json") as f:
        cut = json.load(f)
    colors = {"Female": "tab:red", "Male": "tab:blue", "Unknown": "grey",
              "Conflict": "orange", "Mixed": "purple"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for s, g in df.groupby("reported_sex"):
        ax.scatter(g["xist_score"], g["y_score"], s=12, alpha=0.6,
                   c=colors.get(s, "grey"), label=f"reported {s} (n={len(g)})")
    d = df[df["status"] == "discordant"]
    ax.scatter(d["xist_score"], d["y_score"], s=90, facecolors="none",
               edgecolors="black", linewidths=1.2, label=f"discordant (n={len(d)})")
    ax.axvline(cut["xist"]["cutoff"], ls="--", c="k", lw=0.8)
    ax.axhline(cut["y"]["cutoff"], ls="--", c="k", lw=0.8)
    ax.set_xlabel("XIST expression, log2(TPM+1)")
    ax.set_ylabel("Mean Y-gene expression, log2(TPM+1)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig1_xist_vs_y.png", dpi=300)
    plt.close()

    # ---- Figure 2: status by tissue ----
    prop = (pd.crosstab(df["lineage"], df["status"], normalize="index")
              .reindex(columns=STATUS_ORDER, fill_value=0))
    order = h1.drop(index="ALL").index
    prop = prop.loc[order]
    ax = prop.plot.barh(stacked=True, figsize=(8, 0.35 * len(prop) + 1.5),
                        color=["#4c9f70", "#d1495b", "#edae49", "#8e7dbe", "#bbbbbb"])
    ax.set_yticklabels([f"{l} (n={int(h1.loc[l, 'n'])})" for l in prop.index])
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of cell lines")
    ax.set_ylabel("")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig2_status_by_tissue.png", dpi=300)
    plt.close()

    print("\nSaved results/H1_table.csv, fig1_xist_vs_y.png, fig2_status_by_tissue.png")


if __name__ == "__main__":
    main()