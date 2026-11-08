#Finalize mislabel results and assign a corrected sex label to every line.
import numpy as np
import pandas as pd
from config import PROC, RESULTS, Y_LOSS_CN, MIN_LINES_PER_TISSUE
from utils import pool_lineages, wilson

VALID_DECISIONS = {"accept_inferred", "keep_reported", "uncertain", "exclude"}


def load_reviews():
    """Read manual review decisions from results/mismatches.csv (if filled in)."""
    path = RESULTS / "mismatches.csv"
    if not path.exists():
        return pd.Series(dtype=str)
    m = pd.read_csv(path, index_col="ModelID")
    if "review_decision" not in m.columns:
        print("No review_decision column: using automatic rules for all discordant lines")
        return pd.Series(dtype=str)
    d = m["review_decision"].dropna().astype(str).str.strip().str.lower()
    bad = d[~d.isin(VALID_DECISIONS)]
    if len(bad):
        raise ValueError(f"Unknown review_decision values: {bad.to_dict()}")
    print(f"Loaded {len(d)} manual review decisions")
    return d


def amelogenin_support(inferred, amel):
    """Does the STR Amelogenin result agree with the expression call?"""
    if pd.isna(amel) or amel == "conflicting" or inferred not in ("Female", "Male"):
        return "none"
    if inferred == "Male":
        return "supports" if amel == "X,Y" else "contradicts"
    return "supports" if amel == "X" else "contradicts"


def assign(r):
    """Return (final_sex, label_source, label_confidence) for one line."""
    rep, inf, status, dec = r["reported_sex"], r["inferred_sex"], r["status"], r["review_decision"]
    amel = amelogenin_support(inf, r["amelogenin"])

    if dec == "exclude":
        return rep, "manual_exclude", "exclude"

    if status == "concordant":
        return rep, "reported+expression", "high"

    if status == "missing":                          # no reported label, confident call
        conf = {"supports": "high", "none": "medium", "contradicts": "low"}[amel]
        return inf, "expression_only", conf

    if status == "source_conflict":                  # DepMap vs Cellosaurus disagree
        return inf, "expression_resolved_conflict", "high" if amel == "supports" else "medium"

    if status == "discordant":
        if dec == "accept_inferred":
            return inf, "corrected_manual", "high"
        if dec == "keep_reported":
            return rep, "reported_manual", "medium"
        if dec == "uncertain":
            return rep, "reported_uncertain", "low"
        if amel == "contradicts":                    # no review + conflicting evidence
            return rep, "unresolved", "low"
        return inf, "corrected_auto", "high" if amel == "supports" else "medium"

    # status == "uninformative"
    if inf == "Ambiguous_low" and rep == "Male" and r["y_cn"] < Y_LOSS_CN:
        return "Male", "reported_Y_loss", "medium"
    if rep in ("Female", "Male"):
        return rep, "reported_only", "low"
    return "Unknown", "unresolved", "low"


def tissue_summary(g):
    n_call = g["status"].isin(["concordant", "discordant"]).sum()
    k = int(g["mislabeled"].sum())
    n_unk = int((g["reported_sex"] == "Unknown").sum())
    m_lo, m_hi = wilson(k, n_call)
    u_lo, u_hi = wilson(n_unk, len(g))
    return pd.Series({
        "n": len(g),
        "n_callable": n_call,
        "n_mislabeled": k,
        "mislabel_pct": 100 * k / n_call if n_call else np.nan,
        "mislabel_CI_low": m_lo, "mislabel_CI_high": m_hi,
        "n_reported_unknown": n_unk,
        "missing_pct": 100 * n_unk / len(g),
        "missing_CI_low": u_lo, "missing_CI_high": u_hi,
        "n_recovered": int(g["recovered"].sum()),
        "n_still_unknown": int((g["final_sex"] == "Unknown").sum()),
        "n_uncertain": int(g["uncertain"].sum()),
    })


def main():
    df = pd.read_csv(PROC / "sex_audit.csv", index_col="ModelID")
    df["review_decision"] = load_reviews().reindex(df.index)

    out = df.apply(assign, axis=1, result_type="expand")
    out.columns = ["final_sex", "label_source", "label_confidence"]
    df = df.join(out)

    df["mislabeled"] = df["label_source"].isin(["corrected_manual", "corrected_auto"])
    df["recovered"] = df["status"].eq("missing")
    df["uncertain"] = df["label_confidence"].eq("low")
    df["problematic_line"] = df["problematic"].notna()
    df["exclude_from_sex_analysis"] = (df["label_confidence"].eq("exclude")
                                       | df["problematic_line"]
                                       | df["final_sex"].eq("Unknown"))

    keep = ["CellLineName", "cvcl_final", "OncotreeLineage", "reported_sex", "inferred_sex",
            "status", "review_decision", "final_sex", "label_source", "label_confidence",
            "mislabeled", "recovered", "uncertain", "problematic_line",
            "exclude_from_sex_analysis", "xist_score", "y_score", "y_cn", "amelogenin"]
    df[keep].to_csv(PROC / "sex_labels_final.csv")

    # Per-tissue final mislabel table
    df["lineage"] = pool_lineages(df["OncotreeLineage"], MIN_LINES_PER_TISSUE)
    table = df.groupby("lineage").apply(tissue_summary).sort_values("n", ascending=False)
    table.loc["ALL"] = tissue_summary(df)
    table.round(2).to_csv(RESULTS / "mislabel_final.csv")

    df[df["mislabeled"]][keep].to_csv(RESULTS / "corrected_lines.csv")
    df[df["uncertain"]][keep].to_csv(RESULTS / "uncertain_lines.csv")

    # Summary
    print("\nLabel source:\n", df["label_source"].value_counts())
    print("\nConfidence:\n", df["label_confidence"].value_counts())
    print("\nReported vs final sex:\n", pd.crosstab(df["reported_sex"], df["final_sex"]))
    a = table.loc["ALL"]
    print(f"\nOverall: {int(a.n_mislabeled)}/{int(a.n_callable)} mislabeled "
          f"({a.mislabel_pct:.1f}%, 95% CI {a.mislabel_CI_low:.1f}-{a.mislabel_CI_high:.1f}); "
          f"{int(a.n_recovered)} missing labels recovered")
    print(f"Excluded from sex analysis: {df['exclude_from_sex_analysis'].sum()}")


if __name__ == "__main__":
    main()