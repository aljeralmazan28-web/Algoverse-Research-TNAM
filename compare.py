#Compare inferred vs reported sex; record mismatches and missing labels.
import pandas as pd
from config import PROC, RESULTS


def compare(r):
    confident = r["confidence"] == "high" and r["inferred_sex"] in ("Female", "Male")
    if not confident:
        return "uninformative"
    if r["reported_sex"] == "Unknown":
        return "missing"
    if r["reported_sex"] == "Conflict":
        return "source_conflict"
    if r["reported_sex"] == "Mixed":
        return "uninformative"
    return "concordant" if r["inferred_sex"] == r["reported_sex"] else "discordant"


def main():
    inf = pd.read_csv(PROC / "inferred_sex.csv", index_col="ModelID")
    rep = pd.read_csv(PROC / "reported_sex_main.csv").set_index("ModelID")
    rep_cols = ["CellLineName", "cvcl_final", "reported_sex", "sex_depmap", "sex_cello",
                "metadata_status", "amelogenin", "problematic"]
    df = inf.join(rep[rep_cols])

    df["status"] = df.apply(compare, axis=1)
    df.to_csv(PROC / "sex_audit.csv")

    print(df["status"].value_counts())
    print("\nInferred vs reported (high-confidence calls):")
    hc = df[df["confidence"] == "high"]
    print(pd.crosstab(hc["inferred_sex"], hc["reported_sex"]))

    out_cols = ["CellLineName", "cvcl_final", "OncotreeLineage", "reported_sex",
                "inferred_sex", "xist_score", "y_score", "y_cn", "amelogenin",
                "problematic", "metadata_status"]
    mism = df[df["status"] == "discordant"][out_cols].copy()
    mism["notes"] = ""                                   # fill in after manual review
    mism.to_csv(RESULTS / "mismatches.csv")

    missing = df[df["status"] == "missing"][out_cols]
    missing.to_csv(RESULTS / "recovered_labels.csv")

    print(f"\n{len(mism)} discordant lines -> results/mismatches.csv (review each on cellosaurus.org)")
    print(f"{len(missing)} missing labels recovered -> results/recovered_labels.csv")


if __name__ == "__main__":
    main()