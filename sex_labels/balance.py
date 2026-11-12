"""07 - Sex balance per tissue; flag tissues with too few lines of one sex (H1)."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from config import PROC, RESULTS, SEX_SPECIFIC, MIN_PER_SEX, IMBALANCE_FRAC


def main():
    df = pd.read_csv(PROC / "sex_labels_final.csv", index_col="ModelID")
    df["OncotreeLineage"] = df["OncotreeLineage"].fillna("Unknown lineage")
    use = df[~df["exclude_from_sex_analysis"]]

    rows = []
    for tissue, g_all in df.groupby("OncotreeLineage"):
        g = use[use["OncotreeLineage"] == tissue]
        n_f = int((g["final_sex"] == "Female").sum())
        n_m = int((g["final_sex"] == "Male").sum())
        hc = g[g["label_confidence"] == "high"]
        n_sexed = n_f + n_m
        female_frac = n_f / n_sexed if n_sexed else np.nan
        minority_n = min(n_f, n_m)
        sex_specific = tissue in SEX_SPECIFIC
        p = binomtest(n_f, n_sexed, 0.5).pvalue if (n_sexed and not sex_specific) else np.nan
        rows.append({
            "tissue": tissue,
            "n_total": len(g_all),
            "n_female": n_f,
            "n_male": n_m,
            "n_unknown": len(g_all) - n_sexed,          # unknown or excluded
            "n_sexed": n_sexed,
            "n_female_highconf": int((hc["final_sex"] == "Female").sum()),
            "n_male_highconf": int((hc["final_sex"] == "Male").sum()),
            "female_frac": female_frac,
            "minority_sex": "Female" if n_f < n_m else "Male" if n_m < n_f else "tie",
            "minority_n": minority_n,
            "sex_specific": sex_specific,
            "insufficient": (not sex_specific) and minority_n < MIN_PER_SEX,
            "imbalanced": (not sex_specific) and n_sexed > 0
                          and minority_n / n_sexed < IMBALANCE_FRAC,
            "binom_p": p,
        })

    bal = pd.DataFrame(rows).set_index("tissue").sort_values("n_total", ascending=False)

    # Multiple-testing correction across non-sex-specific tissues
    tested = bal["binom_p"].notna()
    bal["binom_q"] = np.nan
    if tested.any():
        bal.loc[tested, "binom_q"] = multipletests(bal.loc[tested, "binom_p"], method="fdr_bh")[1]
    bal["imbalanced_sig"] = bal["imbalanced"] & (bal["binom_q"] < 0.05)
    bal["usable_within_tissue"] = ~bal["sex_specific"] & ~bal["insufficient"]

    bal.round(4).to_csv(RESULTS / "sex_balance.csv")

    # Overall female fraction outside sex-specific tissues
    ns = use[~use["OncotreeLineage"].isin(SEX_SPECIFIC) & use["final_sex"].isin(["Female", "Male"])]
    k = int((ns["final_sex"] == "Female").sum())
    print(f"Non-sex-specific tissues: {k}/{len(ns)} female ({k / len(ns):.1%}), "
          f"binomial p vs 0.5 = {binomtest(k, len(ns), 0.5).pvalue:.3g}")

    print(f"\nUsable for within-tissue sex comparison ({bal['usable_within_tissue'].sum()}):")
    print(bal[bal["usable_within_tissue"]][["n_female", "n_male", "female_frac"]].to_string())
    print(f"\nToo few lines of one sex (< {MIN_PER_SEX}):")
    print(bal[bal["insufficient"]][["n_female", "n_male", "minority_sex"]].to_string())
    print("\nSignificantly imbalanced (q < 0.05):")
    print(bal[bal["imbalanced_sig"]][["n_female", "n_male", "female_frac", "binom_q"]].to_string())


if __name__ == "__main__":
    main()
