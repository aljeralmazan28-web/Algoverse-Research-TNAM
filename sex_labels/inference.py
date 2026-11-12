#Infer sex from XIST and Y-gene expression.
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from config import (EXPRESSION, COPY_NUMBER, PROC, RESULTS, XIST, Y_GENES,
                    FEMALE_TISSUE, MALE_TISSUE, CONF_POSTERIOR)


def load_gene_matrix(path, genes):
    #Read only the needed gene columns; return cell lines x gene symbols.
    genes = set(genes)
    keep = lambda c: (c.split(" (")[0] in genes or c in ("ModelID", "IsDefaultEntryForModel")
                      or c.startswith("Unnamed"))
    df = pd.read_csv(path, usecols=keep)

    if "IsDefaultEntryForModel" in df.columns:               # newer releases: one row per run
        df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true"])]
        df = df.drop(columns="IsDefaultEntryForModel")
    if "ModelID" in df.columns:
        df = df.set_index("ModelID")
    else:
        df = df.set_index(df.columns[0])
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df.index.name = "ModelID"
    df.columns = [c.split(" (")[0] for c in df.columns]      # "XIST (7503)" -> "XIST"

    if not df.index.is_unique:
        print(f"WARNING: {df.index.duplicated().sum()} duplicate rows; keeping first")
        df = df[~df.index.duplicated()]
    return df


def fit_cutoff(x):
    #2-component Gaussian mixture. Returns cutoff (P=0.5) and confidence band edges.
    v = x.dropna().values.reshape(-1, 1)
    g = GaussianMixture(2, random_state=0, n_init=5).fit(v)
    means = g.means_.ravel()
    hi = int(np.argmax(means))
    grid = np.linspace(means.min(), means.max(), 4000).reshape(-1, 1)
    p_high = g.predict_proba(grid)[:, hi]
    at = lambda q: float(grid[np.argmin(np.abs(p_high - q))][0])
    return {"cutoff": at(0.5), "band_low": at(1 - CONF_POSTERIOR), "band_high": at(CONF_POSTERIOR)}


def main():
    rep = pd.read_csv(PROC / "reported_sex_main.csv", dtype={"COSMICID": str})
    expr = load_gene_matrix(EXPRESSION, [XIST] + Y_GENES)

    y_found = [g for g in Y_GENES if g in expr.columns]
    print(f"Y genes found: {y_found}")
    if XIST not in expr.columns or len(y_found) < 3:
        raise ValueError("Missing XIST or too few Y genes; check the expression file")

    # Cohort: GDSC lines with expression
    cohort = rep[rep["in_gdsc"] & rep["ModelID"].isin(expr.index)].set_index("ModelID")
    e = expr.loc[cohort.index]
    print(f"Cohort: {len(cohort)} GDSC lines with expression")

    print("\nY-gene correlations (should all be high):")
    print(e[y_found].corr().round(2))

    scores = pd.DataFrame({"xist_score": e[XIST], "y_score": e[y_found].mean(axis=1)})

    xc, yc = fit_cutoff(scores["xist_score"]), fit_cutoff(scores["y_score"])
    print(f"\nXIST: {xc}\nY:    {yc}")
    with open(RESULTS / "cutoffs.json", "w") as f:
        json.dump({"xist": xc, "y": yc, "y_genes": y_found}, f, indent=2)

    xist_hi = scores["xist_score"] > xc["cutoff"]
    y_hi = scores["y_score"] > yc["cutoff"]
    scores["inferred_sex"] = np.select(
        [xist_hi & ~y_hi, ~xist_hi & y_hi, ~xist_hi & ~y_hi],
        ["Female", "Male", "Ambiguous_low"], "Ambiguous_high")

    # High confidence = both scores outside the uncertain band
    outside = lambda s, c: (s < c["band_low"]) | (s > c["band_high"])
    scores["confidence"] = np.where(outside(scores["xist_score"], xc) &
                                    outside(scores["y_score"], yc), "high", "low")

    # Optional: Y copy number tiebreak
    scores["y_cn"] = np.nan
    if COPY_NUMBER.exists():
        cn = load_gene_matrix(COPY_NUMBER, Y_GENES)
        if len(cn.columns):
            scores["y_cn"] = cn.reindex(scores.index).mean(axis=1)
            print(f"\nY copy number from {list(cn.columns)}")
        else:
            print("\nNo Y genes in copy-number file; skipping tiebreak")

    scores["OncotreeLineage"] = cohort["OncotreeLineage"]
    scores.to_csv(PROC / "inferred_sex.csv")

    # Summary
    print("\n", pd.crosstab(scores["inferred_sex"], scores["confidence"]))

    # Validation on sex-specific tissues
    val = scores[scores["OncotreeLineage"].isin(FEMALE_TISSUE + MALE_TISSUE)]
    print("\nValidation (sex-specific tissues):")
    print(pd.crosstab(val["OncotreeLineage"], val["inferred_sex"]))

    # Sensitivity: shift both cutoffs and recount
    print("\nSensitivity to cutoff shifts:")
    for d in (-0.5, -0.25, 0, 0.25, 0.5):
        xh = scores["xist_score"] > xc["cutoff"] + d
        yh = scores["y_score"] > yc["cutoff"] + d
        print(f"  shift {d:+.2f}: {(xh & ~yh).sum()} F, {(~xh & yh).sum()} M, "
              f"{((xh & yh) | (~xh & ~yh)).sum()} ambiguous")


if __name__ == "__main__":
    main()
