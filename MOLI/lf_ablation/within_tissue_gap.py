#!/usr/bin/env python3
"""Within-tissue female-vs-male RMSE gap, base vs sex-aware variants.

Why: a pooled F/M gap can move just because the tissue mix of F and M lines differs.
This compares F and M only inside the same tissue, then averages across tissues.

Inputs
  --run NAME=path/to/predictions.csv   (repeat; one per variant, same test rows)
  --base NAME                          which run is the baseline
  --labels sex_labels.csv              cell line -> sex (e.g. corrected_sex_labels.csv)
Prediction CSV needs: cell line id, y_true, y_pred; drug id and tissue are used if present.
Column names are auto-detected (see ALIASES); override with --line-col etc.

Run once per seed / proteomics setting, then compare the outputs.

Example
  /usr/bin/python3 within_tissue_gap.py \
      --run base=preds_base_s42.csv --run abc=preds_abc_s42.csv --run b=preds_b_s42.csv \
      --base base --labels corrected_sex_labels.csv --sex-col final_sex --out wt_s42

Outputs (in --out): per_tissue.csv, summary.csv
Gap = RMSE_female - RMSE_male. "Shrinks" means |gap| gets smaller.
Bootstrap resamples cell lines (clusters) within each tissue x sex stratum; the same
resample is applied to every run so the base-vs-variant change is paired.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "line": ["cell_line", "cell_line_name", "cell_line_id", "line", "depmap_id",
             "model_id", "sample", "sample_id", "ccl_name", "cosmic_id"],
    "drug": ["drug", "drug_name", "drug_id", "compound"],
    "tissue": ["tissue", "tissue_type", "lineage", "primary_site", "oncotree_lineage", "site"],
    "y": ["y_true", "true", "actual", "ln_ic50", "ic50", "target", "label", "y"],
    "pred": ["y_pred", "pred", "prediction", "predicted", "yhat"],
    "sex": ["final_sex", "sex", "sex_final", "corrected_sex", "inferred_sex"],
}


def pick(df, key, override=None, required=True, what=""):
    if override:
        if override not in df.columns:
            raise SystemExit(f"{what}: column '{override}' not in {list(df.columns)}")
        return override
    low = {c.lower(): c for c in df.columns}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    if required:
        raise SystemExit(f"{what}: could not find a '{key}' column in {list(df.columns)}. "
                         f"Pass --{key}-col.")
    return None


def norm_sex(s):
    s = str(s).strip().lower()
    if s in ("f", "female", "woman", "xx"):
        return "F"
    if s in ("m", "male", "man", "xy"):
        return "M"
    return None


def load_run(path, a):
    df = pd.read_csv(path)
    c_line = pick(df, "line", a.line_col, what=path)
    c_y = pick(df, "y", a.y_col, what=path)
    c_p = pick(df, "pred", a.pred_col, what=path)
    c_drug = pick(df, "drug", a.drug_col, required=False, what=path)
    c_t = pick(df, "tissue", a.tissue_col, required=False, what=path)
    out = pd.DataFrame({"line": df[c_line].astype(str).str.strip(),
                        "y": df[c_y].astype(float), "pred": df[c_p].astype(float)})
    out["drug"] = df[c_drug].astype(str).str.strip() if c_drug else (
        out.groupby("line").cumcount().astype(str))
    if c_t:
        out["tissue"] = df[c_t].astype(str)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=predictions.csv")
    ap.add_argument("--base", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="within_tissue_out")
    ap.add_argument("--min-lines", type=int, default=2,
                    help="min cell lines of EACH sex for a tissue to be included (default 2)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.01, help="RMSE tolerance for 'not hurting accuracy'")
    for k in ("line", "drug", "tissue", "y", "pred", "sex"):
        ap.add_argument(f"--{k}-col", default=None)
    ap.add_argument("--label-line-col", default=None, help="cell-line column in the labels file")
    a = ap.parse_args()

    runs = {}
    for r in a.run:
        name, _, path = r.partition("=")
        if not path:
            raise SystemExit(f"--run must look like NAME=path, got {r}")
        runs[name] = load_run(path, a)
    if a.base not in runs:
        raise SystemExit(f"--base {a.base} is not one of {list(runs)}")
    names = [a.base] + [n for n in runs if n != a.base]

    # labels
    lab = pd.read_csv(a.labels)
    l_line = pick(lab, "line", a.label_line_col or a.line_col, what=a.labels)
    l_sex = pick(lab, "sex", a.sex_col, what=a.labels)
    l_t = pick(lab, "tissue", a.tissue_col, required=False, what=a.labels)
    L = pd.DataFrame({"line": lab[l_line].astype(str).str.strip(),
                      "sex": lab[l_sex].map(norm_sex)})
    if l_t:
        L["tissue_lab"] = lab[l_t].astype(str)
    L = L.dropna(subset=["sex"]).drop_duplicates("line")

    # align runs on (line, drug)
    for n in names:
        runs[n] = runs[n].drop_duplicates(["line", "drug"]).set_index(["line", "drug"])
    common = runs[names[0]].index
    for n in names[1:]:
        common = common.intersection(runs[n].index)
    print(f"rows common to all runs: {len(common)} "
          f"(per run: {', '.join(f'{n}={len(runs[n])}' for n in names)})")
    if len(common) == 0:
        raise SystemExit("no overlapping (line, drug) rows across runs")
    D = runs[names[0]].loc[common, ["y"] + (["tissue"] if "tissue" in runs[names[0]] else [])].copy()
    for n in names:
        D[f"se_{n}"] = (runs[n].loc[common, "pred"] - runs[n].loc[common, "y"]) ** 2
    D = D.reset_index().merge(L, on="line", how="inner")
    n_drop = len(common) - len(D)
    print(f"rows dropped for missing/unknown sex: {n_drop} ({n_drop / len(common):.1%})")
    if "tissue" not in D:
        if "tissue_lab" not in D:
            raise SystemExit("no tissue column in predictions or labels; pass --tissue-col")
        D["tissue"] = D["tissue_lab"]

    # keep tissues with enough lines of each sex
    nl = D.groupby(["tissue", "sex"])["line"].nunique().unstack(fill_value=0)
    keep = nl[(nl.get("F", 0) >= a.min_lines) & (nl.get("M", 0) >= a.min_lines)].index
    D = D[D["tissue"].isin(keep)]
    if D.empty:
        raise SystemExit(f"no tissue has >= {a.min_lines} lines of each sex")
    tissues = sorted(keep)
    print(f"tissues kept: {len(tissues)} of {len(nl)}; rows used: {len(D)}")

    # cluster table: one row per (tissue, sex, line)
    se_cols = [f"se_{n}" for n in names]
    C = (D.groupby(["tissue", "sex", "line"])
           .agg(n=("y", "size"), **{c: (c, "sum") for c in se_cols}).reset_index())
    t_idx = {t: i for i, t in enumerate(tissues)}
    C["g"] = C["tissue"].map(t_idx) * 2 + (C["sex"] == "M").astype(int)  # F=0, M=1
    g = C["g"].to_numpy()
    ncl = C["n"].to_numpy(float)
    sse = C[se_cols].to_numpy(float)
    T = len(tissues)

    def stats(w, k):
        N = np.bincount(g, ncl * w, 2 * T)
        S = np.bincount(g, sse[:, k] * w, 2 * T)
        with np.errstate(invalid="ignore", divide="ignore"):
            rm = np.sqrt(S / N)
            gap_t = rm[0::2] - rm[1::2]
            pooled = np.sqrt(S[0::2].sum() / N[0::2].sum()) - np.sqrt(S[1::2].sum() / N[1::2].sum())
            overall = np.sqrt(S.sum() / N.sum())
        return (np.array([np.nanmean(gap_t), np.nanmean(np.abs(gap_t)), pooled, overall]),
                gap_t, rm[0::2], rm[1::2])

    METRICS = ["within_tissue_gap_signed", "within_tissue_gap_abs", "pooled_gap_signed", "overall_rmse"]

    # per-tissue table (point estimates)
    rows = []
    base_gap = stats(np.ones(len(C)), 0)[1]
    for k, n in enumerate(names):
        _, gap_t, rf, rmv = stats(np.ones(len(C)), k)
        for i, t in enumerate(tissues):
            sub = C[C.tissue == t]
            rows.append(dict(
                run=n, tissue=t,
                lines_F=int(((sub.sex == "F")).sum()), lines_M=int(((sub.sex == "M")).sum()),
                rows_F=int(sub.loc[sub.sex == "F", "n"].sum()), rows_M=int(sub.loc[sub.sex == "M", "n"].sum()),
                rmse_F=rf[i], rmse_M=rmv[i], gap=gap_t[i],
                abs_gap_change_vs_base=abs(gap_t[i]) - abs(base_gap[i])))
    pt = pd.DataFrame(rows)

    # paired bootstrap over cell lines, stratified by tissue x sex
    rng = np.random.default_rng(a.seed)
    strata = [np.where(g == s)[0] for s in np.unique(g)]
    boot = np.empty((a.boot, len(names), 4))
    for b in range(a.boot):
        w = np.zeros(len(C))
        for idx in strata:
            w += np.bincount(rng.choice(idx, len(idx)), minlength=len(C))
        for k in range(len(names)):
            boot[b, k] = stats(w, k)[0]
    est = np.array([stats(np.ones(len(C)), k)[0] for k in range(len(names))])

    srows = []
    for k, n in enumerate(names):
        for m, mn in enumerate(METRICS):
            r = dict(run=n, metric=mn, value=est[k, m])
            if k > 0:
                d = boot[:, k, m] - boot[:, 0, m]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                r.update(delta_vs_base=est[k, m] - est[0, m], ci_lo=lo, ci_hi=hi,
                         ci_excludes_0=bool(lo > 0 or hi < 0))
            srows.append(r)
    S = pd.DataFrame(srows)

    # H4-style verdict: |within-tissue gap| shrinks (CI < 0) and overall RMSE not worse by > tol
    print("\n=== within-tissue summary (gap = RMSE_F - RMSE_M) ===")
    for n in names[1:]:
        g_ = S[(S.run == n) & (S.metric == "within_tissue_gap_abs")].iloc[0]
        o_ = S[(S.run == n) & (S.metric == "overall_rmse")].iloc[0]
        shrink = g_.ci_hi < 0
        ok = o_.delta_vs_base <= a.tol
        verdict = "SUPPORTED" if (shrink and ok) else "not supported"
        print(f"{n:>12}: d|gap|={g_.delta_vs_base:+.4f} [{g_.ci_lo:+.4f},{g_.ci_hi:+.4f}]  "
              f"d overall RMSE={o_.delta_vs_base:+.4f} [{o_.ci_lo:+.4f},{o_.ci_hi:+.4f}]  -> {verdict}")
    b0 = S[(S.run == a.base)].set_index("metric")["value"]
    print(f"\nbase: within-tissue gap {b0['within_tissue_gap_signed']:+.4f} "
          f"(|.| {b0['within_tissue_gap_abs']:.4f}), pooled gap {b0['pooled_gap_signed']:+.4f}, "
          f"overall RMSE {b0['overall_rmse']:.4f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pt.to_csv(out / "per_tissue.csv", index=False)
    S.to_csv(out / "summary.csv", index=False)
    print(f"\nwrote {out/'per_tissue.csv'} and {out/'summary.csv'}")


if __name__ == "__main__":
    main()