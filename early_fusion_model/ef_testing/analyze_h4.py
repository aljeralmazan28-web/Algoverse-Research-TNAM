"""
analyze_h4.py -- turn runs.csv files from run_early_fusion.py into the H4 table.

H4: sex-aware add-ons shrink the female/male gap WITHOUT hurting overall accuracy.
Everything is paired by seed, so each add-on is compared with the plain model (variant 'base')
trained with the same seed, split and proteomics setting.

    python analyze_h4.py out/runs.csv out_control/runs.csv --out h4_table.csv
    python analyze_h4.py out/runs.csv --split unseen_cell_lines --tolerance 0.01

For each (split, proteomics, variant) it reports:
  d_overall   mean paired change in overall RMSE vs base (negative = better)
  d_gap_abs   mean paired change in |RMSE_F - RMSE_M| vs base (negative = gap shrank)
  seeds_gap   how many seeds the gap shrank in, out of n
  95% CI      t-interval on the paired differences (needs >= 3 seeds; scipy)
  verdict     'supported' only if the gap shrank on average, the 95% CI excludes 0, AND overall RMSE got
              worse by at most --tolerance; 'inconclusive' if the CI includes 0. Read the shuffled-control
              rows too: if control_keeps_share_of_gain is near or above 1, the gain is not about sex.
For shuffled-control runs (control != none) it also reports how much of the real improvement the control
keeps: if the shuffled run shrinks the gap as much as the real one, the improvement is not about sex.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["split_family", "proteomics", "seed"]


def paired_ci(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import t
    except ImportError:
        return np.nan, np.nan
    half = t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean() - half, d.mean() + half


def compare(runs, ref_variant="base", tolerance=0.01):
    true = runs[runs["control"] == "none"]
    base = true[true["variant"] == ref_variant].set_index(KEYS)
    if base.empty:
        raise SystemExit(f"No '{ref_variant}' runs with control=none found; run the baseline too.")
    out = []
    for (split, prot, variant, control), g in runs.groupby(["split_family", "proteomics", "variant", "control"]):
        if variant == ref_variant and control == "none":
            continue
        g = g.set_index(KEYS)
        idx = g.index.intersection(base.index)
        if len(idx) == 0:
            continue
        d_over = g.loc[idx, "overall_rmse"] - base.loc[idx, "overall_rmse"]
        d_gap = g.loc[idx, "gap_abs"] - base.loc[idx, "gap_abs"]
        lo, hi = paired_ci(d_gap)
        lo_o, hi_o = paired_ci(d_over)
        row = dict(split_family=split, proteomics=prot, variant=variant, control=control, n_seeds=len(idx),
                   base_overall=base.loc[idx, "overall_rmse"].mean(), overall=g.loc[idx, "overall_rmse"].mean(),
                   d_overall=d_over.mean(), d_overall_lo=lo_o, d_overall_hi=hi_o,
                   base_gap_abs=base.loc[idx, "gap_abs"].mean(), gap_abs=g.loc[idx, "gap_abs"].mean(),
                   d_gap_abs=d_gap.mean(), d_gap_lo=lo, d_gap_hi=hi,
                   seeds_gap_shrank=int((d_gap < 0).sum()))
        ci_known = not np.isnan(hi)
        if d_gap.mean() >= 0:
            verdict = "not supported"
        elif d_over.mean() > tolerance:
            verdict = "gap shrank but overall worse"
        elif ci_known and hi >= 0:
            verdict = "inconclusive (CI includes 0)"
        else:
            verdict = "supported"
        row["verdict"] = verdict
        out.append(row)
    res = pd.DataFrame(out)
    if res.empty:
        return res
    # share of the real improvement kept by the shuffled control (same variant, split, proteomics)
    real = res[res["control"] == "none"].set_index(["split_family", "proteomics", "variant"])["d_gap_abs"]
    kept = []
    for _, r in res.iterrows():
        key = (r["split_family"], r["proteomics"], r["variant"])
        if r["control"] != "none" and key in real.index and real[key] < 0:
            kept.append(r["d_gap_abs"] / real[key])
        else:
            kept.append(np.nan)
    res["control_keeps_share_of_gain"] = kept
    return res.sort_values(["split_family", "proteomics", "control", "variant"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="one or more runs.csv files (true and control runs)")
    ap.add_argument("--out", type=Path, default=Path("h4_table.csv"))
    ap.add_argument("--split", default=None, help="only this split_family, e.g. unseen_cell_lines")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="largest overall-RMSE increase still counted as 'not hurting accuracy'")
    a = ap.parse_args()

    runs = pd.concat([pd.read_csv(p) for p in a.runs], ignore_index=True)
    runs = runs.drop_duplicates(subset=["model", "split_family", "seed"], keep="last")
    if a.split:
        runs = runs[runs["split_family"] == a.split]
    res = compare(runs, tolerance=a.tolerance)
    res.to_csv(a.out, index=False)

    show = res.copy()
    for c in ["base_overall", "overall", "d_overall", "d_overall_lo", "d_overall_hi",
              "base_gap_abs", "gap_abs", "d_gap_abs", "d_gap_lo", "d_gap_hi", "control_keeps_share_of_gain"]:
        show[c] = show[c].round(4)
    print(show[["split_family", "proteomics", "variant", "control", "n_seeds", "d_overall", "d_gap_abs",
                "d_gap_lo", "d_gap_hi", "seeds_gap_shrank", "verdict", "control_keeps_share_of_gain"]]
          .to_string(index=False))
    print(f"\nWrote {a.out}. Differences are paired by seed against variant 'base' (negative = better).")
    if (res["n_seeds"] < 3).any():
        print("WARNING: some rows have fewer than 3 seeds, so no confidence interval; treat verdicts as tentative.")


if __name__ == "__main__":
    main()