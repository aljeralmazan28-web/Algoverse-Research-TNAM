#!/usr/bin/env python3
"""Within-tissue female vs male comparison from saved early-fusion predictions (no training).

For every prediction file it computes, per tissue, the female and male RMSE and their gap (F - M),
using only tissues with at least --min-lines test cell lines of EACH sex. Sex comes from the label
file (default: the sensitivity labels), tissue and the true response come from the frozen test split,
so the prediction files only need `sample_id` and `y_pred`.

Weighting follows the harness description: every cell line contributes its average squared error,
and a tissue-sex RMSE is sqrt(mean of those line-level values). The equal-weight mean of the tissue
gaps is the "tissue-adjusted gap" and should be close to the harness's
tissue_adjusted_rmse_gap_female_minus_male. If it differs, compare the definitions before trusting either.

Usage (from early_fusion_model/ef_testing):
  /usr/bin/python3 ef_ablation/within_tissue_compare.py \
      --pred-dirs out_main out_main_prot out_control out_control_prot --min-lines 2
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEXES = ("Female", "Male")


def norm_sex(v):
    return {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}.get(str(v).strip().lower(), "Unknown")


def parse_name(path):
    stem = path.name.split("__")[0]                      # e.g. EF_a_prot_shufboth_s42
    m = re.search(r"_s(\d+)$", stem)
    seed = int(m.group(1)) if m else -1
    core = stem[len("EF_"):] if stem.startswith("EF_") else stem
    core = re.sub(r"_s\d+$", "", core)
    parts = core.split("_")
    return dict(model=stem, variant=parts[0], seed=seed,
                proteomics="prot" in parts, control="shufboth" in parts or "shuf" in core)


def tissue_table(pred, test, sex_map, min_lines):
    d = pred[["sample_id", "y_pred"]].merge(test[["sample_id", "cell_line_id", "tissue", "response"]],
                                            on="sample_id", how="inner")
    dropped = len(pred) - len(d)
    d["sex"] = d.cell_line_id.map(sex_map).fillna("Unknown")
    d["se"] = (d.response.astype(float) - d.y_pred.astype(float)) ** 2
    lines = d.groupby(["tissue", "cell_line_id", "sex"], as_index=False).agg(mse=("se", "mean"), n=("se", "size"))
    lines = lines[lines.sex.isin(SEXES)]
    g = lines.groupby(["tissue", "sex"]).agg(n_lines=("mse", "size"), rmse=("mse", lambda x: float(np.sqrt(x.mean()))))
    g = g.unstack("sex")
    out = pd.DataFrame({
        "n_female_lines": g[("n_lines", "Female")], "n_male_lines": g[("n_lines", "Male")],
        "female_rmse": g[("rmse", "Female")], "male_rmse": g[("rmse", "Male")]})
    out[["n_female_lines", "n_male_lines"]] = out[["n_female_lines", "n_male_lines"]].fillna(0).astype(int)
    out["gap_F_minus_M"] = out.female_rmse - out.male_rmse        # NaN if a sex is absent
    out["eligible"] = (out.n_female_lines >= min_lines) & (out.n_male_lines >= min_lines) & out.gap_F_minus_M.notna()
    return out.reset_index(), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dirs", nargs="+", required=True, help="e.g. out_main out_main_prot (each holds predictions/)")
    ap.add_argument("--splits-dir", default="../../gdsc2_frozen_splits/splits")
    ap.add_argument("--split", default="unseen_cell_lines")
    ap.add_argument("--labels", default="../../harness_results/corrected_sex_labels_sensitivity.csv")
    ap.add_argument("--min-lines", type=int, default=2, help="min test lines of EACH sex for a tissue to count")
    ap.add_argument("--out-dir", default="ef_ablation/within_tissue_out")
    a = ap.parse_args()

    test = pd.read_csv(Path(a.splits_dir) / a.split / "test.csv",
                       dtype={"sample_id": str, "cell_line_id": str, "drug_id": str})
    lab = pd.read_csv(a.labels, dtype={"cell_line_id": str})
    assert not lab.cell_line_id.duplicated().any(), "duplicate cell_line_id in label file"
    sex_map = lab.set_index("cell_line_id")["corrected_sex"].map(norm_sex).to_dict()

    per_tissue, adjusted = [], []
    for d in a.pred_dirs:
        files = sorted((Path(d) / "predictions").glob(f"*__{a.split}.csv"))
        if not files:
            print(f"WARNING: no prediction files in {d}/predictions", file=sys.stderr)
        for f in files:
            meta = parse_name(f)
            pred = pd.read_csv(f, dtype={"sample_id": str})
            t, dropped = tissue_table(pred, test, sex_map, a.min_lines)
            if dropped:
                print(f"WARNING: {f.name}: {dropped} prediction rows not found in test split", file=sys.stderr)
            for k, v in meta.items():
                t[k] = v
            t["source_dir"] = d
            per_tissue.append(t)
            e = t[t.eligible]
            adjusted.append(dict(**meta, source_dir=d, n_eligible_tissues=len(e),
                                 tissue_adjusted_gap=float(e.gap_F_minus_M.mean()) if len(e) else np.nan))

    pt = pd.concat(per_tissue, ignore_index=True)
    adj = pd.DataFrame(adjusted)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pt.to_csv(out / "per_tissue_per_run.csv", index=False)
    adj.to_csv(out / "tissue_adjusted_per_run.csv", index=False)

    keys = ["source_dir", "variant", "proteomics", "control", "tissue"]
    summ = (pt[pt.eligible].groupby(keys)
              .agg(n_seeds=("seed", "nunique"), n_female_lines=("n_female_lines", "first"),
                   n_male_lines=("n_male_lines", "first"), female_rmse=("female_rmse", "mean"),
                   male_rmse=("male_rmse", "mean"), gap_mean=("gap_F_minus_M", "mean"),
                   gap_sd=("gap_F_minus_M", "std")).reset_index())
    summ.to_csv(out / "per_tissue_by_variant.csv", index=False)

    adj_s = (adj.groupby(["source_dir", "variant", "proteomics", "control"])
                .agg(n_seeds=("seed", "nunique"), n_eligible_tissues=("n_eligible_tissues", "first"),
                     adjusted_gap_mean=("tissue_adjusted_gap", "mean"),
                     adjusted_gap_sd=("tissue_adjusted_gap", "std")).reset_index())
    adj_s.to_csv(out / "tissue_adjusted_by_variant.csv", index=False)

    print(f"\nmin lines per sex per tissue: {a.min_lines}")
    print("tissues with enough lines of both sexes:", sorted(pt[pt.eligible].tissue.unique()))
    print("tissues too thin:", sorted(set(pt.tissue) - set(pt[pt.eligible].tissue)))
    print("\nTissue-adjusted gap (F - M, equal-weight over eligible tissues), mean/sd over seeds:")
    print(adj_s.round(4).to_string(index=False))
    print(f"\nwrote {out}/ (per_tissue_per_run.csv, tissue_adjusted_per_run.csv, "
          f"per_tissue_by_variant.csv, tissue_adjusted_by_variant.csv)")


if __name__ == "__main__":
    main()