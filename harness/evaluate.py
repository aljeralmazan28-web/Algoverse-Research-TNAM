#!/usr/bin/env python3
"""TNAM frozen split evaluation: per-sex metrics and tissue-matched H2 inference.

The program NEVER trains a model or invents predictions. Ground truth and metadata
come from the frozen split CSVs. Test-prediction CSVs are joined using sample_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SPLIT_NAMES = ("unseen_cell_lines", "unseen_drugs", "unseen_tissues")
GROUP_COLUMNS = {"unseen_cell_lines": "cell_line_id", "unseen_drugs": "drug_id", "unseen_tissues": "tissue"}
SEXES = ("Male", "Female")
REQUIRED = ("sample_id", "cell_line_id", "drug_id", "tissue", "sex", "response")
OUTPUT_METRICS = (
    "model", "split_design", "sex", "n_observations", "n_cell_lines", "n_drugs", "n_tissues",
    "rmse", "pearson", "spearman", "rmse_ci_low", "rmse_ci_high",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": str, "cell_line_id": str, "drug_id": str},
                       low_memory=False, keep_default_na=False)


def normalize_sex(v: str) -> str:
    s = str(v).strip().lower()
    return {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}.get(s, "Unknown")


def validate_and_load_splits(root: Path):
    result = {}
    summary = []
    for split_name in SPLIT_NAMES:
        partitions = {}
        for part in ("train", "val", "test"):
            file = root / split_name / f"{part}.csv"
            if not file.is_file():
                raise FileNotFoundError(f"Missing frozen split: {file}")
            df = read_csv(file)
            missing = set(REQUIRED) - set(df.columns)
            if missing:
                raise ValueError(f"{file}: missing columns {sorted(missing)}")
            if df["sample_id"].duplicated().any():
                raise ValueError(f"{file}: duplicate sample_id")
            if "split" in df and not (df["split"] == part).all():
                raise ValueError(f"{file}: split column does not match filename")
            if (df["sample_id"] == "").any() or (df["cell_line_id"] == "").any():
                raise ValueError(f"{file}: blank sample or cell-line identifier")
            y = pd.to_numeric(df["response"], errors="coerce")
            if not np.isfinite(y).all():
                raise ValueError(f"{file}: invalid observed response values")
            df["response"] = y.to_numpy(float)
            df["sex"] = df["sex"].map(normalize_sex)
            partitions[part] = df
            summary.append({"split_design": split_name, "partition": part,
                            "n_observations": len(df), "n_lines": df["cell_line_id"].nunique(),
                            "n_drugs": df["drug_id"].nunique(), "n_tissues": df["tissue"].nunique(),
                            "file_sha256": sha256_file(file)})
        group = GROUP_COLUMNS[split_name]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(partitions[a][group]).intersection(partitions[b][group])
            if overlap:
                raise ValueError(f"{split_name}: {len(overlap)} {group} groups overlap between {a} and {b}")
            overlap_ids = set(partitions[a]["sample_id"]).intersection(partitions[b]["sample_id"])
            if overlap_ids:
                raise ValueError(f"{split_name}: {len(overlap_ids)} sample IDs duplicated across {a}/{b}")
        # A given cell line must map to one tissue and one sex across all partitions.
        metadata = pd.concat([partitions[p][["cell_line_id", "tissue", "sex"]]
                              for p in ("train", "val", "test")], ignore_index=True)
        if (metadata.groupby("cell_line_id")["tissue"].nunique() > 1).any():
            raise ValueError(f"{split_name}: cell line mapped to multiple tissues")
        if (metadata.groupby("cell_line_id")["sex"].nunique() > 1).any():
            raise ValueError(f"{split_name}: cell line mapped to multiple sexes")
        result[split_name] = partitions
    return result, summary


def apply_corrected_sex(df: pd.DataFrame, labels_path: Path | None):
    if labels_path is None:
        return df.copy(), "reported_sex_from_uploaded_zip", 0
    labels = read_csv(labels_path)
    if "cell_line_id" not in labels.columns:
        raise ValueError("Corrected-sex CSV must have cell_line_id column")
    sex_col = next((c for c in ("corrected_sex", "final_sex", "sex") if c in labels.columns), None)
    if sex_col is None:
        raise ValueError("Corrected-sex CSV needs corrected_sex, final_sex, or sex")
    if labels["cell_line_id"].duplicated().any():
        raise ValueError("Corrected-sex CSV contains duplicate cell_line_id")
    mapped = dict(zip(labels["cell_line_id"], labels[sex_col].map(normalize_sex)))
    result = df.copy()
    result["sex"] = result["cell_line_id"].map(mapped).fillna("Unknown")
    excluded = int((~result["sex"].isin(SEXES)).sum())
    return result, "corrected_sex_from_" + labels_path.name, excluded


def find_prediction_files(pred_dir: Path):
    if not pred_dir.is_dir():
        return []
    records = []
    for path in sorted(pred_dir.glob("*.csv")):
        matched = False
        for split in SPLIT_NAMES:
            suffix = "__" + split
            if path.stem.endswith(suffix):
                model = path.stem[:-len(suffix)]
                if not model:
                    raise ValueError(f"Missing model name in {path.name}")
                records.append((model, split, path))
                matched = True
                break
        if not matched:
            raise ValueError(f"Prediction filename must end with __<split>.csv: {path.name}")
    return records


def read_prediction_and_join(path: Path, test: pd.DataFrame, allow_partial: bool):
    pred = read_csv(path)
    pred_col = next((c for c in ("y_pred", "predicted_response", "prediction", "pred")
                     if c in pred.columns), None)
    if pred_col is None:
        raise ValueError(f"{path.name}: prediction column y_pred not found")
    if "sample_id" not in pred.columns:
        if {"cell_line_id", "drug_id"}.issubset(pred.columns):
            key = test[["cell_line_id", "drug_id", "sample_id"]]
            if key.duplicated(["cell_line_id", "drug_id"]).any():
                raise ValueError("Cell line + drug is not unique; require sample_id")
            pred = pred.merge(key, on=["cell_line_id", "drug_id"], how="left", validate="many_to_one")
        else:
            raise ValueError(f"{path.name}: need sample_id, or cell_line_id and drug_id")
    if pred["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample_id in predictions")
    if pred["sample_id"].isna().any():
        raise ValueError(f"{path.name}: unmatched cell_line_id/drug_id pair")
    if not set(pred["sample_id"]).issubset(set(test["sample_id"])):
        unknown = len(set(pred["sample_id"]) - set(test["sample_id"]))
        raise ValueError(f"{path.name}: {unknown} predictions refer to non-test observations")
    pred["y_pred"] = pd.to_numeric(pred[pred_col], errors="coerce")
    if not np.isfinite(pred["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: missing, nonnumeric, or infinite predictions")
    columns = ["sample_id", "y_pred"]
    true_col = next((c for c in ("y_true", "actual", "true_response") if c in pred.columns), None)
    if true_col:
        columns.append(true_col)
    joined = test.merge(pred[columns], on="sample_id", how="left", validate="one_to_one")
    missing = int(joined["y_pred"].isna().sum())
    if missing and not allow_partial:
        raise ValueError(f"{path.name}: {missing}/{len(test)} held-out test responses have no prediction. "
                         "Provide all test predictions or use --allow-partial for exploratory analysis")
    if true_col:
        with_truth = joined[joined[true_col].notna()]
        provided = pd.to_numeric(with_truth[true_col], errors="coerce")
        if not np.isfinite(provided.to_numpy(dtype=float)).all() or not np.allclose(
                provided.to_numpy(dtype=float), with_truth["response"].to_numpy(dtype=float),
                rtol=0, atol=1e-5):
            raise ValueError(f"{path.name}: supplied y_true differs from frozen response. "
                             "Check response scaling and split compatibility")
    joined = joined.loc[joined["y_pred"].notna()].copy()
    return joined, missing


def correlations(df: pd.DataFrame):
    if len(df) < 3:
        return np.nan, np.nan
    yt = df["response"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan, np.nan
    p = float(pearsonr(yt, yp).statistic)
    s = float(spearmanr(yt, yp).statistic)
    return p, s


def one_sex_metrics(df: pd.DataFrame, sex: str):
    s = df[df["sex"] == sex]
    residual = s["y_pred"].to_numpy(dtype=float) - s["response"].to_numpy(dtype=float)
    mse = float(np.mean(residual**2)) if len(residual) else np.nan
    pearson, spearman = correlations(s)
    return {"n_observations": len(s), "n_cell_lines": s["cell_line_id"].nunique(),
            "n_drugs": s["drug_id"].nunique(), "n_tissues": s["tissue"].nunique(),
            "rmse": float(np.sqrt(mse)), "pearson": pearson, "spearman": spearman}


def two_way_cluster_bootstrap(df: pd.DataFrame, repetitions: int, seed: int):
    """Two-way nonparametric product bootstrap: resample line clusters (stratified by
    sex) and drug clusters. CIs refer to the evaluated test cohort, not patients.
    """
    rng = np.random.default_rng(seed)
    line_codes, line_ids = pd.factorize(df["cell_line_id"], sort=True)
    drug_codes, drug_ids = pd.factorize(df["drug_id"], sort=True)
    line_sexes = df.drop_duplicates("cell_line_id").set_index("cell_line_id")["sex"].reindex(line_ids)
    male_line_ids = np.flatnonzero(line_sexes.to_numpy() == "Male")
    female_line_ids = np.flatnonzero(line_sexes.to_numpy() == "Female")
    if not len(male_line_ids) or not len(female_line_ids):
        return np.full((repetitions, 3), np.nan)
    sexes = df["sex"].to_numpy()
    error2 = (df["y_pred"].to_numpy(float) - df["response"].to_numpy(float))**2
    out = np.full((repetitions, 3), np.nan)
    for r in range(repetitions):
        counts = np.zeros(len(line_ids), dtype=int)
        counts[male_line_ids] = rng.multinomial(len(male_line_ids),
                                                  np.full(len(male_line_ids), 1 / len(male_line_ids)))
        counts[female_line_ids] = rng.multinomial(len(female_line_ids),
                                                    np.full(len(female_line_ids), 1 / len(female_line_ids)))
        dc = rng.multinomial(len(drug_ids), np.full(len(drug_ids), 1 / len(drug_ids)))
        w = counts[line_codes] * dc[drug_codes]
        rm = []
        for sex in SEXES:
            mask = sexes == sex
            denom = float(np.sum(w[mask]))
            rm.append(float(np.sqrt(np.sum(w[mask] * error2[mask]) / denom)) if denom else np.nan)
        out[r] = (rm[0], rm[1], rm[1]-rm[0])
    return out


def interval(arr, confidence: float):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, np.nan
    a = (1 - confidence) / 2
    lo, hi = np.quantile(arr, [a, 1-a])
    return float(lo), float(hi)


def tissue_adjusted_test(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int,
                         confidence: float, min_each_sex: int = 2):
    """Sex gap standardized to same set of tissues. Each tissue contributes equally,
    and each line contributes its mean squared response error equally. Cell-line sex
    labels are shuffled only WITHIN tissue; preserve one label per whole cell line.
    """
    d = df[df["sex"].isin(SEXES)].copy()
    d["sq_error"] = (d["response"] - d["y_pred"])**2
    per_line = d.groupby("cell_line_id", sort=True).agg(
        tissue=("tissue", "first"), sex=("sex", "first"), mse=("sq_error", "mean"),
        n_observations=("sq_error", "size")
    ).reset_index()
    # Ensure cell-level sex/tissue metadata is internally consistent.
    if ((d.groupby("cell_line_id")["sex"].nunique() > 1).any() or
        (d.groupby("cell_line_id")["tissue"].nunique() > 1).any()):
        raise ValueError("A single cell line has inconsistent sex or tissue labels")
    tab = per_line.groupby(["tissue", "sex"]).size().unstack(fill_value=0)
    for s in SEXES:
        if s not in tab:
            tab[s] = 0
    tissues = sorted(tab.index[(tab["Male"] >= min_each_sex) &
                               (tab["Female"] >= min_each_sex)].tolist())
    eligible = per_line[per_line["tissue"].isin(tissues)].copy()
    common = {t: {s: eligible.loc[(eligible["tissue"] == t) &
                                  (eligible["sex"] == s), "mse"].to_numpy(float)
                      for s in SEXES} for t in tissues}
    summary = {"tissues_with_both_sexes": len(tissues),
               "matched_tissues": "; ".join(tissues),
               "n_tissue_matched_cell_lines": len(eligible),
               "n_tissue_matched_observations": int(d[d["tissue"].isin(tissues)].shape[0]),
               "tissue_adjusted_rmse_male": np.nan,
               "tissue_adjusted_rmse_female": np.nan,
               "tissue_adjusted_rmse_gap_female_minus_male": np.nan,
               "tissue_adjusted_gap_ci_low": np.nan,
               "tissue_adjusted_gap_ci_high": np.nan,
               "within_tissue_permutation_p_two_sided": np.nan,
               "permutation_count": n_perm}
    if not tissues:
        summary["test_status"] = "NOT_TESTABLE_NO_MIXED_SEX_TISSUES"
        return summary

    def tissue_standardized_gap(m_by_t, f_by_t):
        # sqrt of mean of cell-line MSE, first averaging within tissue, then
        # equal-weighting the tissue categories for both sexes.
        mr = float(np.sqrt(np.mean(m_by_t)))
        fr = float(np.sqrt(np.mean(f_by_t)))
        return mr, fr, fr-mr

    original = tissue_standardized_gap([np.mean(common[t]["Male"]) for t in tissues],
                                       [np.mean(common[t]["Female"]) for t in tissues])
    summary["tissue_adjusted_rmse_male"] = original[0]
    summary["tissue_adjusted_rmse_female"] = original[1]
    summary["tissue_adjusted_rmse_gap_female_minus_male"] = original[2]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        m = [np.mean(rng.choice(common[t]["Male"], size=len(common[t]["Male"]), replace=True))
             for t in tissues]
        f = [np.mean(rng.choice(common[t]["Female"], size=len(common[t]["Female"]), replace=True))
             for t in tissues]
        boot[i] = tissue_standardized_gap(m, f)[2]
    summary["tissue_adjusted_gap_ci_low"], summary["tissue_adjusted_gap_ci_high"] = interval(boot, confidence)

    count_extreme = 0
    for i in range(n_perm):
        m = []
        f = []
        for t in tissues:
            ms = common[t]["Male"]
            fs = common[t]["Female"]
            all_mse = np.concatenate((ms, fs))
            shuffled = rng.permutation(all_mse)
            m.append(np.mean(shuffled[:len(ms)]))
            f.append(np.mean(shuffled[len(ms):]))
        gap = tissue_standardized_gap(m, f)[2]
        count_extreme += bool(abs(gap) >= abs(original[2]) - 1e-12)
    summary["within_tissue_permutation_p_two_sided"] = (count_extreme+1)/(n_perm+1)
    summary["test_status"] = "OK" if len(tissues) >= 3 else "LOW_TISSUE_COUNT"
    return summary


def benjamini_hochberg(rows, field="within_tissue_permutation_p_two_sided"):
    valid = [(i, float(row[field])) for i, row in enumerate(rows)
             if pd.notna(row.get(field)) and np.isfinite(row[field])]
    if not valid:
        return
    ordered = sorted(valid, key=lambda p: p[1])
    adjusted = np.zeros(len(ordered))
    current = 1.0
    for rank0 in reversed(range(len(ordered))):
        p = ordered[rank0][1]
        current = min(current, p*len(ordered)/(rank0+1))
        adjusted[rank0] = current
    for (orig_index, _), q in zip(ordered, adjusted):
        rows[orig_index]["p_fdr_bh"] = float(q)


def write_results(out_dir: Path, inventory, per_sex, gaps, freeze_check, warnings, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory, columns=["file", "model", "split_design", "status", "detail"]).to_csv(
        out_dir / "prediction_inventory.csv", index=False)
    pd.DataFrame(per_sex, columns=OUTPUT_METRICS).to_csv(out_dir / "per_sex_metrics.csv", index=False)
    gap_cols = ["model", "split_design", "sex_label_source", "n_test_observations", "n_predicted",
                "n_missing", "n_excluded_unknown_sex", "male_rmse", "female_rmse",
                "pooled_rmse_gap_female_minus_male", "pooled_gap_ci_low", "pooled_gap_ci_high",
                "male_pearson", "female_pearson", "pearson_gap_female_minus_male",
                "male_spearman", "female_spearman", "spearman_gap_female_minus_male",
                "tissues_with_both_sexes", "matched_tissues", "n_tissue_matched_cell_lines",
                "n_tissue_matched_observations", "tissue_adjusted_rmse_male",
                "tissue_adjusted_rmse_female", "tissue_adjusted_rmse_gap_female_minus_male",
                "tissue_adjusted_gap_ci_low", "tissue_adjusted_gap_ci_high",
                "within_tissue_permutation_p_two_sided", "p_fdr_bh", "permutation_count",
                "test_status", "training_split_verification", "analysis_status"]
    pd.DataFrame(gaps, columns=gap_cols).to_csv(out_dir / "h2_performance_gap_table.csv", index=False)
    pd.DataFrame(freeze_check).to_csv(out_dir / "frozen_split_checksums.csv", index=False)
    pd.DataFrame(warnings, columns=["model", "split_design", "warning"]).to_csv(
        out_dir / "analysis_warnings.csv", index=False)
    (out_dir / "run_summary.json").write_text(json.dumps(config, indent=2, allow_nan=False))


def parse_model_manifest(path: Path | None):
    if path is None:
        return {}
    obj = json.loads(path.read_text())
    return {run["prediction_file"]: run for run in obj.get("runs", [])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "splits")
    p.add_argument("--predictions-dir", type=Path, default=Path(__file__).parent / "predictions")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--corrected-sex-labels", type=Path, default=None)
    p.add_argument("--model-runs", type=Path, default=None,
                   help="Optional Aljer training-split hash manifest; see README")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--allow-partial", action="store_true", help="Exploratory only; incomplete test predictions")
    args = p.parse_args(argv)
    if args.bootstrap < 30 or args.permutations < 99 or not 0 < args.confidence < 1:
        p.error("Require --bootstrap >= 30, --permutations >= 99, and 0 < --confidence < 1")

    partitions, check = validate_and_load_splits(args.splits_dir)
    runs = parse_model_manifest(args.model_runs)
    inventory = []
    sexrows = []
    gaps = []
    warnings = []
    prediction_files = find_prediction_files(args.predictions_dir)
    seen_files = set()
    for model, split_name, path in prediction_files:
        seen_files.add((model, split_name))
        try:
            test = partitions[split_name]["test"]
            evaluated, missing = read_prediction_and_join(path, test, args.allow_partial)
            evaluated, sex_source, excluded = apply_corrected_sex(evaluated, args.corrected_sex_labels)
            evaluated = evaluated[evaluated["sex"].isin(SEXES)].copy()
            if not set(evaluated["sex"]) == set(SEXES):
                raise ValueError("Both Male and Female lines must be present among valid predictions")
            if evaluated["cell_line_id"].nunique() < 4:
                raise ValueError("At least four distinct cell lines are required")
            if not len(evaluated):
                raise ValueError("No valid observations to score")

            run = runs.get(path.name, {})
            train_expected = sha256_file(args.splits_dir / split_name / "train.csv")
            val_expected = sha256_file(args.splits_dir / split_name / "val.csv")
            provided_train = run.get("train_sha256")
            provided_val = run.get("val_sha256")
            if provided_train or provided_val:
                if provided_train != train_expected or provided_val != val_expected:
                    raise ValueError(f"{path.name}: model-reported training/validation hashes do not match frozen files")
                training_status = "MATCHED_MODEL_DECLARED_HASHES"
            else:
                training_status = "UNVERIFIED_NO_MODEL_TRAINING_HASHES"
                warnings.append({"model": model, "split_design": split_name,
                                 "warning": "Baseline training partition is unverified; metrics are provisional"})

            ms = one_sex_metrics(evaluated, "Male")
            fs = one_sex_metrics(evaluated, "Female")
            b = two_way_cluster_bootstrap(evaluated, args.bootstrap,
                                          args.seed+101*len(gaps)+1)
            cis = [interval(b[:, j], args.confidence) for j in range(3)]
            for sex, m, ci in (("Male", ms, cis[0]), ("Female", fs, cis[1])):
                sexrows.append({"model": model, "split_design": split_name, "sex": sex,
                                **m, "rmse_ci_low": ci[0], "rmse_ci_high": ci[1]})
            adjusted = tissue_adjusted_test(evaluated, args.bootstrap, args.permutations,
                                            args.seed+101*len(gaps)+7, args.confidence)
            if adjusted["test_status"] != "OK":
                warnings.append({"model": model, "split_design": split_name,
                                 "warning": "Within-tissue comparison has "
                                 f"{adjusted['tissues_with_both_sexes']} qualifying tissue(s); "
                                 "interpret significance cautiously"})
            if args.corrected_sex_labels is None:
                warnings.append({"model": model, "split_design": split_name,
                                 "warning": "Uses reported donor sex in ZIP, not verified Natasha corrected labels"})
            if missing:
                warnings.append({"model": model, "split_design": split_name,
                                 "warning": f"Missing {missing} test predictions: EXPERIMENTAL/PARTIAL only"})
            gaps.append({"model": model, "split_design": split_name,
                         "sex_label_source": sex_source, "n_test_observations": len(test),
                         "n_predicted": len(evaluated), "n_missing": missing,
                         "n_excluded_unknown_sex": excluded,
                         "male_rmse": ms["rmse"], "female_rmse": fs["rmse"],
                         "pooled_rmse_gap_female_minus_male": fs["rmse"]-ms["rmse"],
                         "pooled_gap_ci_low": cis[2][0], "pooled_gap_ci_high": cis[2][1],
                         "male_pearson": ms["pearson"], "female_pearson": fs["pearson"],
                         "pearson_gap_female_minus_male": fs["pearson"]-ms["pearson"],
                         "male_spearman": ms["spearman"], "female_spearman": fs["spearman"],
                         "spearman_gap_female_minus_male": fs["spearman"]-ms["spearman"],
                         **adjusted, "training_split_verification": training_status,
                         "analysis_status": "PROVISIONAL" if (missing or training_status.startswith("UNVERIFIED")
                                                              or args.corrected_sex_labels is None) else "CHECKED"})
            inventory.append({"file": path.name, "model": model, "split_design": split_name,
                              "status": "EVALUATED", "detail": f"Scored {len(evaluated)}/{len(test)} test rows"})
            print(f"EVALUATED {path.name}: {len(evaluated):,} observations; "
                  f"pooled gap={fs['rmse']-ms['rmse']:.4f}; "
                  f"tissue-adjusted gap={adjusted['tissue_adjusted_rmse_gap_female_minus_male']:.4f}")
        except (ValueError, KeyError) as e:
            inventory.append({"file": path.name, "model": model, "split_design": split_name,
                              "status": "ERROR", "detail": str(e)})
            warnings.append({"model": model, "split_design": split_name,
                             "warning": f"Prediction not evaluated: {e}"})
            print(f"ERROR {path.name}: {e}", file=sys.stderr)

    # Report absent expected baseline files without pretending they were evaluated.
    for model in ("expression_only", "MOLI", "DeepCDR"):
        for split_name in SPLIT_NAMES:
            if (model, split_name) not in seen_files:
                inventory.append({"file": f"{model}__{split_name}.csv", "model": model,
                                  "split_design": split_name, "status": "MISSING_PREDICTIONS",
                                  "detail": "No baseline prediction file supplied"})
    benjamini_hochberg(gaps)
    config = {"status": "EVALUATION_COMPLETE" if gaps else "WAITING_FOR_BASELINE_PREDICTIONS",
              "models_evaluated": len(gaps), "prediction_files_found": len(prediction_files),
              "bootstrap_repetitions": args.bootstrap, "permutation_repetitions": args.permutations,
              "confidence": args.confidence, "random_seed": args.seed,
              "point_estimate": "Female RMSE minus Male RMSE; positive means higher female error",
              "tissue_adjusted_test": "Equal-weight tissues with >=2 lines per sex; each line contributes average squared error; "
                                      "sex labels permuted at line level within tissue",
              "pooled_ci": "Two-way cell-line x drug product bootstrap, resampling cell lines separately by sex",
              "adjusted_ci": "Stratified cell-line bootstrap within sex and tissue",
              "h2_p_value": "Tissue-matched within-tissue sex-label permutation (not pooled gap); BH-FDR across tested runs",
              "source": "GDSC2-only frozen test response, originally with Cellosaurus reported donor sex",
              "limitations": ["No clinical claims", "No actual H2 result without model predictions",
                              "Model training split provenance must be verified",
                              "Within-tissue adjustment does not remove other confounders",
                              "Unseen-tissues inference has only three held-out tissues; CIs conditional on those tissues",
                              "Bootstrap intervals for tissue-adjusted gap do not model dependence across drugs"]}
    write_results(args.output_dir, inventory, sexrows, gaps, check, warnings, config)
    print(f"Wrote results under: {args.output_dir}")
    if not gaps:
        print("No real model predictions evaluated. Send Aljer's test predictions to populate the H2 table.")
    return 1 if any(x["status"] == "ERROR" for x in inventory) else 0


if __name__ == "__main__":
    sys.exit(main())
