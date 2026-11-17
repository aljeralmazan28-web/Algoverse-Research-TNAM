"""
ef_prep.py -- data preparation for the sex-aware early fusion runner (NO torch needed).

This is a faithful copy of the preprocessing in Malavika's early_fusion_model.ipynb
(per-layer median impute -> standardize -> truncated SVD fit on TRAINING cells only,
one-hot drug id, the same QC rules), so baselines stay comparable to EF1/EF2.

New here:
  * the X/Y marker values and the sex labels the MODEL sees are built separately from
    the TRUE labels, so the shuffled-label control never touches evaluation;
  * the shuffle assigns each cell line the sex and/or X/Y profile of ANOTHER cell line
    (a "donor") from the same tissue, so a "both" control stays internally consistent;
  * two weighting options: Natasha's reweighting.py (within-tissue) or Malavika's formula.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reweighting import apply_weights, fit_sex_tissue_weights
from sex_shuffle import shuffle_sex_labels

SEX_MARKER_GENES = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY"]
OMICS_FILENAMES = {
    "expression": "expression_aligned.csv",
    "mutation": "mutation_aligned.csv",
    "cnv": "cnv_aligned.csv",
    "proteomics": "proteomics_aligned.csv",
}
N_COMPONENTS = 32
PREP_SEED = 42          # SVD seed: fixed so every training seed sees the same features


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def find_input(input_dir, filename, split_group=None):
    candidates = list(Path(input_dir).rglob(filename))
    if split_group:
        candidates = [p for p in candidates if split_group in p.parts]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} under {input_dir}"
            f"{' / ' + split_group if split_group else ''}, found {candidates}")
    return candidates[0]


def load_omics(input_dir):
    omics = {}
    for layer, fname in OMICS_FILENAMES.items():
        df = pd.read_csv(find_input(input_dir, fname), low_memory=False)
        assert "SIDM" in df.columns, f"{fname} missing SIDM"
        if df.SIDM.duplicated().any():
            raise ValueError(f"{fname}: duplicate SIDM IDs")
        df = df.set_index("SIDM")
        df.index = df.index.astype(str)
        omics[layer] = df.astype(np.float32)
    shared = set.intersection(*(set(df.index) for df in omics.values()))
    return omics, shared


def load_labels(input_dir):
    labels = pd.read_csv(find_input(input_dir, "sex_labels_matched.csv"))
    need = {"SIDM", "final_sex", "usable_sex", "exclude_from_sex_analysis"}
    assert need.issubset(labels.columns), f"labels missing {need - set(labels.columns)}"
    labels["SIDM"] = labels["SIDM"].astype(str)
    assert labels["SIDM"].is_unique
    if "label_confidence" not in labels.columns:
        labels["label_confidence"] = np.nan
    return labels[["SIDM", "final_sex", "usable_sex", "exclude_from_sex_analysis",
                   "label_confidence"]].rename(columns={"SIDM": "cell_line_id"})


def prepare_responses(input_dir, split_family, labels, shared_ids):
    """Same QC as the notebook. Returns (data, qc_excluded, sex_conflicts, n_raw_test_rows)."""
    frames = []
    for part in ["train", "val", "test"]:
        df = pd.read_csv(find_input(input_dir, f"{part}.csv", split_family))
        assert {"sample_id", "cell_line_id", "drug_id", "tissue", "sex",
                "response", "split"}.issubset(df.columns)
        assert set(df["split"].astype(str)) == {part}, (split_family, part)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["cell_line_id"] = raw["cell_line_id"].astype(str)
    raw["drug_id"] = raw["drug_id"].astype(str)
    raw["response"] = pd.to_numeric(raw["response"], errors="coerce")
    if raw["sample_id"].duplicated().any():
        raise ValueError("sample_id duplicates across frozen splits")
    n_raw_test = int((raw["split"] == "test").sum())

    raw = raw.merge(labels, on="cell_line_id", how="left", validate="many_to_one")
    pairs = raw[["cell_line_id", "sex", "final_sex"]].drop_duplicates()
    conflicts = pairs[pairs["final_sex"].isin(["Male", "Female"]) & (pairs["sex"] != pairs["final_sex"])]
    usable = (raw["usable_sex"].eq(True) & raw["exclude_from_sex_analysis"].eq(False) &
              raw["final_sex"].isin(["Male", "Female"]) & raw["response"].notna() &
              raw["cell_line_id"].isin(shared_ids))
    qc = raw.loc[~usable, ["cell_line_id", "sex", "final_sex", "usable_sex",
                           "exclude_from_sex_analysis"]].drop_duplicates()
    data = raw.loc[usable].copy()
    data["sex_reported"] = data["sex"]
    data["sex"] = data["final_sex"]          # corrected label
    data["sex_true"] = data["final_sex"]     # evaluation ALWAYS uses this
    data.reset_index(drop=True, inplace=True)

    field = {"unseen_cell_lines": "cell_line_id", "unseen_tissues": "tissue",
             "unseen_drugs": "drug_id"}[split_family]
    sets = [set(data.loc[data.split == p, field]) for p in ["train", "val", "test"]]
    if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError(f"Leakage: {field} overlaps frozen partitions")
    if any(len(data[data.split == p]) == 0 for p in ["train", "val", "test"]):
        raise ValueError("A split is empty after QC")
    # each cell line must have exactly one tissue and one sex (shuffles work per cell line)
    if (data.groupby("cell_line_id")[["tissue", "sex_true"]].nunique().max() > 1).any():
        raise ValueError("A cell line has more than one tissue or sex")
    return data, qc, conflicts, n_raw_test


# --------------------------------------------------------------------------
# Features (fit on TRAINING cell lines only)
# --------------------------------------------------------------------------
def build_features(data, omics, seed=PREP_SEED, n_components=N_COMPONENTS):
    train = data.split.eq("train").to_numpy()
    cell_ids = pd.Index(data["cell_line_id"].unique(), name="cell_line_id")
    train_cell_ids = pd.Index(data.loc[train, "cell_line_id"].unique())
    row_indices = cell_ids.get_indexer(data["cell_line_id"])
    assert (row_indices >= 0).all()

    layer_embeddings = {}
    for layer, table in omics.items():
        # the X/Y marker genes are NOT part of the base features in any variant
        base = table.drop(columns=SEX_MARKER_GENES, errors="ignore")
        x_train = base.loc[train_cell_ids].to_numpy(dtype=np.float32)
        x_all = base.loc[cell_ids].to_numpy(dtype=np.float32)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imp.fit_transform(x_train).astype(np.float32)
        x_all = imp.transform(x_all).astype(np.float32)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train).astype(np.float32)
        x_all = sc.transform(x_all).astype(np.float32)
        k = min(n_components, len(train_cell_ids) - 1, x_train.shape[1] - 1)
        if k < 1:
            raise ValueError("Not enough training cells/features for SVD")
        svd = TruncatedSVD(n_components=k, random_state=seed, n_iter=4)
        x_train = svd.fit_transform(x_train).astype(np.float32)
        x_all = svd.transform(x_all).astype(np.float32)
        ssc = StandardScaler().fit(x_train)
        layer_embeddings[layer] = ssc.transform(x_all).astype(np.float32)

    train_drugs = set(data.loc[train, "drug_id"])
    unknown = set(data["drug_id"]) - train_drugs
    if unknown:
        raise ValueError(f"{len(unknown)} drugs absent from training; one-hot drug id is "
                         "invalid for unseen drugs")
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="error", dtype=np.float32)
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="error", dtype=np.float32)
    enc.fit(data.loc[train, ["drug_id"]])
    drug_ohe = enc.transform(data[["drug_id"]]).astype(np.float32)

    expr = omics["expression"]
    missing = [g for g in SEX_MARKER_GENES if g not in expr]
    if missing:
        raise ValueError(f"Missing expression sex marker genes: {missing}")
    raw_mod = expr.loc[cell_ids, SEX_MARKER_GENES].to_numpy(dtype=np.float32)
    tr_mod = expr.loc[train_cell_ids, SEX_MARKER_GENES].to_numpy(dtype=np.float32)
    mimp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tr_mod)
    msc = StandardScaler().fit(mimp.transform(tr_mod))
    marker_module = msc.transform(mimp.transform(raw_mod)).astype(np.float32)
    return {"cell_ids": cell_ids, "row_indices": row_indices,
            "layer_embeddings": layer_embeddings, "drug_ohe": drug_ohe,
            "marker_module": marker_module}


def assemble_base(feats, use_proteomics):
    layers = ["expression", "mutation", "cnv"] + (["proteomics"] if use_proteomics else [])
    parts = [feats["layer_embeddings"][l][feats["row_indices"]] for l in layers]
    parts.append(feats["drug_ohe"])
    return np.hstack(parts).astype(np.float32)


# --------------------------------------------------------------------------
# What the MODEL sees (true vs shuffled) -- evaluation never uses these
# --------------------------------------------------------------------------
def cell_info(data, feats):
    info = (data[["cell_line_id", "tissue", "sex_true"]]
            .drop_duplicates("cell_line_id").set_index("cell_line_id"))
    return info.loc[feats["cell_ids"]]


def donor_index(tissue_per_cell, scope, seed):
    """donor[i] = index of the cell line whose sex/X-Y profile cell line i receives."""
    n = len(tissue_per_cell)
    return shuffle_sex_labels(np.arange(n), np.asarray(tissue_per_cell),
                              within_tissue=(scope == "within_tissue"), seed=seed)


def model_inputs(data, feats, control="none", scope="within_tissue", seed=0):
    """
    control: 'none' | 'sex' | 'xy' | 'both'
      sex  -> shuffle the sex label fed to (a) sex input and (c) reweighting
      xy   -> shuffle the X/Y expression fed to (b)
      both -> shuffle both, using the SAME donor so the fake sex matches the fake X/Y
    Returns (sex_model_per_row [str], xy_per_row [float32 N x 5]).
    """
    if control not in ("none", "sex", "xy", "both"):
        raise ValueError(control)
    info = cell_info(data, feats)
    n = len(info)
    ident = np.arange(n)
    donor = donor_index(info["tissue"].values, scope, seed) if control != "none" else ident
    sex_donor = donor if control in ("sex", "both") else ident
    xy_donor = donor if control in ("xy", "both") else ident
    cell_sex_model = pd.Series(info["sex_true"].values[sex_donor], index=info.index)
    sex_rows = cell_sex_model.reindex(data["cell_line_id"]).to_numpy()
    xy_rows = feats["marker_module"][xy_donor][feats["row_indices"]].astype(np.float32)
    return sex_rows, xy_rows


# --------------------------------------------------------------------------
# Sample weights (fit on TRAINING rows only, from the sex the MODEL sees)
# --------------------------------------------------------------------------
def train_weights(data, sex_model, weighting="reweighting"):
    mask = data.split.eq("train").to_numpy()
    tr = data.loc[mask, ["cell_line_id", "tissue"]].copy()
    tr["sex"] = np.asarray(sex_model)[mask]
    if weighting == "reweighting":                      # Natasha: within-tissue, cap 10
        lines = tr.drop_duplicates("cell_line_id")
        table = fit_sex_tissue_weights(lines, sex_col="sex", tissue_col="tissue")
        w = apply_weights(tr, table, sex_col="sex", tissue_col="tissue").to_numpy(np.float32)
    elif weighting == "malavika":                       # her notebook's formula
        counts = tr.groupby(["sex", "tissue"]).size()
        cw = ((len(tr) / (len(counts) * counts)) ** 0.5).clip(lower=0.5, upper=3.0)
        w = pd.Series(pd.MultiIndex.from_frame(tr[["sex", "tissue"]]).map(cw).to_numpy(np.float32))
        w = (w.to_numpy() / w.mean()).astype(np.float32)
    else:
        raise ValueError(weighting)
    return w