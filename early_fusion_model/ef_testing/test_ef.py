import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import ef_prep as P

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())

# ---------- synthetic inputs ----------
N = 80
cells = [f"SIDM{i:05d}" for i in range(N)]
tissues = rng.choice(["Lung", "Skin", "Brain", "Breast"], N)
sex = np.where(rng.random(N) < 0.4, "Female", "Male")
sex[tissues == "Breast"] = "Female"                       # a single-sex tissue
sex[:4] = "Unknown"                                       # a few unknown-sex lines
genes = [f"g{i}" for i in range(50)] + P.SEX_MARKER_GENES
female = (sex == "Female")


def omics_table(ids, cols, with_nan=False):
    a = rng.normal(size=(len(ids), len(cols))).astype(np.float32)
    df = pd.DataFrame(a, index=ids, columns=cols)
    if with_nan:
        df.iloc[:, 0] = np.where(rng.random(len(ids)) < 0.1, np.nan, df.iloc[:, 0])
    return df


expr = omics_table(cells, genes)
expr.loc[female, "XIST"] += 4.0                           # make markers carry sex, like real data
expr.loc[~female, ["RPS4Y1", "DDX3Y", "KDM5D", "UTY"]] += 4.0
mut = omics_table(cells, [f"m{i}" for i in range(30)])
cnv = omics_table(cells, [f"c{i}" for i in range(30)])
prot_ids = cells[:70]                                      # proteomics missing for 10 cells
prot = omics_table(prot_ids, [f"p{i}" for i in range(25)], with_nan=True)
inp = tmp / "inputs"
inp.mkdir()
for name, df in [("expression", expr), ("mutation", mut), ("cnv", cnv), ("proteomics", prot)]:
    df.rename_axis("SIDM").reset_index().to_csv(inp / P.OMICS_FILENAMES[name], index=False)

labels = pd.DataFrame({"SIDM": cells, "final_sex": sex, "usable_sex": sex != "Unknown",
                       "exclude_from_sex_analysis": False, "label_confidence": "high"})
labels.loc[10:12, "exclude_from_sex_analysis"] = True     # flagged lines
labels.to_csv(inp / "sex_labels_matched.csv", index=False)

drugs = [f"D{i}" for i in range(6)]
order = rng.permutation(N)
part = {"train": order[:56], "val": order[56:68], "test": order[68:]}
d = inp / "splits" / "unseen_cell_lines"
d.mkdir(parents=True)
for p, idx in part.items():
    recs = [dict(sample_id=f"{cells[i]}__{dr}", cell_line_id=cells[i], drug_id=dr, tissue=tissues[i],
                 sex=("Male" if sex[i] == "Female" else "Female") if i % 17 == 0 else sex[i],  # a few 'reported' conflicts
                 response=float(rng.normal()), split=p) for i in idx for dr in drugs]
    pd.DataFrame(recs).to_csv(d / f"{p}.csv", index=False)

# ---------- loading + QC ----------
omics, shared = P.load_omics(inp)
assert len(shared) == 70, len(shared)
lab = P.load_labels(inp)
data, qc, conflicts, n_raw_test = P.prepare_responses(inp, "unseen_cell_lines", lab, shared)
bad = set(cells[:4]) | set(cells[10:13]) | set(cells[70:])       # unknown, flagged, no proteomics
assert not (set(data.cell_line_id) & bad)
assert set(data.sex) <= {"Female", "Male"} and (data.sex == data.sex_true).all()
assert (data.sex != data.sex_reported).any(), "corrected labels should override some reported labels"
assert n_raw_test == 12 * 6 and (data.split == "test").sum() <= n_raw_test
print("QC OK:", len(data), "usable rows; test rows", int((data.split == 'test').sum()), "of", n_raw_test)

# ---------- features ----------
feats = P.build_features(data, omics)
n_cells = len(feats["cell_ids"])
k = {l: v.shape[1] for l, v in feats["layer_embeddings"].items()}
Xb = P.assemble_base(feats, False)
Xp = P.assemble_base(feats, True)
n_drugs = feats["drug_ohe"].shape[1]
assert Xb.shape == (len(data), k["expression"] + k["mutation"] + k["cnv"] + n_drugs)
assert Xp.shape[1] == Xb.shape[1] + k["proteomics"]
assert feats["marker_module"].shape == (n_cells, 5)
tr_ids = data.loc[data.split == "train", "cell_line_id"].unique()
tr_pos = feats["cell_ids"].get_indexer(tr_ids)
assert np.allclose(feats["marker_module"][tr_pos].mean(0), 0, atol=1e-4)      # standardized on TRAIN cells
# the marker genes must NOT be inside the base expression features
assert all(g not in omics["expression"].drop(columns=P.SEX_MARKER_GENES).columns for g in P.SEX_MARKER_GENES)
print("features OK: base", Xb.shape, "with proteomics", Xp.shape)

# ---------- control = none returns the TRUE sex and X/Y ----------
sx, xy = P.model_inputs(data, feats, "none")
assert (sx == data.sex_true.to_numpy()).all()
assert np.allclose(xy, feats["marker_module"][feats["row_indices"]])

# ---------- shuffled controls ----------
info = P.cell_info(data, feats)
for scope in ["within_tissue", "global"]:
    sx, xy = P.model_inputs(data, feats, "both", scope, seed=3)
    assert (data.sex_true.to_numpy() == data.sex.to_numpy()).all()         # evaluation labels untouched
    cell_model = pd.Series(sx, index=data.cell_line_id).groupby(level=0).first().loc[info.index]
    changed = (cell_model.values != info.sex_true.values).mean()
    assert changed > 0.1, (scope, changed)
    if scope == "within_tissue":
        for t in info.tissue.unique():
            m = (info.tissue == t).values
            assert Counter(cell_model.values[m]) == Counter(info.sex_true.values[m]), t
    # coherence: each cell's fake sex and fake X/Y come from the SAME donor cell line
    donor = P.donor_index(info.tissue.values, scope, 3)
    assert (cell_model.values == info.sex_true.values[donor]).all()
    xy_cell = pd.DataFrame(xy).groupby(feats["row_indices"]).first().to_numpy()
    assert np.allclose(xy_cell, feats["marker_module"][donor])
    if scope == "within_tissue":
        assert (info.tissue.values[donor] == info.tissue.values).all()     # donors stay in-tissue
print("shuffled controls OK (counts per tissue kept, evaluation labels untouched, donor-coherent)")

sx_only, xy_true = P.model_inputs(data, feats, "sex", seed=3)
assert np.allclose(xy_true, feats["marker_module"][feats["row_indices"]])  # 'sex' leaves X/Y true
sx_true, xy_shuf = P.model_inputs(data, feats, "xy", seed=3)
assert (sx_true == data.sex_true.to_numpy()).all() and not np.allclose(xy_shuf, xy_true)  # 'xy' leaves sex true
sa, _ = P.model_inputs(data, feats, "both", seed=3)
sb, _ = P.model_inputs(data, feats, "both", seed=4)
assert (sa != sb).any(), "different seeds should give different shuffles"
print("control='sex' / 'xy' leave the other input true; seeds differ OK")

# ---------- weights ----------
n_train = int((data.split == "train").sum())
sx_true, _ = P.model_inputs(data, feats, "none")
w = P.train_weights(data, sx_true, "reweighting")
assert w.shape == (n_train,) and np.isfinite(w).all() and (w > 0).all()
tr = data.loc[data.split == "train"].assign(w=w).drop_duplicates("cell_line_id")
tot = tr.groupby(["tissue", "sex"])["w"].sum().unstack()
both = tot.dropna()
assert len(both) >= 1 and np.allclose(both["Female"], both["Male"], rtol=0.02), both   # equal sex totals per tissue
wm = P.train_weights(data, sx_true, "malavika")
assert wm.shape == (n_train,) and abs(wm.mean() - 1) < 1e-4 and wm.min() > 0
sx_shuf, _ = P.model_inputs(data, feats, "sex", seed=3)
w_shuf = P.train_weights(data, sx_shuf, "reweighting")
assert not np.allclose(w, w_shuf), "weights must follow the sex the model sees"
print("weights OK: equal female/male totals within tissue; follow the shuffled sex")
print("\nALL ef_prep TESTS PASSED")