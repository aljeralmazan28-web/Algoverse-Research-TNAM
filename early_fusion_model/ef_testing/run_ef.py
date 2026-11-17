"""
run_early_fusion.py -- early fusion runner with INDEPENDENT sex-aware switches and seeds.

Same data, features, MLP, optimiser and early stopping as Malavika's early_fusion_model.ipynb,
but each add-on is its own switch (uses sex_aware.py):
    a = sex as input            (SexInput, one-hot [Female, Male])
    b = X/Y gene module         (XYModule branch, or raw genes with --xy-mode concat)
    c = sex-balanced reweighting (weighted_mse + reweighting.py weights)

Variants: base, a, b, c, ab, ac, bc, abc.   Add --proteomics for the +proteomics versions.

Example (smoke test, 2 epochs):
    python run_early_fusion.py --input-dir /path/to/Early_Fusion_Model_inputs \
        --out-dir out_smoke --variants base abc --seeds 42 --max-epochs 2

Main H4 run (5 seeds, all variants, then the controls):
    python run_early_fusion.py --input-dir ... --out-dir out --variants base a b c abc \
        --seeds 42 43 44 45 46
    python run_early_fusion.py --input-dir ... --out-dir out --variants a b c abc \
        --seeds 42 43 44 45 46 --control both

Sanity check against the notebook: `--variants base --seeds 42` should land close to EF1,
and with `--proteomics` close to EF2 (same seed, same data).

Outputs (in --out-dir):
    predictions/<model>__<split>.csv   sample_id, cell_line_id, drug_id, y_pred, y_true
                                       (harness naming; evaluation labels are the TRUE ones)
    runs.csv                           one row per run (flags, seed, per-sex metrics)
    summary_by_variant.csv             mean / sd over seeds
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ef_prep as P                                              # noqa: E402
from sex_aware import (SexInput, XYModule, concat_sex_aware,     # noqa: E402
                       encode_sex, sex_aware_extra_dim, weighted_mse)

BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4


def parse_variant(v):
    if v == "base":
        return {"a": False, "b": False, "c": False}
    if not v or not set(v) <= set("abc"):
        raise ValueError(f"variant must be 'base' or letters from a,b,c: {v!r}")
    return {k: (k in v) for k in "abc"}


class SexAwareEarlyFusionMLP(nn.Module):
    """Her EarlyFusionMLP, with the add-ons attached at the fusion point."""

    def __init__(self, n_base, sex_input, xy_module, n_xy_concat=0):
        super().__init__()
        self.sex_input = sex_input
        self.xy_module = xy_module
        n_in = n_base + sex_aware_extra_dim(sex_input, xy_module) + n_xy_concat
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 1))
        self.use_xy_concat = n_xy_concat > 0

    def forward(self, x, sex, xy):
        z = concat_sex_aware(x, self.sex_input, self.xy_module, sex_feats=sex, xy_expr=xy)
        if self.use_xy_concat:
            z = torch.cat([z, xy], dim=-1)
        return self.net(z).squeeze(-1)


def batch_predict(model, tensors, device, batch_size=8192):
    model.eval()
    out = []
    n = len(tensors[0])
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb, sb, xyb = [t[i:i + batch_size].to(device) for t in tensors]
            out.append(model(xb, sb, xyb).cpu().numpy())
    return np.concatenate(out)


def train_one(X, S, XY, y, w, split, flags, xy_mode, seed, device, max_epochs, patience):
    tr = (split == "train")
    va = (split == "val")
    te = (split == "test")
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))   # noqa: E731
    Xt, St, XYt, yt = T(X), T(S), T(XY), T(y)
    tr, va, te = (torch.from_numpy(m) for m in (tr, va, te))      # bool tensors for indexing

    train_ds = TensorDataset(Xt[tr], St[tr], XYt[tr], yt[tr], T(w))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sex_input = SexInput(enabled=flags["a"], mode="exclude")
    xy_module = XYModule(XY.shape[1], enabled=(flags["b"] and xy_mode == "branch"),
                         hidden_dim=64, embed_dim=32)
    n_concat = XY.shape[1] if (flags["b"] and xy_mode == "concat") else 0
    model = SexAwareEarlyFusionMLP(X.shape[1], sex_input, xy_module, n_concat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    val_t = (Xt[va], St[va], XYt[va])
    y_val = y[va.numpy()]
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, sb, xyb, yb, wb in loader:
            xb, sb, xyb, yb, wb = (t.to(device) for t in (xb, sb, xyb, yb, wb))
            opt.zero_grad()
            loss = weighted_mse(model(xb, sb, xyb), yb, wb)
            loss.backward()
            opt.step()
        val_rmse = float(np.sqrt(np.mean((batch_predict(model, val_t, device) - y_val) ** 2)))
        if val_rmse < best - 1e-5:
            best, best_epoch, bad = val_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    test_pred = batch_predict(model, (Xt[te], St[te], XYt[te]), device)
    return test_pred, best_epoch, best


def metrics(y, pred):
    if len(y) == 0:
        return {"n": 0, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan}
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return {"n": len(y), "rmse": rmse, "pearson": np.nan, "spearman": np.nan}
    return {"n": len(y), "rmse": rmse, "pearson": float(pearsonr(y, pred).statistic),
            "spearman": float(spearmanr(y, pred).statistic)}


def model_name(variant, proteomics, control, seed):
    name = f"EF_{variant}" + ("_prot" if proteomics else "")
    if control != "none":
        name += f"_shuf{control}"
    return f"{name}_s{seed}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="folder containing the *_aligned.csv files, sex_labels_matched.csv and the splits")
    ap.add_argument("--out-dir", type=Path, default=Path("ef_sex_aware_out"))
    ap.add_argument("--splits", nargs="+", default=["unseen_cell_lines"],
                    choices=["unseen_cell_lines", "unseen_tissues"])
    ap.add_argument("--variants", nargs="+", default=["base", "abc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--proteomics", action="store_true")
    ap.add_argument("--control", choices=["none", "sex", "xy", "both"], default="none",
                    help="shuffled control: 'sex' = sex label fed to (a)+(c); 'xy' = X/Y rows fed to (b); "
                         "'both' = both, from the same donor cell line")
    ap.add_argument("--shuffle-scope", choices=["within_tissue", "global"], default="within_tissue")
    ap.add_argument("--xy-mode", choices=["branch", "concat"], default="branch",
                    help="branch = XYModule embedding; concat = raw standardized genes (the notebook's way)")
    ap.add_argument("--weighting", choices=["reweighting", "malavika"], default="reweighting")
    ap.add_argument("--max-epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if a.device == "auto" else a.device))
    variants = {v: parse_variant(v) for v in a.variants}
    pred_dir = a.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    n_runs = len(a.splits) * len(variants) * len(a.seeds)
    print(f"device={device} | {n_runs} runs = {len(a.splits)} split(s) x {len(variants)} variant(s) x "
          f"{len(a.seeds)} seed(s) | proteomics={a.proteomics} control={a.control} "
          f"xy_mode={a.xy_mode} weighting={a.weighting}")

    omics, shared = P.load_omics(a.input_dir)
    labels = P.load_labels(a.input_dir)
    rows = []

    for family in a.splits:
        data, qc, conflicts, n_raw_test = P.prepare_responses(a.input_dir, family, labels, shared)
        n_test = int((data.split == "test").sum())
        n_drop = n_raw_test - n_test
        note = (f"{n_drop:,} dropped by QC -> harness needs --allow-partial or a filtered test set"
                if n_drop else "none dropped by QC -> full test coverage")
        print(f"\n{family}: {len(data):,} QC-usable rows; test rows {n_test:,} of {n_raw_test:,} frozen ({note})")
        qc.to_csv(a.out_dir / f"{family}_qc_excluded_cell_lines.csv", index=False)
        feats = P.build_features(data, omics)
        X = P.assemble_base(feats, a.proteomics)
        y = data["response"].to_numpy(dtype=np.float32)
        split = data["split"].to_numpy()
        test = data.loc[split == "test"]
        true_sex = test["sex_true"].to_numpy()

        for seed in a.seeds:
            sex_model, XY = P.model_inputs(data, feats, a.control, a.shuffle_scope, seed)
            S, keep = encode_sex(sex_model.tolist(), mode="exclude")
            assert bool(keep.all()), "QC should leave only Female/Male lines"
            S = S.numpy()
            for v, flags in variants.items():
                w = (P.train_weights(data, sex_model, a.weighting) if flags["c"]
                     else np.ones(int((split == "train").sum()), dtype=np.float32))
                name = model_name(v, a.proteomics, a.control, seed)
                pred, best_epoch, best_val = train_one(
                    X, S, XY, y, w, split, flags, a.xy_mode, seed, device, a.max_epochs, a.patience)

                y_true = test["response"].to_numpy(dtype=np.float32)
                fm = metrics(y_true[true_sex == "Female"], pred[true_sex == "Female"])
                mm = metrics(y_true[true_sex == "Male"], pred[true_sex == "Male"])
                om = metrics(y_true, pred)
                rows.append({
                    "split_family": family, "model": name, "variant": v, "seed": seed,
                    "sex_input": flags["a"], "xy_module": flags["b"], "reweighting": flags["c"],
                    "proteomics": a.proteomics, "control": a.control, "shuffle_scope": a.shuffle_scope,
                    "xy_mode": a.xy_mode, "weighting": a.weighting,
                    "n_train": int((split == "train").sum()), "n_val": int((split == "val").sum()),
                    "n_test": n_test, "n_test_frozen": n_raw_test,
                    "best_epoch": best_epoch, "val_rmse": best_val,
                    "overall_rmse": om["rmse"], "female_n": fm["n"], "male_n": mm["n"],
                    "female_rmse": fm["rmse"], "male_rmse": mm["rmse"],
                    "gap_female_minus_male": fm["rmse"] - mm["rmse"],
                    "gap_abs": abs(fm["rmse"] - mm["rmse"]),
                    "female_pearson": fm["pearson"], "male_pearson": mm["pearson"],
                    "female_spearman": fm["spearman"], "male_spearman": mm["spearman"],
                })
                pd.DataFrame({"sample_id": test["sample_id"].to_numpy(),
                              "cell_line_id": test["cell_line_id"].to_numpy(),
                              "drug_id": test["drug_id"].to_numpy(),
                              "y_pred": pred, "y_true": y_true}
                             ).to_csv(pred_dir / f"{name}__{family}.csv", index=False)
                pd.DataFrame(rows).to_csv(a.out_dir / "runs.csv", index=False)   # survives a crash
                print(f"  {family} {name:<28s} overall={om['rmse']:.4f} F={fm['rmse']:.4f} "
                      f"M={mm['rmse']:.4f} gap(F-M)={fm['rmse'] - mm['rmse']:+.4f} epoch={best_epoch}")

    runs = pd.DataFrame(rows)
    keys = ["split_family", "variant", "proteomics", "control"]
    summ = (runs.groupby(keys)
            .agg(n_seeds=("seed", "nunique"),
                 overall_rmse_mean=("overall_rmse", "mean"), overall_rmse_sd=("overall_rmse", "std"),
                 female_rmse_mean=("female_rmse", "mean"), male_rmse_mean=("male_rmse", "mean"),
                 gap_mean=("gap_female_minus_male", "mean"), gap_sd=("gap_female_minus_male", "std"),
                 gap_abs_mean=("gap_abs", "mean"))
            .reset_index())
    summ.to_csv(a.out_dir / "summary_by_variant.csv", index=False)
    print("\nSummary over seeds (gap = Female RMSE - Male RMSE):")
    print(summ.round(4).to_string(index=False))
    print(f"\nWrote {a.out_dir}/runs.csv, summary_by_variant.csv and predictions/ (harness naming).")


if __name__ == "__main__":
    main()