"""Tests for sex_aware.py (needs torch). Run: python test_sex_aware.py"""
import numpy as np
import torch
import torch.nn.functional as F
from sex_aware import (encode_sex, SexInput, XYModule, select_xy_columns, concat_sex_aware,
                       sex_aware_extra_dim, build_sex_aware, SexAwareConfig, weighted_mse,
                       make_sex_tissue_weights, PROVISIONAL_XY_GENES)

torch.manual_seed(0); np.random.seed(0)

# ---- (a) encoding ----
labels = ["F", "M", "F", None, "M", "female"]
unc = [False, False, True, False, False, False]
f, keep = encode_sex(labels, unc, "unknown_indicator")
assert f.shape == (6, 3) and keep.all()
assert f[0].tolist() == [1, 0, 0] and f[1].tolist() == [0, 1, 0]
assert f[2].tolist() == [0, 0, 1] and f[3].tolist() == [0, 0, 1] and f[5].tolist() == [1, 0, 0]
f2, keep2 = encode_sex(labels, unc, "exclude")
assert f2.shape == (6, 2) and keep2.tolist() == [True, True, False, False, True, True]
print("(a) encode_sex OK")

x = torch.randn(6, 10)
assert SexInput(False)(x).equal(x)
out = SexInput(True)(x, f)
assert out.shape == (6, 13) and out[:, :10].equal(x)
print("(a) SexInput on/off OK")

# ---- (b) X/Y gene selection + module ----
genes = ["A", "XIST", "B", "RPS4Y1", "C", "KDM5D"]
idx, found, missing = select_xy_columns(genes)
assert idx == [1, 3, 5] and found == ["XIST", "RPS4Y1", "KDM5D"] and missing == ["DDX3Y", "UTY"]
print("(b) select_xy_columns OK, missing:", missing)

xy = XYModule(n_xy=3, enabled=True, hidden_dim=8, embed_dim=4)
assert xy(torch.randn(6, 3)).shape == (6, 4)
off = XYModule(n_xy=3, enabled=False)
assert off(torch.randn(6, 3)) is None and off.extra_dim == 0
assert sum(p.numel() for p in off.parameters()) == 0   # disabled adds no parameters
print("(b) XYModule on/off OK")

# ---- combined fusion-point helper ----
base = torch.randn(6, 10)
si, xm = build_sex_aware({"use_sex_input": True, "use_xy_module": True,
                          "xy_embed_dim": 4, "xy_hidden_dim": 8}, n_xy=3)
fused = concat_sex_aware(base, si, xm, sex_feats=f, xy_expr=torch.randn(6, 3))
assert fused.shape == (6, 10 + 4 + 3) and sex_aware_extra_dim(si, xm) == 7
assert fused[:, :10].equal(base) and fused[:, -3:].equal(f)      # order: x, xy, sex
si0, xm0 = build_sex_aware({}, n_xy=3)                            # everything off
assert concat_sex_aware(base, si0, xm0).equal(base) and sex_aware_extra_dim(si0, xm0) == 0
print("concat_sex_aware: order and all-off == baseline OK")

# ---- (c) weighted loss ----
p, t = torch.randn(50), torch.randn(50)
assert torch.allclose(weighted_mse(p, t, torch.ones(50)), F.mse_loss(p, t))
assert torch.allclose(weighted_mse(p, t, None), F.mse_loss(p, t))
w = make_sex_tissue_weights(["F"] * 10 + ["M"] * 40, ["lung"] * 50)
assert abs(w.mean().item() - 1) < 1e-5 and w[0] > w[-1]
print("(c) weighted_mse and weights OK")

# ---- toy training: all three on, loss falls, and the flags change the model ----
N = 400
genes = [f"g{i}" for i in range(18)] + ["XIST", "RPS4Y1"]
X = torch.randn(N, len(genes))
sex = np.random.choice(["F", "M"], N, p=[0.3, 0.7]); tis = np.random.choice(["lung", "skin", "brain"], N)
sf, _ = encode_sex(sex)
xy_idx, _, _ = select_xy_columns(genes)
xy_expr = X[:, xy_idx]
y = X[:, 0] + 0.8 * (sf[:, 0] - sf[:, 1]) + 0.1 * torch.randn(N)
wts = make_sex_tissue_weights(sex, tis)

def run(cfg):
    torch.manual_seed(0)
    si, xm = build_sex_aware(cfg, n_xy=len(xy_idx))
    head = torch.nn.Sequential(torch.nn.Linear(len(genes) + sex_aware_extra_dim(si, xm), 16),
                               torch.nn.ReLU(), torch.nn.Linear(16, 1))
    params = list(head.parameters()) + list(xm.parameters())
    opt = torch.optim.Adam(params, 1e-2)
    first = last = None
    for _ in range(250):
        opt.zero_grad()
        z = concat_sex_aware(X, si, xm, sex_feats=sf, xy_expr=xy_expr)
        loss = weighted_mse(head(z).squeeze(-1), y, wts if cfg.get("use_reweighting") else None)
        loss.backward(); opt.step()
        first = first if first is not None else loss.item(); last = loss.item()
    return first, last

base_f, base_l = run({})
all_f, all_l = run({"use_sex_input": True, "use_xy_module": True, "use_reweighting": True,
                    "xy_embed_dim": 8, "xy_hidden_dim": 16})
assert base_l < base_f and all_l < all_f
assert all_l < base_l, "sex input should help on this toy task, where y depends on sex"
print(f"toy training OK: baseline {base_f:.3f}->{base_l:.3f}, all three on {all_f:.3f}->{all_l:.3f}")