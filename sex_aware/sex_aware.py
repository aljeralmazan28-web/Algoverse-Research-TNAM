"""
sex_aware.py -- plug-in modules for the Week 3 sex-aware add-ons.

    (a) sex as input          -> encode_sex(), SexInput
    (b) X/Y gene module       -> XYModule, select_xy_columns()
    (c) sex-balanced weights  -> weighted_mse()   (weights come from your Week 2 function)

Everything is switched on/off from one config block, so any teammate's model
(early fusion, MOLI-style late fusion, transformer, graph) can import this file
instead of re-implementing it:

    sex_aware:
      use_sex_input: false
      use_xy_module: false
      use_reweighting: false
      sex_label_mode: "unknown_indicator"   # or "exclude"
      xy_embed_dim: 32
      xy_hidden_dim: 64

All flags off  ->  the model is exactly the baseline (same shapes, same loss).
"""
import numpy as np
import torch
import torch.nn as nn

# ==========================================================================
# Config
# ==========================================================================
class SexAwareConfig:
    """Reads the shared config block (a plain dict, e.g. from YAML)."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.use_sex_input = bool(cfg.get("use_sex_input", False))
        self.use_xy_module = bool(cfg.get("use_xy_module", False))
        self.use_reweighting = bool(cfg.get("use_reweighting", False))
        self.sex_label_mode = cfg.get("sex_label_mode", "unknown_indicator")
        self.xy_embed_dim = int(cfg.get("xy_embed_dim", 32))
        self.xy_hidden_dim = int(cfg.get("xy_hidden_dim", 64))
        if self.sex_label_mode not in ("unknown_indicator", "exclude"):
            raise ValueError(f"bad sex_label_mode: {self.sex_label_mode}")


# ==========================================================================
# (a) Sex as input
# ==========================================================================
# One-hot order is [female, male, unknown].
SEX_DIM = {"unknown_indicator": 3, "exclude": 2}


def encode_sex(labels, uncertain=None, mode="unknown_indicator"):
    """
    labels    : sequence of 'F'/'M' (or 'female'/'male'); anything else = missing
    uncertain : optional bool sequence, True = line was flagged uncertain in Week 2
    mode      : 'unknown_indicator' -> [F, M, unknown]; uncertain/missing -> unknown
                'exclude'           -> [F, M]; caller must drop rows where keep_mask is False
    Returns (features [N, SEX_DIM[mode]] float tensor, keep_mask [N] bool tensor).
    """
    n = len(labels)
    unc = np.zeros(n, dtype=bool) if uncertain is None else np.asarray(uncertain, dtype=bool)
    up = [str(l).upper() for l in labels]
    is_f = np.array([u in ("F", "FEMALE") for u in up])
    is_m = np.array([u in ("M", "MALE") for u in up])
    known = (is_f | is_m) & ~unc

    if mode == "unknown_indicator":
        feats = np.zeros((n, 3), dtype=np.float32)
        feats[known & is_f, 0] = 1.0
        feats[known & is_m, 1] = 1.0
        feats[~known, 2] = 1.0
        keep = np.ones(n, dtype=bool)
    elif mode == "exclude":
        feats = np.zeros((n, 2), dtype=np.float32)
        feats[known & is_f, 0] = 1.0
        feats[known & is_m, 1] = 1.0
        keep = known
    else:
        raise ValueError(f"unknown sex_label_mode: {mode}")
    return torch.from_numpy(feats), torch.from_numpy(keep)


class SexInput(nn.Module):
    """
    Appends the sex features to a tensor: the concatenated omics vector in early
    fusion, or the combined embedding just before the prediction head in late fusion.
    Disabled -> returns x unchanged. The layer after it must take
    base_dim + sex_input.extra_dim inputs.
    """

    def __init__(self, enabled=True, mode="unknown_indicator"):
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.extra_dim = SEX_DIM[mode] if enabled else 0

    def forward(self, x, sex_feats=None):
        if not self.enabled:
            return x
        if sex_feats is None:
            raise ValueError("SexInput is enabled but sex_feats was not passed")
        return torch.cat([x, sex_feats.to(dtype=x.dtype, device=x.device)], dim=-1)


# ==========================================================================
# (b) X/Y gene module
# ==========================================================================
# PROVISIONAL list from the Week 1 sex-inference script. Replace with Trishna's
# X/Y module list when she exports it (pass it in; nothing else changes).
PROVISIONAL_XY_GENES = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY"]


def select_xy_columns(gene_names, xy_genes=None):
    """
    Finds which columns of the expression matrix are X/Y genes.
    Returns (indices list, found gene names, missing gene names).
    Always print/log `missing` -- a silent name mismatch (e.g. 'XIST' vs 'XIST ')
    is the most likely bug here.
    """
    xy_genes = list(PROVISIONAL_XY_GENES if xy_genes is None else xy_genes)
    pos = {g: i for i, g in enumerate(gene_names)}
    idx, found, missing = [], [], []
    for g in xy_genes:
        if g in pos:
            idx.append(pos[g]); found.append(g)
        else:
            missing.append(g)
    return idx, found, missing


class XYModule(nn.Module):
    """
    Small separate branch that sees ONLY the X/Y gene expression and outputs an
    embedding, giving the model a dedicated path for sex-linked signal.

        xy_expr [N, n_xy]  ->  Linear -> ReLU -> Linear  ->  [N, embed_dim]

    Concatenate that embedding onto the rest of the model (same place as SexInput).
    Disabled -> forward returns None and extra_dim == 0, so the Week 4 ablation
    "remove X/Y genes" is one flag.

    The module also needs the *same* X/Y genes to be removed from the main omics
    input in the Week 4 ablation -- that is a data-side step, not done here.
    """

    def __init__(self, n_xy, enabled=True, hidden_dim=64, embed_dim=32, dropout=0.1):
        super().__init__()
        self.enabled = enabled
        self.extra_dim = embed_dim if enabled else 0
        if enabled:
            self.net = nn.Sequential(
                nn.Linear(n_xy, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
            )

    def forward(self, xy_expr):
        if not self.enabled:
            return None
        return self.net(xy_expr)


def concat_sex_aware(x, sex_input, xy_module, sex_feats=None, xy_expr=None):
    """
    One call that applies whichever add-ons are on, in a fixed order:
    x -> [x, xy_embedding] -> [.., sex_onehot]. Use this at the fusion point so
    teammates do not each pick their own order.
    Final width = base_dim + xy_module.extra_dim + sex_input.extra_dim.
    """
    if xy_module is not None and xy_module.enabled:
        if xy_expr is None:
            raise ValueError("XYModule is enabled but xy_expr was not passed")
        x = torch.cat([x, xy_module(xy_expr)], dim=-1)
    if sex_input is not None:
        x = sex_input(x, sex_feats)
    return x


def sex_aware_extra_dim(sex_input, xy_module):
    """How many extra input features the layer after the fusion point must accept."""
    return (sex_input.extra_dim if sex_input is not None else 0) + \
           (xy_module.extra_dim if xy_module is not None else 0)


def build_sex_aware(cfg, n_xy):
    """Builds both modules from the config. Returns (SexInput, XYModule)."""
    c = cfg if isinstance(cfg, SexAwareConfig) else SexAwareConfig(cfg)
    return (
        SexInput(enabled=c.use_sex_input, mode=c.sex_label_mode),
        XYModule(n_xy, enabled=c.use_xy_module,
                 hidden_dim=c.xy_hidden_dim, embed_dim=c.xy_embed_dim),
    )


# ==========================================================================
# (c) Sex-balanced reweighting (loss only)
# ==========================================================================
def make_sex_tissue_weights(sex, tissue):
    """
    STAND-IN for testing. For real runs use your Week 2 function.
    Inverse-frequency weight per (sex x tissue) group, normalised to mean 1.
    """
    keys = np.array([f"{s}|{t}" for s, t in zip(sex, tissue)])
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def weighted_mse(pred, target, weights=None):
    """
    sum(w * (pred - target)^2) / sum(w). weights=None or all ones == plain MSE.
    Only the loss changes; the architecture is untouched.
    NOTE: if one row of the data is a (cell line, drug) pair, weights must be
    per ROW (repeat each line's weight for each of its drugs), in the same order.
    """
    sq = (pred - target) ** 2
    if weights is None:
        return sq.mean()
    w = weights.to(dtype=sq.dtype, device=sq.device)
    return (w * sq).sum() / w.sum()