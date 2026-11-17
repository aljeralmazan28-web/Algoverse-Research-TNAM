"""Tests for sex_shuffle.py (numpy only). Run: python test_sex_shuffle.py"""
import numpy as np
from collections import Counter
from sex_shuffle import shuffle_sex_labels

rng = np.random.default_rng(1)
N = 600
tissue = rng.choice(["lung", "skin", "brain", "blood"], N)
sex = np.where(rng.random(N) < 0.35, "F", "M")

# within-tissue: per-tissue F/M counts unchanged, labels actually changed
sh = shuffle_sex_labels(sex, tissue, within_tissue=True, seed=0)
for t in np.unique(tissue):
    m = tissue == t
    assert Counter(sex[m]) == Counter(sh[m]), t
assert (sh != sex).mean() > 0.2, "shuffle changed almost nothing"
print("within-tissue: counts per tissue preserved, labels changed OK")

# global: overall counts unchanged, but tissue counts may move
g = shuffle_sex_labels(sex, tissue, within_tissue=False, seed=0)
assert Counter(sex) == Counter(g)
assert any(Counter(sex[tissue == t]) != Counter(g[tissue == t]) for t in np.unique(tissue))
print("global: overall counts preserved OK")

# reproducible per seed, different across seeds, input untouched
orig = sex.copy()
assert (shuffle_sex_labels(sex, tissue, seed=3) == shuffle_sex_labels(sex, tissue, seed=3)).all()
assert (shuffle_sex_labels(sex, tissue, seed=3) != shuffle_sex_labels(sex, tissue, seed=4)).any()
assert (sex == orig).all()
print("seeding and no in-place edit OK")

# a tissue with a single line (or single sex) must not crash
t2 = np.array(["a", "b", "b", "b"]); s2 = np.array(["F", "M", "M", "M"])
assert list(shuffle_sex_labels(s2, t2)) == list(s2)
print("degenerate tissues OK")