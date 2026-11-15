"""Self-tests with deliberately SYNTHETIC model predictions; do not report as H2 results."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import (validate_and_load_splits, read_prediction_and_join,
                      one_sex_metrics, tissue_adjusted_test, two_way_cluster_bootstrap,
                      benjamini_hochberg)

ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, _ = validate_and_load_splits(ROOT / "splits")

    def test_holdouts_are_disjoint(self):
        keys = {"unseen_cell_lines": "cell_line_id", "unseen_drugs": "drug_id",
                "unseen_tissues": "tissue"}
        for split, key in keys.items():
            p = self.splits[split]
            self.assertFalse(set(p["train"][key]) & set(p["test"][key]))
            self.assertFalse(set(p["val"][key]) & set(p["test"][key]))

    def test_known_gap_and_tissue_permutation(self):
        test = self.splits["unseen_drugs"]["test"].copy()
        # Synthetic perfect-on-scale offset: males +0.5, females +1.5 response units.
        test["y_pred"] = test["response"] + np.where(test["sex"] == "Male", 0.5, 1.5)
        male = one_sex_metrics(test, "Male")
        female = one_sex_metrics(test, "Female")
        self.assertAlmostEqual(male["rmse"], 0.5, places=9)
        self.assertAlmostEqual(female["rmse"], 1.5, places=9)
        self.assertAlmostEqual(female["rmse"] - male["rmse"], 1.0, places=9)
        adj = tissue_adjusted_test(test, n_boot=40, n_perm=99, seed=42, confidence=0.95)
        self.assertAlmostEqual(adj["tissue_adjusted_rmse_gap_female_minus_male"], 1.0, places=9)
        self.assertLessEqual(adj["within_tissue_permutation_p_two_sided"], 0.05)
        self.assertGreater(adj["tissues_with_both_sexes"], 0)
        ci = two_way_cluster_bootstrap(test, 40, 42)
        self.assertTrue(np.isfinite(ci).all())
        self.assertTrue(np.allclose(ci[:, 2], 1.0))

    def test_bad_predictions_rejected(self):
        test = self.splits["unseen_cell_lines"]["test"]
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "m.csv"
            pd.DataFrame({"sample_id": [test["sample_id"].iat[0]] * 2,
                          "y_pred": [0, 0]}).to_csv(f, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_prediction_and_join(f, test, allow_partial=False)
            pd.DataFrame({"sample_id": [test["sample_id"].iat[0]],
                          "y_pred": [0]}).to_csv(f, index=False)
            with self.assertRaisesRegex(ValueError, "no prediction"):
                read_prediction_and_join(f, test, allow_partial=False)
            df, missing = read_prediction_and_join(f, test, allow_partial=True)
            self.assertEqual(len(df), 1)
            self.assertEqual(missing, len(test)-1)

    def test_bh_correction_monotonic(self):
        d = [{"within_tissue_permutation_p_two_sided": p} for p in (0.01, 0.02, 0.40)]
        benjamini_hochberg(d)
        self.assertAlmostEqual(d[0]["p_fdr_bh"], 0.03)
        self.assertAlmostEqual(d[1]["p_fdr_bh"], 0.03)
        self.assertAlmostEqual(d[2]["p_fdr_bh"], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
