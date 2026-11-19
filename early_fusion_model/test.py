import pandas as pd, numpy as np

p = pd.read_csv("/Users/natashaprabhu/PycharmProjects/Algoverse-Research-TNAM/early_fusion_model/ef_testing/out_main/predictions/EF_base_s42__unseen_cell_lines.csv")

lab = pd.read_csv("/Users/natashaprabhu/Downloads/Early_Fusion_Model/sex_labels_matched.csv")[["SIDM", "final_sex"]]
p = p.merge(lab, left_on="cell_line_id", right_on="SIDM")

rmse = lambda d: np.sqrt(((d.y_pred - d.y_true) ** 2).mean())
print("rows:", len(p))
print("overall:", round(rmse(p), 4))
print("female: ", round(rmse(p[p.final_sex == "Female"]), 4))
print("male:   ", round(rmse(p[p.final_sex == "Male"]), 4))