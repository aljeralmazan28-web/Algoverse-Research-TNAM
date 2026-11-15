PLEASE READ!! - malavika

early_fusion_results/: contains all trained early fusion models, preprocessing files, evaluation results, and ablation experiments.

unseen_cell_lines/: experiments evaluating model generalization to held-out ce lines.

- models/ — Trained EF1–EF4 PyTorch models 
- training_history/ — Training histories for all four models
- preprocessing/ — Saved feature preprocessing objects 
- quality_checks/ — Excluded cell lines and sex-label conflict reports
- ablation_analysis/ — Comparisons measuring the effects of proteomics and sex-aware components

unseen_tissue: experiments evaluating model generalization to held-out tissue categories.

- `models/` — Trained EF1–EF4 PyTorch models
- `training_history/` — Training histories
- `preprocessing/` — Saved preprocessing objects
- `quality_checks/` — Sex-metadata validation reports

combined_results/: contains results across both unseen-cell-line and unseen-tissue experiments.

- `early_fusion_master_benchmark.csv` — Summary metrics for all four early fusion variants across both split families
- `early_fusion_benchmark_female_male_gap.csv` — Female, male, and absolute performance-gap metrics
- `early_fusion_test_predictions.csv` — Actual and predicted drug-response values

The combined files include a `split_family` column to distinguish unseen-cell-line and unseen-tissue experiments.

harness_results/: contains the shared evaluation framework designed to compare early, intermediate, late, and graph fusion models.

code/

- `master_benchmark_harness.ipynb` — shared evaluation notebook

`results/unseen_cell_lines/`

- `master_benchmark_wide.csv` — Side-by-side comparison of female and male model performance
- `master_benchmark_long.csv` — Evaluation metrics in long format
- `master_benchmark_24row_template.csv` — Benchmark template with missing model configurations marked pending
- `matched_test_predictions.csv` — Predictions aligned to comparable test observations
- `matched_test_cohorts.csv` — Information about matched evaluation cohorts
- `model_coverage.csv` — Tracks which fusion configurations have been evaluated


The benchmark template also includes the proteomics dimension, so the current template contains 48 configuration/group rows rather than only 24.

2. Early Fusion Model Configurations: four early fusion variants were trained to evaluate the contributions of proteomics and sex-aware components.

| Model | Proteomics | Sex-Aware |
| EF1   | No         | No        |
| EF2   | Yes        | No        |
| EF3   | No         | Yes.      |
| EF4   | Yes        | Yes       |

All four models use the same underlying early fusion neural-network approach, with different feature and training configurations.

Sex-aware configurations incorporate reviewed sex labels, an X/Y-related gene module, and sex-by-tissue training weights.

3. Ablation Experiments

Ablation experiments measure how individual components affect model performance.

The following comparisons are included:

 EF2 vs. EF1:  Effect of adding proteomics 
 EF3 vs. EF1:  Effect of adding sex-aware components 
 EF4 vs. EF2:  Effect of adding sex-aware components when proteomics is included 
 EF4 vs. EF3:  Effect of adding proteomics when sex-aware components are included 

Results are stored in: early_fusion_results/unseen_cell_lines/ablation_analysis/early_fusion_ablation_comparison.csv

The comparison reports changes in overall RMSE, female RMSE, male RMSE, and the absolute female–male RMSE gap.

For RMSE differences calculated as new model minus baseline, negative values indicate lower prediction error, while positive values indicate higher prediction error.

The current ablation-comparison CSV covers unseen cell lines only. Unseen-tissue model metrics are available in the combined benchmark.


4. Evaluation Experiments

Unseen Cell Lines: Models are evaluated on cell lines excluded from training to measure generalization across different cellular profiles.

Unseen Tissues: Models are evaluated on held-out tissue categories to measure their ability to generalize across biological contexts.


 5. Evaluation Metrics

The models are assessed using:

- RMSE: Measures prediction error between actual and predicted drug responses; lower is better.
- Pearson Correlation: Measures linear association between actual and predicted responses.
- Spearman Correlation: Measures how well predicted responses preserve the ranking of actual responses.
- Female RMSE: Prediction error for female-labeled cell lines.
- Male RMSE: Prediction error for male-labeled cell lines.
- RMSE Gap: Absolute difference between female and male RMSE.

The primary performance-disparity metric is:

RMSE Gap = |Female RMSE − Male RMSE|

A smaller gap indicates more similar prediction errors across groups, but should be interpreted alongside overall accuracy and group-specific RMSE.



