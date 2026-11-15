# Multimodal Attention Fusion for Drug Response Prediction

## Overview

This project implements a transformer-based multimodal attention fusion model to predict drug responses in cancer cell lines using molecular features from multiple omics datasets.

The primary objective is to investigate whether incorporating proteomics and explicit sex-aware information improves drug response prediction compared to a baseline model using gene-associated expression scores, mutation, and copy number variation (CNV) features.

The model was evaluated using official cell-line-blind and tissue-blind splits, with performance reported overall and separately for female- and male-labeled cell lines.

## Datasets and Preprocessing

The experiments integrate four molecular feature modalities:

- **Gene-associated expression scores:** 18,025 original features from Harmonizome, representing gene association scores rather than conventional quantitative RNA expression measurements.
- **Mutation:** Binary mutation features indicating mutation status.
- **Copy Number Variation (CNV):** Copy number alteration features.
- **Proteomics:** Protein abundance measurements from the ProCan-DepMapSanger dataset.

Drug-response measurements were obtained from the `GDSC_IC50.csv` file in the DeepCDR dataset. The exact transformation of the reported IC50 values requires confirmation.

After matching molecular data, drug-response measurements, and usable sex labels, the final modeling dataset contained:

| Dataset characteristic | Count |
|---|---:|
| Eligible cancer cell lines | 491 |
| Drugs | 266 |
| Observed cell line–drug response pairs | 106,153 |
| Female-labeled cell lines | 224 |
| Male-labeled cell lines | 267 |

Preprocessing included missing-value imputation, constant-feature removal, feature filtering, and scaling where appropriate. All preprocessing parameters were fitted exclusively on each experiment's training partition to prevent data leakage.

## Model Architecture

The model uses modality-specific neural network encoders followed by a transformer encoder to integrate molecular representations.

Each omics modality is processed through a feedforward encoder that projects its features into a shared 64-dimensional representation. The encoded modalities are combined using a two-layer transformer encoder with four attention heads.

The resulting fused representation is concatenated with a learned drug-ID embedding and, when enabled, a learned sex embedding. A multilayer perceptron then predicts the drug-response value.

## Experimental Configurations

Four model configurations were implemented to evaluate the contributions of proteomics and sex-aware input.

| Model | Expression-associated scores | Mutation | CNV | Proteomics | Sex-aware |
|---|---|---|---|---|---|
| A: Baseline | Yes | Yes | Yes | No | No |
| B: Proteomics | Yes | Yes | Yes | Yes | No |
| C: Sex-aware | Yes | Yes | Yes | No | Yes |
| D: Full fusion | Yes | Yes | Yes | Yes | Yes |

All configurations used the same training procedure to support controlled comparisons.

## Training and Evaluation

Models were trained using PyTorch with the following settings:

- **Optimizer:** AdamW
- **Learning rate:** 0.001
- **Batch size:** 256
- **Maximum epochs:** 15
- **Early stopping patience:** 4 epochs
- **Loss function:** Mean Squared Error (MSE)
- **Random seeds:** 42, 123, 456

Two official evaluation strategies were used:

**Cell-line-blind:** Cell lines were partitioned into separate training, validation, and test sets, preventing the same cell line from appearing across partitions.

**Tissue-blind:** Tissue categories were held out entirely from training. Validation tissues were Large Intestine, Ovary, and Pancreas, while test tissues were Breast, Skin, and Stomach.

Performance was measured using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Pearson correlation. Sex-stratified performance was evaluated separately for female- and male-labeled cell lines.

## Results

The tables below report mean held-out test RMSE across three training seeds. Lower RMSE indicates better prediction accuracy.

### Cell-Line-Blind Evaluation

| Model | Overall RMSE | Female RMSE | Male RMSE |
|---|---:|---:|---:|
| A: Baseline | 1.3507 | 1.3772 | 1.3297 |
| **B: Proteomics** | **1.3360** | **1.3560** | **1.3202** |
| C: Sex-aware | 1.3452 | 1.3732 | 1.3229 |
| D: Full fusion | 1.3439 | 1.3618 | 1.3298 |

Adding proteomics without explicit sex-aware input reduced mean test RMSE by approximately 1.09% compared with the baseline.

### Tissue-Blind Evaluation

| Model | Overall RMSE | Female RMSE | Male RMSE |
|---|---:|---:|---:|
| A: Baseline | 1.4367 | 1.4402 | 1.4262 |
| **B: Proteomics** | **1.4224** | **1.4252** | **1.4142** |
| C: Sex-aware | 1.4465 | 1.4418 | 1.4605 |
| D: Full fusion | 1.4335 | 1.4352 | 1.4282 |

Adding proteomics reduced mean test RMSE by approximately 1.00% compared with the baseline.

### Proteomics Ablation

Model B achieved lower overall test RMSE than Model A across all three training seeds in both evaluation strategies.

| Evaluation | Mean RMSE difference (B − A) | Seeds showing improvement |
|---|---:|---:|
| Cell-line-blind | -0.0148 | 3/3 |
| Tissue-blind | -0.0143 | 3/3 |

These results provide preliminary descriptive support for the contribution of proteomics to multimodal drug-response prediction.

The additional sex embedding did not consistently improve predictive performance. Sex-stratified results also varied across evaluation strategies, emphasizing the importance of evaluating model performance across different biological groups.

Model selection based on validation RMSE differed from the configurations achieving the lowest observed test RMSE. Test-set rankings are reported as exploratory ablation results rather than used for retrospective model selection.

## Statistical Analysis

Cell-line-level bootstrap resampling was used to estimate uncertainty in female–male RMSE gaps and paired model comparisons.

The bootstrap intervals from the initial single-seed comparisons generally included zero, indicating that the observed performance differences were not clearly distinguishable from zero under those analyses.

Multiple-seed experiments provided an additional assessment of sensitivity to model initialization and training randomness.

## Limitations

- The exact transformation of the drug-response values in `GDSC_IC50.csv` has not yet been verified.
- Sex labels include provisional assignments and may contain classification uncertainty.
- The expression-associated input contains gene association scores rather than standard quantitative gene expression measurements.
- Differences in tissue composition and sample sizes may influence sex-stratified performance.
- Three training seeds assess training variability but do not replace independent dataset validation.
- The model uses learned drug-ID embeddings and is not designed to generalize to entirely unseen drugs.
- Results are not directly comparable with GDSC2 AUC benchmarks without aligning drug-response targets and evaluation protocols.

## Repository Contents

- `multimodal_fusion_model.ipynb`: Model implementation, training, ablation experiments, and evaluation.
- `multiomics_preprocessing_pipeline.py`: Reusable preprocessing pipeline for molecular features.
- `results/`: Summary tables for multi-seed test performance and proteomics ablation results.

Large model checkpoints, molecular datasets, and full experimental archives are stored separately rather than committed directly to the repository.

## Conclusion

The multimodal attention-fusion framework successfully integrated molecular features for drug-response prediction and enabled controlled evaluation of proteomics and sex-aware model components.

Across three training seeds, proteomics integration without explicit sex-aware input produced lower mean test RMSE than the baseline in both cell-line-blind and tissue-blind experiments. However, the observed improvements remain exploratory and require further statistical and external validation.

These findings provide a foundation for investigating how additional molecular modalities influence drug-response prediction and whether their predictive contributions vary across biological groups.
