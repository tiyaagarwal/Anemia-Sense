# Why every model reported ~100% accuracy

> **This document is historical.** It analyzes the original Gender+Hemoglobin
> dataset this project started with (preserved at `analysis/legacy_dataset/anemia.csv`).
> This finding is *why* the project was redesigned around a richer, genuine
> multi-class CBC dataset — see `reports/data_quality/DATASET_CARD.md` for
> the current dataset's own (much smaller, still real) data-quality issues,
> and the root [README](../README.md) for the current pipeline and results.

Reproduce this analysis yourself: `python analysis/diagnose_data_issues.py`

## Root causes

**1. 62% of the dataset is exact duplicate rows.** 887 of 1421 rows in `data/anemia.csv` are byte-for-byte duplicates (534 unique rows). The original pipeline called `train_test_split` directly on the raw data, so copies of the same row routinely land on both sides of the split — part of the reported "test" accuracy was the model recognizing rows it was literally trained on, not generalizing.

**2. `Result` is a deterministic threshold function of Hemoglobin + Gender, with zero label noise.** This synthetic dataset encodes the WHO anemia cutoff directly instead of sampling clinically noisy labels:

| Gender | Anemic (Result=1) Hemoglobin range | Not anemic (Result=0) Hemoglobin range |
|---|---|---|
| 0 (female) | 6.6 – 11.9 | 12.0 – 16.9 |
| 1 (male) | 9.0 – 13.4 | 13.5 – 16.9 |

There is **zero overlap** at either cutoff. Given Hemoglobin and Gender, `Result` isn't a pattern to be learned — it's a lookup table that's already fully present in the input. Any reasonable classifier will fit this perfectly, which is expected, not a bug in the model code.

## Proof: an ablation on the deduplicated data

Cross-validated accuracy (stratified 5-fold, deduplicated data), with vs. without Hemoglobin as an input feature:

| Model | With Hemoglobin | Without Hemoglobin |
|---|---|---|
| Logistic Regression | 98.3% (±1.6%) | 60.3% (±4.9%) |
| Random Forest | 99.8% (±0.4%) | 47.4% (±4.1%) |
| Decision Tree | 100.0% (±0.0%) | 42.1% (±4.6%) |
| Naive Bayes | 93.4% (±1.8%) | 59.4% (±5.5%) |
| SVM | 97.6% (±1.7%) | 58.6% (±4.8%) |
| Gradient Boosting | 100.0% (±0.0%) | 50.4% (±5.7%) |

Once Hemoglobin is withheld, accuracy collapses to near-chance (the dataset is roughly 54%/46% class balance) — confirming MCH/MCHC/MCV don't independently carry the signal in this particular synthetic dataset, and that the near-100% accuracy with Hemoglobin included is a direct, provable consequence of the label definition rather than a leftover pipeline bug.

## What this means for the shipped model

Deduplicating the data (fix #1) is a real, necessary fix — it removes train/test contamination and makes the evaluation methodologically sound. Deduplicating alone does **not** bring accuracy down to something more "realistic", because cause #2 is not a bug: Hemoglobin is the actual clinical basis for the WHO anemia definition this dataset's labels were generated from, so a model with access to it is expected to score very high.

The shipped model (`train.py`) is trained on the deduplicated data with Hemoglobin included, evaluated with stratified cross-validation and a held-out test set (see `metrics.json` and the root [README](../README.md#model-performance) for the actual numbers), and its accuracy is reported alongside this diagnosis so a reader isn't left wondering whether it's real. See the README's **Limitations** section for what this dataset's small size (534 unique rows) and synthetic, noise-free labels mean for using this model beyond a portfolio demo.
