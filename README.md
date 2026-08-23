# Anemia-Sense

[![Tests](https://github.com/tiyaagarwal/Anemia-Sense/actions/workflows/tests.yml/badge.svg)](https://github.com/tiyaagarwal/Anemia-Sense/actions/workflows/tests.yml)

A Flask web app that predicts anemia from CBC blood parameters (Gender, Hemoglobin, MCH, MCHC, MCV), comparing 6 classifiers (Logistic Regression, Random Forest, Decision Tree, Naive Bayes, SVM, Gradient Boosting) via cross-validation and shipping the best one.

## The interesting part: diagnosing why every model hit ~100% accuracy

The original version of this project trained 7 models on a single random `train_test_split` and reported that all of them scored ~100% accuracy — the kind of number that should make you suspicious, not proud. Digging in found two real, independent bugs:

1. **62% of the dataset (887 of 1421 rows) is exact duplicate rows.** A plain `train_test_split` puts copies of the same row on both sides, so part of the reported "test" accuracy was the model recognizing rows it was literally trained on.
2. **`Result` is a deterministic threshold function of Hemoglobin + Gender, with zero label noise** — this synthetic Kaggle dataset encodes the WHO anemia cutoff directly (female: anemic iff Hemoglobin < 12.0; male: anemic iff Hemoglobin < 13.5, with *zero* overlap at either boundary). Given Hemoglobin, `Result` isn't a pattern to learn, it's a lookup table already present in the input.

Full write-up with the reproducible diagnosis script and an ablation proving accuracy collapses to near-chance once Hemoglobin is withheld: **[`Anemia Detection Code/analysis/FINDINGS.md`](Anemia%20Detection%20Code/analysis/FINDINGS.md)**.

### What was actually fixed

- Deduplicated the dataset before any split — a real, necessary fix that removes train/test contamination.
- Replaced the single-split accuracy number with **stratified 5-fold cross-validation** plus a genuine **held-out test set** that never touches model selection.
- Model selection now picks by CV mean with a documented interpretability tie-break, instead of naive `argmax` over a single accuracy number (see `train.py`).
- Extracted training out of the Flask app into a standalone, reproducible `train.py` — `app.py` now only serves pre-trained artifacts and fails loudly if they're missing, instead of silently retraining 7 models on cold start.

Deduplicating alone does **not** bring accuracy down to something more "realistic", because cause #2 above isn't a bug — Hemoglobin genuinely is the clinical basis for the WHO anemia definition this dataset's labels were generated from. A model with access to it is *expected* to score very high. The point of this fix isn't a lower accuracy number; it's a methodologically sound one, backed by evidence instead of an unexamined single train/test split.

## Model performance (current, reproducible via `python train.py`)

Selected model: **Random Forest**, chosen by 5-fold cross-validation on the training split.

| Model | CV accuracy (5-fold) |
|---|---|
| Random Forest | 99.8% (±0.5%) |
| Decision Tree | 99.8% (±0.5%) |
| Gradient Boosting | 99.8% (±0.5%) |
| Logistic Regression | 98.6% (±1.7%) |
| SVM | 97.0% (±2.6%) |
| Naive Bayes | 93.5% (±3.6%) |

Held-out test set (107 rows, never used in model selection): **100% accuracy, ROC-AUC 1.0000**, confusion matrix `[[58, 0], [0, 49]]`. Full numbers (precision/recall/F1 per class) are written to `Anemia Detection Code/metrics.json` by `train.py`.

## Limitations (read before treating this as more than a portfolio demo)

- **The dataset is small and synthetic.** After deduplication there are only 534 unique rows, and the labels have no measurement noise — real clinical hemoglobin/CBC readings do. This model has not been validated against a real, noisy clinical dataset and shouldn't be treated as one.
- **This is not a diagnostic tool.** It reproduces a known clinical threshold rule from CBC inputs; it does not add independent diagnostic signal beyond what the WHO Hemoglobin cutoff already tells you.
- **`npm`-style dependency pinning, not a training-data audit.** `requirements.txt` pins the exact versions this was verified against; it does not certify the source dataset's provenance beyond what's in `data/anemia.csv`.

## Project structure

```
Anemia Detection Code/
  app.py                  Flask app — loads model.pkl/scaler.pkl and serves predictions
  train.py                 Reproducible training pipeline (dedup -> CV -> held-out test -> save artifacts)
  data/anemia.csv           Source dataset
  model.pkl, scaler.pkl,
  metrics.json               Committed output of the last `python train.py` run
  analysis/
    diagnose_data_issues.py  Reproduces the duplicate-row + threshold + ablation findings
    FINDINGS.md               Write-up of the diagnosis, with real captured output
  tests/
    test_data_quality.py     Regression guards for the two root causes above
    test_app.py                Flask route tests (health, predict form, /result validation)
  templates/, static/         Web UI
```

## Running it locally

Requires Python 3.11+.

```bash
cd "Anemia Detection Code"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python train.py          # trains and saves model.pkl / scaler.pkl / metrics.json
python -m pytest tests/  # 11 tests: data-quality regression guards + Flask routes

python app.py             # http://localhost:5000 (set FLASK_DEBUG=true for the interactive debugger)
```

`model.pkl`/`scaler.pkl`/`metrics.json` are committed, so `python app.py` works right after `pip install` without retraining — `train.py` is there for reproducibility and for anyone who wants to change the feature set or model list.

## Tech stack

Python, scikit-learn, pandas, Flask, pytest, GitHub Actions.
