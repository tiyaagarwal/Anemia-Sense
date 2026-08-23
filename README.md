# Anemia-Sense: Leakage-Aware CBC Analytics & Machine Learning

[![Tests](https://github.com/tiyaagarwal/Anemia-Sense/actions/workflows/tests.yml/badge.svg)](https://github.com/tiyaagarwal/Anemia-Sense/actions/workflows/tests.yml)

An end-to-end, production-style ML system that classifies anemia-related conditions from a full Complete Blood Count (CBC) panel — 9 real diagnostic classes, benchmarked models, SHAP explainability, a REST API, and honest, reproducible metrics. Built after diagnosing a data-leakage bug in an earlier version that made the whole problem trivial.

**Live in the app:** the Model Performance and Data & Methodology pages (`/performance`, `/methodology` — see [Application](#application) below) render everything in this README directly from the committed `reports/` artifacts — nothing on this page or in the app is hand-typed separately from what the pipeline actually produced.

## Why the original version was suspicious, not good

An earlier version of this project used only **Gender + Hemoglobin** as input, compared 7 classifiers, and reported ~100% accuracy on all of them. That's a red flag, not an achievement. Two real bugs were behind it:

1. **62% of the dataset was exact duplicate rows.** A plain `train_test_split` put copies of the same row on both sides, so part of the "test" accuracy was the model recognizing rows it had literally trained on.
2. **The label was a deterministic threshold function of Hemoglobin + Gender**, with zero label noise — female: anemic iff Hemoglobin < 12.0; male: anemic iff Hemoglobin < 13.5, with *zero* overlap at either boundary. Given Hemoglobin, the label wasn't a pattern to learn, it was a lookup table already present in the input.

Full history, with the original reproducible diagnosis script and its captured output: [`analysis/FINDINGS.md`](analysis/FINDINGS.md).

**Rather than presenting inflated metrics, the project was redesigned around a richer CBC feature space and a leakage-aware evaluation methodology** — that redesign is documented end-to-end below.

## Dataset

**Source:** [Anemia Types Classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification) (Kaggle, Apache 2.0) — real CBC data, manually diagnosed. Full dataset card, audit, and license details: [`reports/data_quality/DATASET_CARD.md`](reports/data_quality/DATASET_CARD.md).

- 1281 raw rows → 1232 after deduplication (49 exact duplicates, 3.8%) → **1199 after dropping physiologically implausible rows** (33 rows, 2.7% — e.g. MCV=990, HGB=-10; these read as data-entry corruption, not extreme pathology).
- **14 real CBC features:** WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT.
- **9 genuine diagnostic classes** (present in the source data, not invented): Healthy, Iron deficiency anemia, Normocytic hypochromic anemia, Normocytic normochromic anemia, Other microcytic anemia, Macrocytic anemia, Thrombocytopenia, Leukemia, Leukemia with thrombocytopenia. Imbalanced — Healthy (319) down to Leukemia with thrombocytopenia (10).
- **Leakage check, done before committing to this dataset:** a depth-3 decision-tree stump on any single feature tops out at **62.1% accuracy** (Hemoglobin). No feature reconstructs the label the way Hemoglobin alone did in the original dataset. Full audit and reasoning: [`reports/data_quality/DATASET_CARD.md`](reports/data_quality/DATASET_CARD.md).

## Model performance (reproducible via `python -m src.models.train`)

**Selected model: Gradient Boosting** — picked by 5-fold cross-validation on macro-F1 (not raw accuracy, since classes are imbalanced 319:10), then tuned with `RandomizedSearchCV`.

| Model | CV macro-F1 (5-fold) | CV accuracy |
|---|---|---|
| **Gradient Boosting** | **92.9% (±4.7%)** | 98.6% |
| HistGradientBoosting | 91.5% (±3.8%) | 97.4% |
| Random Forest | 87.2% (±6.5%) | 97.3% |
| Logistic Regression | 75.5% (±5.1%) | 87.3% |
| SVM | 71.9% (±5.2%) | 84.4% |

**Held-out test set** (240 rows, stratified, never touched during model selection or tuning): accuracy 99.6%, macro-F1 99.5%, weighted-F1 99.6%, ROC-AUC (OVR, macro) 0.9999. Full per-class precision/recall/F1 and the confusion matrix: [`reports/metrics/latest_metrics.json`](reports/metrics/latest_metrics.json) (or the app's Model Performance page).

These numbers are genuinely computed, not targeted — Logistic Regression and SVM visibly struggle relative to the boosted-tree models, which is exactly the kind of real differentiation you'd expect from a non-trivial multiclass problem. Two classes (Macrocytic anemia: 16 rows, Leukemia with thrombocytopenia: 10 rows) have very few examples; their per-class metrics carry wide uncertainty — see Limitations.

## Explainability

- **Global (model-wide):** permutation importance on the held-out test set, scored by macro-F1 drop when a feature is shuffled. HGB and MCV dominate (consistent with how anemia is clinically diagnosed from CBC indices); several features show ~zero importance for the selected model. [`reports/figures/global_feature_importance.json`](reports/figures/global_feature_importance.json).
- **Local (per-prediction):** SHAP explains every individual prediction — the app's results page and the API response both show the top contributing features with direction, not just a bare label.

## Data drift monitoring (offline demonstration)

`scripts/drift_report.py` runs a per-feature Kolmogorov-Smirnov test between two real feature distributions. There's no live production traffic to monitor, so this is explicitly a demonstration of the mechanism: by default it compares the train/test split (expected result: no drift, since both are samples of the same data), and `--classes "Healthy" "Iron deficiency anemia"` demonstrates it correctly flagging real drift (HGB and MCH show the strongest shift — consistent with clinical practice).

## Application

A 5-page Flask app (`app/`) plus a REST API:

| Page/Route | What it shows |
|---|---|
| `/` | Overview, how the pipeline works, key capabilities |
| `/predict` | Grouped CBC input form (Red Cell Indices / White Cells / Platelets), server + browser-side validation against real physiological ranges — no PII fields |
| `/predict` (POST) | Prediction, confidence, full class probabilities, SHAP top-contributing-features, input summary |
| `/performance` | Model comparison table, held-out metrics, confusion matrix, global feature importance, experiment log — all loaded from `reports/`, nothing hardcoded |
| `/methodology` | Dataset card, leakage check, evaluation strategy, drift monitoring, limitations |
| `POST /api/v1/predict` | Same prediction, as JSON, with structured validation errors |
| `GET /health`, `GET /api/v1/health` | Liveness checks |

No name/address/phone collected anywhere — only the 14 CBC values the model was actually trained on.

## Project structure

```
app/                        Flask app (factory + blueprint)
  routes.py                   Web pages + REST API
  services/prediction_service.py   Shared prediction+validation logic (used by both web and API)
  templates/, static/         5-page UI

src/
  data/                       load_data.py, validate_data.py (dedup, physiological-range checks, leakage check)
  features/                   feature_engineering.py (sklearn ColumnTransformer/Pipeline)
  models/                     train.py (benchmark+tune+select), evaluate.py, tuning.py
  explainability/             explain.py (permutation importance + SHAP)

data/
  raw/                        Source CSV, as downloaded
  processed/                  Cleaned dataset (generated)

models/                      model.joblib + metadata.json (versioned artifact)
reports/
  data_quality/                DATASET_CARD.md, data_quality_report.json
  metrics/                     latest_metrics.json, experiments.json (lightweight experiment log)
  figures/                     global_feature_importance.json, drift_report.json

scripts/drift_report.py      Offline drift-monitoring demonstration
tests/                        26 pytest tests: data validation, features, models, app, API
analysis/                     HISTORICAL — the original leakage diagnosis (see above)
```

## Running it locally

Requires Python 3.12+ (numpy 2.5 requires it; CI runs on 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.data.validate_data     # cleans data/raw -> data/processed, writes the data-quality report
python -m src.models.train           # benchmark -> tune -> held-out eval -> models/model.joblib
python -m src.explainability.explain # global feature importance report
python -m scripts.drift_report       # offline drift-monitoring demo

python -m pytest tests/ -v           # 26 tests
ruff check .                         # lint

python wsgi.py                       # http://localhost:5000 (set FLASK_DEBUG=true for the interactive debugger)
```

`models/model.joblib`, `metadata.json`, and everything in `reports/` are committed, so the app runs immediately after `pip install` without retraining. Re-run the pipeline yourself to verify every number in this README.

## API usage example

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"WBC":7.5,"LYMp":35,"NEUTp":55,"LYMn":2.5,"NEUTn":4.0,"RBC":5.0,"HGB":14.5,"HCT":43,"MCV":88,"MCH":29,"MCHC":33,"PLT":280,"PDW":15,"PCT":0.2}'
```

Returns prediction, confidence, full class probabilities, and the top contributing features.

## Testing

26 pytest tests across `tests/`: data-quality regression guards (duplicate detection, physiological-range validation, the single-feature leakage check), feature-engineering pipeline shape/scaling, evaluation metrics, the committed model artifact (loads, predicts, probabilities sum to 1), and the full Flask app + REST API (all pages, valid/invalid input, no PII fields). CI (`.github/workflows/tests.yml`) lints, re-validates the data, retrains from scratch, regenerates explainability, and runs the suite on every push.

## Limitations

- **Two classes have very few examples** (Macrocytic anemia: 16, Leukemia with thrombocytopenia: 10) — their per-class precision/recall carry wide uncertainty; read the confusion matrix's rows/columns for those classes accordingly.
- **Manually diagnosed, single-source data.** The dataset's own description states diagnoses were made manually from CBC values; this hasn't been cross-validated against a second, independent clinical dataset.
- **Physiologically-implausible-row filtering uses fixed reference ranges** (see `src/data/validate_data.py`) — generous enough to keep genuine extreme pathology, but a judgment call, not a formally validated clinical threshold set.
- **Not a diagnostic tool.** This reproduces patterns in a labeled dataset; it has not been through clinical validation and does not replace a clinician.

## Ethical & medical disclaimer

This application is an educational machine learning project and is not a medical diagnostic tool or a substitute for professional healthcare advice.

## Tech stack

Python, scikit-learn, SHAP, pandas, Flask, pytest, ruff, GitHub Actions.
