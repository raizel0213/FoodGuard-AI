# FoodGuard AI

Food-delivery problem detection and root-cause analysis. FoodGuard predicts whether an
Indian food-delivery order will report a delivery issue **before dispatch**, then explains
the probable contributing factors and recommends corrective actions.

> Not a mockup: a real, deterministic ML pipeline (6 models compared by cross-validation)
> wired into a 6-page Streamlit dashboard. Every number shown in the app is computed from
> data - nothing is hardcoded, and performance is reported honestly.

## Quickstart

```bash
cd foodguard-ai

# 1. (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train the models (optional - the app also trains automatically on first launch)
python -m src.train

# 4. start the dashboard
streamlit run app.py             # opens http://localhost:8501
```

If `streamlit` is not on your PATH, use `python -m streamlit run app.py`.

## What you get

| Page | What it shows |
| --- | --- |
| Overview Dashboard | Scored dataset KPIs, issue rate by city / traffic / rainfall, aggregate drivers |
| Order Analysis | Single-order scoring: probability gauge, risk level, probable contributing factors, recommendations |
| Model Performance | CV comparison of 6 models, confusion matrix, ROC / PR / calibration curves, permutation importance |
| Root Cause Analysis | Aggregate drivers of flagged orders + evidence tables (observed issue rates per condition) |
| Data Explorer | Filterable raw data, distributions, box plots, correlation matrix |
| About | Methodology, leakage policy, dataset attribution, run instructions |

## Methodology (short version)

- **Data**: 1,000 orders, target `delivery_issue_reported` (12.4% issue rate).
- **Leakage-safe features only** (available at dispatch time): city, distance, order value,
  traffic index, rainfall, prep estimate, queue length, rider experience, peak/weekend
  flags, vehicle type. Excluded: `order_id` (identifier) and `delivery_time_minutes`
  (post-event - using it would leak the outcome).
- **Models**: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
  Hist Gradient Boosting, XGBoost - compared with 5-fold stratified CV; best mean ROC-AUC
  is selected and refit on the full training split.
- **Imbalance**: balanced class weights / sample weights, applied on training folds only.
- **Threshold**: tuned for max F1 on out-of-fold training predictions (never on the test set).
- **Risk levels**: Low < 10% < Medium < 20% < High < 35% < Critical.
- **Explanations**: SHAP for tree models, with a transparent reference-delta fallback
  (probability drop when a feature is reset to its training baseline). Aggregates always
  use the reference-delta method. Output wording is deliberately "probable contributing
  factors" - the dataset has no labels for *why* issues occurred.
- **Recommendations**: a rule engine that cites the empirical issue rate of the order's
  condition (from training-data evidence tables) versus the overall base rate.

## Honest performance note (and the 95% question)

A 95%+ accuracy target was tested seriously before building the app: 8 model families
(including tuned XGBoost, Random Forest, HistGB, soft-voting and stacking ensembles) x 2
feature sets (dispatch-time only, and dispatch-time + post-event `delivery_time_minutes`),
all evaluated with 5-fold stratified CV. Result: **the honest ceiling on this dataset is
~88% accuracy / ~0.68 ROC-AUC** - barely above the 87.6% majority baseline, because the
label is close to noise relative to the features (max correlation ~0.22; even the slowest
delivery-time decile shows only a 30% issue rate). Any tool claiming 95%+ on this exact CSV
is reporting training-set accuracy, leaking the label, or fabricating numbers.

FoodGuard reports exactly what the selected model achieves on the untouched holdout
(printed by `python -m src.train`, shown on the Model Performance page) and trades a little
accuracy for real issue recall, which the baseline never delivers. Full numbers:
`reports/evaluation_report.json`.

## Dataset attribution

`data/IndianFoodDelivery.csv` comes from the **YBI Foundation** Machine Learning collection
(public educational dataset):
https://raw.githubusercontent.com/YBIFoundation/MachineLearning/refs/heads/main/Dataset/IndianFoodDelivery.csv

## Project structure

```
foodguard-ai/
├── app.py                     # entry point (streamlit run app.py)
├── config.py                  # paths, features, risk bands, theme
├── pages/                     # 6 dashboard pages (auto-discovered by Streamlit)
├── src/
│   ├── boot.py                # app bootstrap: auto-train + cached artefacts
│   ├── data_loader.py         # load + validate CSV
│   ├── preprocessing.py       # impute / scale / one-hot (leakage-safe)
│   ├── train.py               # CV, model selection, artifacts  (python -m src.train)
│   ├── predict.py             # FoodGuardPredictor service (validation, risk bands)
│   ├── root_cause.py          # SHAP + reference-delta explanations
│   ├── recommendations.py     # evidence-based action engine
│   └── ui.py                  # navy/cyan theme helpers
├── data/IndianFoodDelivery.csv
├── models/foodguard_model.joblib   # created by training
└── reports/                        # model_comparison.csv, evaluation_report.json, feature_importance.csv
```

## Deployment

- **Streamlit Community Cloud**: push this folder to GitHub, select `app.py` as the entry
  point, add `requirements.txt` when prompted. The app trains on first launch, or commit the
  `models/` artefact to skip that step.
- **Docker**:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```
  then `docker build -t foodguard-ai . && docker run -p 8501:8501 foodguard-ai`.
- **Any VM/PaaS**: `pip install -r requirements.txt`, `python -m src.train`,
  `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.

## Troubleshooting

**`AttributeError: module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'` (or any other error inside `joblib.load`)**

The pre-trained `models/foodguard_model.joblib` was pickled with a different
scikit-learn version than the one installed in your environment. Pickled
scikit-learn estimators only load reliably with the version that trained
them, so this is expected after upgrading Python/scikit-learn (e.g. Python
3.14 with a very recent scikit-learn).

Fix - retrain the model with your own scikit-learn (takes well under a minute):

```bash
python -m src.train
streamlit run app.py
```

Since the current version of `src/boot.py` this is handled automatically:
if the bundled model cannot be unpickled, or was trained with a different
scikit-learn version, the app detects it at startup and retrains silently
with your local version. Deleting `models/foodguard_model.joblib` and
restarting the app has the same effect.

**Other first-run problems**

- `ModuleNotFoundError: xgboost` -> the trainer skips XGBoost automatically if
  it is not installed; no action needed.
- Port already in use -> `streamlit run app.py --server.port 8502`.

## Limitations

- Small dataset (1,000 rows) -> wide uncertainty on every metric.
- No labels for why an issue occurred, so the explanation layer describes model behaviour,
  not verified business causes.
- Retrain periodically as city operations, traffic and weather patterns drift.
