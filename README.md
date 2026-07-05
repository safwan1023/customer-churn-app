# 📉 Customer Churn Prediction & Retention Engine

🔗 **Live Demo:** https://customer-churn-app-cjwyflqv7j6jjgrjuqmfrd.streamlit.app/

An end-to-end, production-style machine learning system that predicts which telecom
customers are likely to churn, explains *why* using SHAP, and serves predictions
through both a REST API and an interactive web app.

Built on the real-world **IBM Telco Customer Churn** dataset (7,043 customers, 20 features).

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Findings](#key-findings-from-eda)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Results](#model-comparison-results)
- [Explainability (SHAP)](#explainability-shap)
- [API Reference](#api-reference)
- [Running Locally](#running-locally)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Telecom companies typically lose 20-30% of customers annually. Acquiring a new
customer costs 5-7x more than retaining an existing one. This project builds a
system that flags at-risk customers *before* they leave, with enough explainability
that a retention team can act on the prediction rather than just trust a black box.

## Key Findings from EDA

- **`TotalCharges`** was stored as a string with 11 hidden blank values — all
  brand-new (`tenure=0`) customers. Fixed via coercion + explicit zero-fill,
  not dropped or mean-imputed.
- Target is imbalanced: **73.5% retained / 26.5% churned.**
- **Contract type is the strongest churn driver**: Month-to-month customers churn
  at 42.7% vs. 2.8% for Two-year contracts.
- **Fiber optic internet** customers churn more than twice as often as DSL
  (41.9% vs 19%), likely tied to price sensitivity.
- Churn is front-loaded: 52.9% in months 0-6, dropping to 9.5% by months 49-72.
- These same four signals were **independently rediscovered** by every model's
  feature importances and confirmed by SHAP — strong evidence the pipeline isn't
  leaking information or fitting noise.

## Architecture

```
Raw CSV → Cleaning → Feature Engineering → Train/Test Split (stratified)
   → ColumnTransformer (Scale + One-Hot Encode)
   → 5 Models Compared (5-fold CV) → Best Model Tuned (RandomizedSearchCV)
   → SHAP Explainability → Single Joblib Pipeline Artifact
        ↓                                    ↓
   FastAPI (/predict REST API)      Streamlit (interactive UI)
                    ↓
        Streamlit Community Cloud (public deployment)
```

The preprocessor and the trained classifier are bundled into **one `sklearn.Pipeline`**
and saved as a single `.joblib` file. Both the API and the web app load this same
artifact, guaranteeing identical scaling/encoding behavior in every environment —
no risk of a mismatched scaler/model pair in production.

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Features | pandas, numpy, scikit-learn |
| Modeling | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Tuning | RandomizedSearchCV, StratifiedKFold cross-validation |
| Explainability | SHAP (TreeExplainer) |
| Serialization | joblib |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
customer-churn-app/
├── .github/workflows/ci.yml       # CI: install deps, run tests
├── api/
│   └── main.py                    # FastAPI backend
├── app/
│   └── streamlit_app.py           # Streamlit frontend
├── data/
│   ├── raw/                       # Original, untouched dataset
│   └── processed/                 # Cleaned output
├── models/
│   └── churn_pipeline.joblib      # Final trained pipeline (preprocessor + model)
├── reports/figures/                # EDA charts, SHAP summary plot
├── src/
│   ├── data/                      # load_data.py, preprocess.py, eda.py
│   ├── features/                  # build_features.py (split + ColumnTransformer)
│   ├── models/                    # train_all_models.py, tune_model.py
│   └── explain/                   # explain_model.py (SHAP)
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Model Comparison Results

5-fold stratified cross-validation on the training set, evaluated on a held-out
20% test set (never touched during training or tuning):

| Model | CV F1 (mean) | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0.634 | 0.540 | 0.770 | 0.635 | 0.846 |
| XGBoost | 0.629 | 0.524 | 0.786 | 0.629 | 0.834 |
| Decision Tree | 0.605 | 0.504 | 0.818 | 0.624 | 0.826 |
| Logistic Regression | 0.628 | 0.498 | 0.791 | 0.611 | 0.842 |
| Gradient Boosting | 0.581 | 0.650 | 0.505 | 0.568 | 0.838 |

Random Forest was selected and tuned further via `RandomizedSearchCV`
(15 combinations × 5-fold CV), reaching a **tuned CV F1 of 0.642** and
**recall of 0.79 on the Churn class** on the final held-out test set —
meaning the deployed model catches ~4 out of 5 customers who actually churn.

`class_weight="balanced"` was used deliberately: on a 73.5/26.5 imbalanced
target, optimizing for raw accuracy would reward a model for ignoring churners
entirely. Recall on the minority class was prioritized instead, since missing
an at-risk customer is costlier to the business than one unnecessary retention offer.

## Explainability (SHAP)

Top features by mean absolute SHAP value on the tuned model:

1. `InternetService_Fiber optic`
2. `tenure`
3. `Contract_Two year`
4. `PaymentMethod_Electronic check`
5. `TotalCharges` / `MonthlyCharges`

These match the EDA findings exactly, and were derived completely independently
by the model — nothing was hard-coded to produce this alignment.

See `reports/figures/shap_summary.png` for the full plot.

## API Reference

Interactive Swagger docs available at `/docs` when running locally.

```
GET  /            → health check message
GET  /health       → {"status": "ok", "model_loaded": true}
POST /predict       → {customer features JSON} -> {churn_prediction, churn_probability, risk_level}
```

Example request body:
```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 90.5, "TotalCharges": 180.0
}
```

## Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/safwan1023/customer-churn-app.git
cd customer-churn-app
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Reproduce the pipeline (optional — churn_pipeline.joblib is already committed)
python src/data/preprocess.py
python src/models/train_all_models.py
python src/models/tune_model.py
python src/explain/explain_model.py

# 3. Run the API
uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs

# 4. Run the Streamlit app (in a separate terminal)
streamlit run app/streamlit_app.py
# Visit http://localhost:8501
```

## Testing

```bash
pytest tests/ -v
```

Covers: the `TotalCharges` cleaning fix, target encoding correctness, end-to-end
cleaning pipeline integrity (no nulls, no leaked ID column), and stratified
split/preprocessing behavior.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs dependencies,
runs the full pytest suite, and lints the codebase — catching broken cleaning
logic or import errors before they reach `main`.

## Future Improvements

- Real-time feature drift monitoring (Evidently AI) with scheduled retraining
- Move from a single joblib artifact to an MLflow model registry with staged promotion
- Causal/uplift modeling to estimate the effect of specific retention offers,
  not just churn probability
- A batch prediction endpoint (`/predict/batch`) for scoring an uploaded CSV
- Migrate deployment from Streamlit Community Cloud to a containerized
  (Docker + AWS/Azure) setup for finer infrastructure control





