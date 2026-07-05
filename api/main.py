from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

MODEL_PATH = "models/churn_pipeline.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability from account/service features",
    version="1.0.0",
)

pipeline = joblib.load(MODEL_PATH)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: float = Field(..., examples=[845.5])


def add_tenure_group(tenure: int) -> str:
    if tenure <= 12:
        return "0-12mo"
    elif tenure <= 24:
        return "13-24mo"
    elif tenure <= 48:
        return "25-48mo"
    else:
        return "49-72mo"


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. See /docs for the interactive Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    data = customer.model_dump()
    data["tenure_group"] = add_tenure_group(data["tenure"])

    input_df = pd.DataFrame([data])

    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)

    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
    }