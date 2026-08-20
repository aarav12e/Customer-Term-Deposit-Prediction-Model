"""
FastAPI backend for the Term Deposit Subscription Predictor.

Loads the sklearn Pipeline (preprocessing + model) saved from the notebook
and exposes a single /predict endpoint that the React frontend calls.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Term Deposit Predictor API")

# Allow the React frontend (any origin during dev; lock this down in prod by
# setting ALLOWED_ORIGIN as an env var on Render to your deployed frontend URL)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "term_deposit_model.joblib")
model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        # Don't crash the server on startup — /predict will report the error instead,
        # which makes the Render deploy logs much easier to debug.
        print(f"WARNING: model file not found at {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Request / response schema
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str  # "yes" / "no"
    balance: int
    housing: str  # "yes" / "no"
    loan: str     # "yes" / "no"
    contact: str
    day: int = Field(..., ge=1, le=31)
    month: str
    campaign: int = Field(..., ge=1)
    never_contacted: bool
    pdays: int = 30       # ignored if never_contacted is True
    previous: int = 0
    poutcome: str


class PredictionResponse(BaseModel):
    prediction: str
    probability_percent: float
    label: str  # "yes" or "no" — convenient for the frontend to branch on


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Make sure term_deposit_model.joblib is present next to main.py."
        )

    pdays_value = -1 if customer.never_contacted else customer.pdays
    was_contacted_before = 0 if customer.never_contacted else 1

    row = pd.DataFrame([{
        "age": customer.age,
        "job": customer.job,
        "marital": customer.marital,
        "education": customer.education,
        "default": customer.default,
        "balance": customer.balance,
        "housing": customer.housing,
        "loan": customer.loan,
        "contact": customer.contact,
        "day": customer.day,
        "month": customer.month,
        "campaign": customer.campaign,
        "pdays": pdays_value,
        "previous": customer.previous,
        "poutcome": customer.poutcome,
        "was_contacted_before": was_contacted_before,
    }])

    proba = float(model.predict_proba(row)[0][1])
    label = "yes" if proba >= 0.5 else "no"
    prediction = "Likely to subscribe" if label == "yes" else "Unlikely to subscribe"

    return PredictionResponse(
        prediction=prediction,
        probability_percent=round(proba * 100, 1),
        label=label,
    )
