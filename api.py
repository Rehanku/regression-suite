from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Regression Suite — Ames Housing Price Prediction",
    description="Predicts house sale prices using a Ridge Regression model trained on the Ames Housing dataset.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_pipeline.joblib")
pipeline = joblib.load(MODEL_PATH)

# Extract coefficients and feature names once at startup
def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps['preprocessor']
    numeric_features = preprocessor.transformers_[0][2]
    ohe = preprocessor.named_transformers_['categorical'].named_steps['encoder']
    categorical_features = preprocessor.transformers_[1][2]
    ohe_names = list(ohe.get_feature_names_out(categorical_features))
    return list(numeric_features) + ohe_names

try:
    FEATURE_NAMES = get_feature_names(pipeline)
    COEFFICIENTS = pipeline.named_steps['model'].coef_
except Exception:
    FEATURE_NAMES = []
    COEFFICIENTS = []

class HouseFeatures(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10)
    TotalSF: float = Field(..., ge=0, le=15000)
    GrLivArea: float = Field(..., ge=0, le=6000)
    GarageCars: int = Field(..., ge=0, le=5)
    TotalBsmtSF: float = Field(..., ge=0, le=6000)
    FullBath: int = Field(..., ge=0, le=5)
    YearBuilt: int = Field(..., ge=1872, le=2010)
    HouseAge: int = Field(..., ge=0, le=150)
    RemodAge: int = Field(..., ge=0, le=100)
    HasGarage: int = Field(..., ge=0, le=1)
    HasBsmt: int = Field(..., ge=0, le=1)
    Has2ndFloor: int = Field(..., ge=0, le=1)
    Neighborhood: str
    MSZoning: str
    SaleCondition: str
    BldgType: str
    HouseStyle: str
    RoofStyle: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "OverallQual": 7, "TotalSF": 2800, "GrLivArea": 1800,
                "GarageCars": 2, "TotalBsmtSF": 1000, "FullBath": 2,
                "YearBuilt": 2003, "HouseAge": 7, "RemodAge": 7,
                "HasGarage": 1, "HasBsmt": 1, "Has2ndFloor": 1,
                "Neighborhood": "CollgCr", "MSZoning": "RL",
                "SaleCondition": "Normal", "BldgType": "1Fam",
                "HouseStyle": "2Story", "RoofStyle": "Gable"
            }]
        }
    }

class Contribution(BaseModel):
    feature: str
    dollar_impact: float
    direction: str

class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    price_formatted: str
    confidence_note: str
    contributions: list[Contribution]
    model_info: dict

@app.get("/health")
def health():
    return {"status": "ok", "model": "Ridge Regression", "dataset": "Ames Housing"}

@app.get("/")
def root():
    return {"message": "Ames Housing Price Prediction API", "docs": "/docs", "health": "/health", "predict": "POST /predict"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        input_df = pd.DataFrame([features.model_dump()])

        # Get preprocessed features
        preprocessor = pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(input_df)

        # Predict in log space
        log_prediction = pipeline.named_steps['model'].predict(X_transformed)[0]
        predicted_price = float(np.exp(log_prediction))

        # Confidence interval
        rmse_log = 0.128
        price_low = float(np.exp(log_prediction - 1.96 * rmse_log))
        price_high = float(np.exp(log_prediction + 1.96 * rmse_log))

        # Feature contributions
        # contribution_i = coef_i * x_i (in log space)
        # dollar impact = exp(base + coef_i * x_i) - exp(base)
        contributions = []
        if len(COEFFICIENTS) > 0:
            coef_contributions = COEFFICIENTS * X_transformed[0]

            # Map back to readable names — only numeric features for display
            numeric_features = preprocessor.transformers_[0][2]
            numeric_transformed = preprocessor.named_transformers_['numeric'].transform(
                input_df[numeric_features]
            )

            numeric_coefs = COEFFICIENTS[:len(numeric_features)]
            numeric_contribs = numeric_coefs * numeric_transformed[0]

            base_log = log_prediction - np.sum(coef_contributions)

            for i, fname in enumerate(numeric_features):
                contrib_log = numeric_contribs[i]
                # Dollar impact: how much does this feature add/subtract
                dollar_impact = float(np.exp(log_prediction) - np.exp(log_prediction - contrib_log))
                contributions.append({
                    "feature": fname,
                    "dollar_impact": round(dollar_impact, 0),
                    "direction": "positive" if dollar_impact > 0 else "negative"
                })

            # Sort by absolute impact, take top 5
            contributions.sort(key=lambda x: abs(x["dollar_impact"]), reverse=True)
            contributions = contributions[:5]

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            price_range_low=round(price_low, 2),
            price_range_high=round(price_high, 2),
            price_formatted=f"${predicted_price:,.0f}",
            confidence_note=f"95% confidence interval: ${price_low:,.0f} — ${price_high:,.0f}",
            contributions=[Contribution(**c) for c in contributions],
            model_info={
                "model": "Ridge Regression",
                "cv_r2": 0.8757,
                "test_r2": 0.8841,
                "dataset": "Ames Housing — 1,429 samples after cleaning"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))