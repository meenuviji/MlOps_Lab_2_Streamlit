"""
FastAPI Backend for Loan Approval Prediction
Modified version of the Iris prediction API
Place this in FastAPI_Labs/src/ directory
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval using machine learning",
    version="1.0.0"
)

# Load model and scaler at startup
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Loan model loaded successfully")
    
    with open('loan_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
    
    with open('loan_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    logger.info("Model metadata loaded successfully")
    
except FileNotFoundError as e:
    logger.error(f"Model files not found: {e}")
    logger.error("Please run train_loan_model.py first!")
    model = None
    scaler = None
    metadata = {}

# Define input schema with validation
class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age (18-100)")
    income: float = Field(..., gt=0, description="Annual income in dollars")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., ge=12, le=480, description="Loan term in months")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_length: float = Field(..., ge=0, description="Years of employment")
    home_ownership: int = Field(..., ge=0, le=2, description="0=Rent, 1=Own, 2=Mortgage")
    purpose: int = Field(..., ge=0, le=4, description="Loan purpose code")
    dti_ratio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    has_cosigner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    previous_defaults: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000,
                "loan_amount": 200000,
                "loan_term": 240,
                "credit_score": 720,
                "employment_length": 5,
                "home_ownership": 1,
                "purpose": 0,
                "dti_ratio": 0.35,
                "has_cosigner": 0,
                "previous_defaults": 0
            }
        }

# Response models
class PredictionResponse(BaseModel):
    response: int
    approval_status: str
    probability: float
    confidence: str
    risk_factors: list
    recommendations: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    timestamp: str

# Root endpoint
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Predict loan approval",
            "/model-info": "Model information",
            "/docs": "API documentation"
        }
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/model-info", tags=["Model"])
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return metadata

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    """
    Predict loan approval based on application details
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist."
        )
    
    try:
        # Prepare input data
        input_data = np.array([[
            application.age,
            application.income,
            application.loan_amount,
            application.loan_term,
            application.credit_score,
            application.employment_length,
            application.home_ownership,
            application.purpose,
            application.dti_ratio,
            application.has_cosigner,
            application.previous_defaults
        ]])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Determine approval probability
        approval_probability = float(probability[1])
        
        # Determine confidence level
        if approval_probability > 0.8 or approval_probability < 0.2:
            confidence = "High"
        elif approval_probability > 0.6 or approval_probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Identify risk factors
        risk_factors = []
        if application.credit_score < 650:
            risk_factors.append("Low credit score")
        if application.dti_ratio > 0.43:
            risk_factors.append("High debt-to-income ratio")
        if application.previous_defaults == 1:
            risk_factors.append("Previous loan defaults")
        if application.employment_length < 2:
            risk_factors.append("Short employment history")
        if application.income < application.loan_amount * 0.3:
            risk_factors.append("Low income relative to loan amount")
        
        # Generate recommendations
        recommendations = []
        if prediction == 0:  # Rejected
            if application.credit_score < 650:
                recommendations.append("Improve credit score to at least 650")
            if application.dti_ratio > 0.43:
                recommendations.append("Reduce existing debts to lower DTI ratio")
            if application.has_cosigner == 0:
                recommendations.append("Consider adding a co-signer")
            if application.loan_amount > application.income * 3:
                recommendations.append("Consider a smaller loan amount")
        else:  # Approved
            recommendations.append("Congratulations! Maintain good payment history")
            recommendations.append("Consider automatic payments to avoid late fees")
        
        return PredictionResponse(
            response=int(prediction),
            approval_status="Approved" if prediction == 1 else "Rejected",
            probability=approval_probability,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(applications: list[LoanApplication]):
    """
    Predict loan approval for multiple applications
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist."
        )
    
    results = []
    for app in applications:
        try:
            # Use the single prediction logic
            result = predict(app)
            results.append({
                "application_id": f"{app.age}_{app.income}_{app.loan_amount}",
                "result": result.dict()
            })
        except Exception as e:
            results.append({
                "application_id": f"{app.age}_{app.income}_{app.loan_amount}",
                "error": str(e)
            })
    
    return {"predictions": results}

# Statistics endpoint
@app.get("/stats", tags=["Model"])
def get_statistics():
    """
    Get model statistics and performance metrics
    """
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return {
        "model_accuracy": metadata.get("accuracy", "N/A"),
        "training_approval_rate": metadata.get("approval_rate", "N/A"),
        "training_samples": metadata.get("training_samples", "N/A"),
        "top_features": metadata.get("feature_importance", [])[:5]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)