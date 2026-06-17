"""FastAPI application for Medical Data Analyser with trend endpoints."""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Data Analyser API",
    description="Healthcare analytics platform with ML-based disease prediction and trend analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Request/Response Models
class PatientData(BaseModel):
    """Patient data model."""
    age: int
    gender: str
    blood_pressure: Optional[int] = None
    cholesterol: Optional[int] = None
    heart_rate: Optional[int] = None
    glucose: Optional[int] = None


class TrendData(BaseModel):
    """Trend data model."""
    metric: str
    values: List[float]
    dates: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: str
    confidence: float
    risk_level: str


class TrendResponse(BaseModel):
    """Trend response model."""
    metric: str
    trend_direction: str
    trend_strength: float
    latest_value: float
    forecast: Optional[List[float]] = None


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    status: str
    total_records: int
    summary: Dict[str, Any]


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Medical Data Analyser API",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict",
            "trends": "/trends",
            "trends_patient": "/trends/patient/{patient_id}",
            "analyze": "/analyze",
            "compare_patients": "/compare/patients"
        }
    }


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Medical Data Analyser API",
        "version": "0.1.0"
    }


# Prediction Endpoints
@app.post("/predict")
async def predict(patient_data: PatientData) -> PredictionResponse:
    """Make disease prediction for patient data.
    
    Args:
        patient_data: Patient information
        
    Returns:
        Prediction response
    """
    try:
        # Placeholder prediction logic
        # In production, this would load and use the trained model
        
        risk_score = (patient_data.blood_pressure or 0) / 100 if patient_data.blood_pressure else 0.3
        
        if risk_score < 0.5:
            prediction = "low_risk"
            risk_level = "Low"
            confidence = 0.85
        elif risk_score < 0.75:
            prediction = "moderate_risk"
            risk_level = "Moderate"
            confidence = 0.72
        else:
            prediction = "high_risk"
            risk_level = "High"
            confidence = 0.78
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# Trend Analysis Endpoints
@app.post("/trends")
async def analyze_trends(trend_data: List[TrendData]) -> Dict[str, Any]:
    """Analyze trends in health metrics.
    
    Args:
        trend_data: List of trend data
        
    Returns:
        Trend analysis results
    """
    try:
        results = {}
        
        for trend in trend_data:
            values = np.array(trend.values)
            
            # Calculate trend statistics
            if len(values) > 1:
                x = np.arange(len(values))
                from scipy import stats as sp_stats
                slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, values)
                
                direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                
                results[trend.metric] = {
                    'direction': direction,
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'latest_value': float(values[-1])
                }
        
        return {"status": "success", "trends": results}
        
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Trend analysis failed")


@app.get("/trends/patient/{patient_id}")
async def get_patient_trends(patient_id: str) -> Dict[str, Any]:
    """Get trends for a specific patient.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Patient trend data
    """
    try:
        # Placeholder - In production, fetch from database
        return {
            "patient_id": patient_id,
            "trends": {
                "blood_pressure": {
                    "direction": "increasing",
                    "values": [120, 122, 125, 127, 130]
                },
                "cholesterol": {
                    "direction": "stable",
                    "values": [200, 202, 201, 203, 202]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error fetching patient trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch patient trends")


# Analysis Endpoints
@app.post("/analyze")
async def analyze(data: Dict[str, Any]) -> AnalysisResponse:
    """Analyze health data.
    
    Args:
        data: Health data to analyze
        
    Returns:
        Analysis results
    """
    try:
        # Placeholder analysis logic
        return AnalysisResponse(
            status="success",
            total_records=len(data.get('records', [])),
            summary={
                "message": "Analysis completed",
                "records_processed": len(data.get('records', []))
            }
        )
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


# Comparison Endpoints
@app.post("/compare/patients")
async def compare_patients(patient_ids: List[str] = Query(...), 
                          metric: str = Query("blood_pressure")) -> Dict[str, Any]:
    """Compare a metric across multiple patients.
    
    Args:
        patient_ids: List of patient IDs
        metric: Metric to compare
        
    Returns:
        Comparison results
    """
    try:
        comparison = {}
        
        for patient_id in patient_ids:
            # Placeholder - In production, fetch from database
            comparison[patient_id] = {
                "metric": metric,
                "mean_value": 125.5,
                "trend": "increasing"
            }
        
        return {
            "status": "success",
            "metric": metric,
            "comparison": comparison
        }
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail="Comparison failed")


# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status": "error",
        "message": "An unexpected error occurred"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
