"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "endpoints" in response.json()

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict(self):
        """Test prediction endpoint."""
        patient_data = {
            "age": 45,
            "gender": "M",
            "blood_pressure": 130,
            "cholesterol": 220,
            "heart_rate": 75,
            "glucose": 105
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "risk_level" in data

    def test_analyze_trends(self):
        """Test trend analysis endpoint."""
        trend_data = [
            {
                "metric": "blood_pressure",
                "values": [120, 122, 125, 127, 130]
            },
            {
                "metric": "cholesterol",
                "values": [200, 202, 205, 203, 207]
            }
        ]
        
        response = client.post("/trends", json=trend_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "trends" in data
        assert "blood_pressure" in data["trends"]

    def test_get_patient_trends(self):
        """Test patient trends endpoint."""
        response = client.get("/trends/patient/1")
        assert response.status_code == 200
        
        data = response.json()
        assert "patient_id" in data
        assert "trends" in data

    def test_analyze(self):
        """Test analysis endpoint."""
        data = {
            "records": [
                {"age": 45, "blood_pressure": 130},
                {"age": 46, "blood_pressure": 132}
            ]
        }
        
        response = client.post("/analyze", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
        assert result["total_records"] == 2

    def test_compare_patients(self):
        """Test patient comparison endpoint."""
        response = client.get("/compare/patients?patient_ids=1&patient_ids=2&patient_ids=3&metric=blood_pressure")
        assert response.status_code == 200
        
        data = response.json()
        assert "comparison" in data
        assert "metric" in data
