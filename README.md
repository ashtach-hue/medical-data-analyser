# Medical Data Analyser

A comprehensive healthcare analytics platform for processing patient data and visualizing health trends over time using ML-based analysis.

## 🎯 Core Features

- **Patient Data Processing**: ETL pipelines for patient health records and time-series data
- **Trend Analysis**: Identify patterns and trends in patient health metrics over time
- **Data Visualization**: Interactive charts and heatmaps showing health trends
- **Statistical Analysis**: Comprehensive statistical analysis of health metrics
- **Disease Risk Prediction**: Machine learning models for disease risk assessment
- **API Integration**: RESTful API for data queries and real-time analysis
- **Patient Insights**: Personalized health recommendations based on trends

## 🏗️ Project Structure

```
medical-data-analyser/
├── data/                          # Data files and datasets
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed datasets
│   └── external/                 # External data sources
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_processing/          # Data ETL and cleaning
│   ├── trends/                   # Trend analysis module
│   ├── models/                   # ML models for predictions
│   ├── analysis/                 # Statistical analysis
│   ├── visualization/            # Data visualization utilities
│   ├── api/                      # FastAPI application
│   └── utils/                    # Helper functions
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit and integration tests
├── config/                       # Configuration files
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variables template
├── Dockerfile                    # Docker containerization
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # This file
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ashtach-hue/medical-data-analyser.git
cd medical-data-analyser
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Process Patient Data and Analyze Trends

```python
from src.data_processing import DataProcessor
from src.trends import TrendAnalyzer
from src.visualization import HealthVisualizer

# Load and clean patient data
processor = DataProcessor()
df = processor.load_and_clean('data/raw/patients.csv')

# Analyze trends over time
trend_analyzer = TrendAnalyzer()
trends = trend_analyzer.analyze_patient_trends(df)

# Visualize the trends
visualizer = HealthVisualizer()
visualizer.plot_trends(trends)
visualizer.plot_trend_heatmap(df)
```

### Run the API Server

```bash
# Start the FastAPI server
python -m src.api.main

# The API will be available at http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

## 📖 Usage Examples

### 1. Patient Data Processing
```python
from src.data_processing import DataProcessor

processor = DataProcessor()

# Load and clean data
df = processor.load_and_clean('data/raw/patients.csv')

# Split into train/test sets
train_df, test_df = processor.train_test_split(df, test_size=0.2)

# Handle missing values
df_cleaned = processor.handle_missing_values(df)
```

### 2. Trend Analysis
```python
from src.trends import TrendAnalyzer

analyzer = TrendAnalyzer()

# Analyze patient trends
trends = analyzer.analyze_patient_trends(df)

# Calculate growth rates
growth_rates = analyzer.calculate_growth_rates(df, 'blood_pressure')

# Detect trend changes
anomalies = analyzer.detect_trend_changes(df, 'cholesterol')

# Get statistical summary
summary = analyzer.get_trend_summary(df)
```

### 3. Visualization
```python
from src.visualization import HealthVisualizer

visualizer = HealthVisualizer()

# Plot trends over time
visualizer.plot_trends(trends_data)

# Create trend heatmap
visualizer.plot_trend_heatmap(df)

# Analyze correlation
visualizer.plot_correlation_matrix(df)

# Save visualizations
visualizer.save_figure('trends.png')
```

### 4. Statistical Analysis
```python
from src.analysis import HealthAnalysis

analysis = HealthAnalysis()

# Comprehensive analysis
results = analysis.analyze(df)

# Correlations
corr = analysis.get_correlations(df)

# Outlier detection
outliers = analysis.identify_outliers(df, 'blood_pressure')
```

### 5. Disease Prediction
```python
from src.models import DiseasePredictor

predictor = DiseasePredictor(model_type='ensemble')

# Train model
predictor.fit(X_train, y_train)

# Make predictions
predictions = predictor.predict(X_test)

# Feature importance
importance = predictor.get_feature_importance()
```

### 6. API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get trends for a patient
curl http://localhost:8000/trends/patient/123

# Analyze multiple patients
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"patient_ids": [1, 2, 3]}'

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_data": {"age": 45, "blood_pressure": 130, ...}}'

# Get API documentation
curl http://localhost:8000/docs
```

## 🔧 Technologies

- **Data Processing**: Pandas, NumPy
- **Time Series Analysis**: Statsmodels, SciPy
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: FastAPI, Uvicorn, Pydantic
- **Testing**: Pytest, Pytest-cov
- **Development**: Jupyter, IPython
- **Database**: SQLAlchemy, PostgreSQL
- **Code Quality**: Black, Flake8, MyPy

## 🐛 Troubleshooting

### Common Issues

**Dependency conflicts during installation:**
```bash
pip cache purge
pip install -r requirements.txt --force-reinstall
```

**Module import errors:**
```bash
# Ensure you're in the project root directory
cd medical-data-analyser

# Verify virtual environment is activated
source venv/bin/activate

# Install in editable mode
pip install -e .
```

**API server won't start:**
- Check if port 8000 is available
- Verify all dependencies: `pip install -r requirements.txt`
- Check logs: `tail -f logs/app.log`

**Trend analysis issues:**
- Ensure data has timestamp column: `df['date']` or `df['timestamp']`
- Check data types: `df.info()`
- Verify sufficient data points for meaningful trends

**For additional help:**
```bash
# Provide this information when opening an issue:
python --version
pip list | grep -E "pandas|numpy|scikit-learn"
head -20 data/raw/your_data.csv
```

## 📊 Data Format

Expected patient data CSV format:

```
patient_id,date,age,blood_pressure,cholesterol,heart_rate,glucose,weight
1,2024-01-01,45,120,200,72,100,75.5
1,2024-01-08,45,122,202,71,101,75.6
2,2024-01-01,52,135,220,78,110,82.3
2,2024-01-08,52,133,218,77,109,82.1
```

Key columns:
- `patient_id`: Unique patient identifier
- `date`: Measurement date (YYYY-MM-DD format)
- Health metrics: Any numeric health measurements
- Other demographic data: Age, weight, etc.

## 🤝 Contributing

Contributions are welcome! Please:
1. Create a new branch for your feature (`git checkout -b feature/my-feature`)
2. Write tests for new functionality
3. Follow PEP 8 style guidelines
4. Submit a pull request with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Contact the maintainers directly

---

**Happy analyzing! 📈**
