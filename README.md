# Medical Data Analyser

A comprehensive healthcare analytics platform with ML-based disease prediction using patient health records and genomic data.

## Features

- **Data Processing**: ETL pipelines for patient health records and genomic data
- **Disease Prediction**: Machine learning models for disease risk assessment
- **Data Analysis**: Statistical analysis and visualization of health metrics
- **Patient Insights**: Personalized health recommendations
- **API Integration**: RESTful API for data queries and predictions

## Project Structure

```
medical-data-analyser/
├── data/                          # Data files and datasets
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed datasets
│   └── external/                 # External data sources
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_processing/          # Data ETL and cleaning
│   ├── feature_engineering/      # Feature creation and selection
│   ├── models/                   # ML models for predictions
│   ├── analysis/                 # Statistical analysis
│   ├── visualization/            # Data visualization utilities
│   └── utils/                    # Helper functions
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit and integration tests
├── config/                       # Configuration files
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git ignore rules
├── Dockerfile                    # Docker containerization
└── README.md                     # This file
```

## Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows

## Installation

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

## Quick Start

Get up and running in minutes:

```python
# Data Processing
from src.data_processing import DataProcessor

processor = DataProcessor()
df = processor.load_and_clean('data/raw/patients.csv')

# Disease Prediction
from src.models import DiseasePredictor

predictor = DiseasePredictor(model_type='ensemble')
predictions = predictor.predict(df)

# Analysis and Visualization
from src.analysis import HealthAnalysis
from src.visualization import HealthVisualizer

analysis = HealthAnalysis()
visualizer = HealthVisualizer()

results = analysis.analyze(df)
visualizer.plot_metrics(results)
```

## Usage

### Data Processing
```python
from src.data_processing import DataProcessor

processor = DataProcessor()
df = processor.load_and_clean('data/raw/patients.csv')
```

### Disease Prediction
```python
from src.models import DiseasePredictor

predictor = DiseasePredictor(model_type='ensemble')
predictions = predictor.predict(patient_data)
```

### Analysis and Visualization
```python
from src.analysis import HealthAnalysis
from src.visualization import HealthVisualizer

analysis = HealthAnalysis()
results = analysis.analyze(patient_data)

visualizer = HealthVisualizer()
visualizer.plot_metrics(results)
```

### API Endpoints
```bash
# Start the API server
python -m src.api.main

# Example requests
curl http://localhost:8000/predict -X POST -d '{"patient_data": {...}}'
curl http://localhost:8000/analysis -X GET
```

## Technologies

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow
- **Analysis**: SciPy, Statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: Flask or FastAPI
- **Testing**: Pytest
- **Containerization**: Docker

## Troubleshooting

### Common Issues

**Dependency conflicts during installation:**
- Clear pip cache: `pip cache purge`
- Use a fresh virtual environment
- Install specific versions: `pip install -r requirements.txt --force-reinstall`

**Module import errors:**
- Ensure you're in the project root directory
- Verify virtual environment is activated
- Run: `pip install -e .`

**API server won't start:**
- Check if port 8000 is available
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check logs in `logs/` directory

For additional issues, please open a GitHub issue with:
- Python version (`python --version`)
- Full error message and traceback
- Steps to reproduce

## Contributing

Contributions are welcome! Please:
1. Create a new branch for your feature
2. Write tests for new functionality
3. Submit a pull request with a clear description

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration, please open an issue or contact the maintainers.
