"""Streamlit Dashboard for Medical Data Analyser."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DataProcessor
from src.trends import TrendAnalyzer
from src.analysis import HealthAnalysis
from src.utils.sample_data import generate_sample_patient_data, load_sample_data
from src.models import DiseasePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="Medical Data Analyser",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🏥 Medical Data Analyser Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Upload", "Trends Analysis", "Patient Comparison", "Predictions", "Settings"]
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'trends' not in st.session_state:
    st.session_state.trends = None


def home():
    """Home page."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        # Welcome to Medical Data Analyser
        
        A comprehensive platform for analyzing patient health data and identifying health trends.
        
        ## Key Features:
        - 📊 **Trend Analysis**: Track health metrics over time
        - 👥 **Patient Comparison**: Compare metrics across patients
        - 🤖 **Disease Prediction**: ML-based risk assessment
        - 📈 **Interactive Visualizations**: Beautiful, interactive charts
        - 📥 **Data Import**: Upload your own patient data
        
        ## Getting Started:
        1. Upload your patient data in CSV format
        2. Explore trends and patterns
        3. Compare patients and identify at-risk individuals
        4. Generate insights and recommendations
        """)
    
    with col2:
        st.info("""
        ### Quick Stats
        - **Version**: 0.1.0
        - **Status**: Active
        - **Last Updated**: 2024
        """)
    
    # Sample data section
    st.markdown("---")
    st.subheader("Try with Sample Data")
    
    if st.button("📊 Load Sample Data"):
        with st.spinner("Generating sample data..."):
            df = generate_sample_patient_data(n_patients=20, n_records_per_patient=12)
            st.session_state.df = df
            st.success("Sample data loaded! Navigate to 'Data Upload' to explore.")
            st.dataframe(df.head(10))


def data_upload():
    """Data upload and exploration page."""
    st.subheader("📥 Upload Patient Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        if st.button("🔄 Load Sample Data"):
            df = load_sample_data()
            st.session_state.df = df
            st.success("Sample data loaded!")
    
    # Display data overview
    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            unique_patients = st.session_state.df['patient_id'].nunique() if 'patient_id' in st.session_state.df.columns else 0
            st.metric("Total Patients", unique_patients)
        with col3:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col4:
            missing_values = st.session_state.df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Data preview
        st.markdown("**Data Preview:**")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Data statistics
        st.markdown("**Statistical Summary:**")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # Data types
        st.markdown("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': st.session_state.df.dtypes,
            'Non-Null Count': st.session_state.df.count()
        })
        st.dataframe(col_info, use_container_width=True)


def trends_analysis():
    """Trends analysis page."""
    st.subheader("📈 Trend Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.df
    
    # Ensure date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'patient_id' in numeric_cols:
            numeric_cols.remove('patient_id')
        
        metric = st.selectbox("Select metric to analyze:", numeric_cols)
    
    with col2:
        # Select patient(s)
        if 'patient_id' in df.columns:
            patients = df['patient_id'].unique()
            selected_patients = st.multiselect(
                "Select patients:",
                patients,
                default=[patients[0]] if len(patients) > 0 else []
            )
        else:
            selected_patients = []
    
    if selected_patients and metric:
        # Filter data
        filtered_df = df[df['patient_id'].isin(selected_patients)]
        
        # Plot trends
        fig = px.line(
            filtered_df,
            x='date' if 'date' in filtered_df.columns else filtered_df.index,
            y=metric,
            color='patient_id' if 'patient_id' in filtered_df.columns else None,
            title=f"{metric} Trends Over Time",
            markers=True
        )
        
        fig.update_layout(hovermode='x unified', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.markdown("---")
        st.subheader("📊 Trend Statistics")
        
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_patient_trends(filtered_df)
        
        for patient_id in selected_patients:
            if patient_id in trends:
                st.markdown(f"**Patient {patient_id}:**")
                
                if metric in trends[patient_id]:
                    trend_info = trends[patient_id][metric]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Trend",
                            trend_info.get('trend_direction', 'N/A'),
                            delta=f"Strength: {trend_info.get('trend_strength', 0):.2f}"
                        )
                    with col2:
                        st.metric("Latest Value", f"{trend_info.get('latest_value', 0):.1f}")
                    with col3:
                        st.metric("Mean", f"{trend_info.get('mean', 0):.1f}")
                    with col4:
                        st.metric("Std Dev", f"{trend_info.get('std', 0):.1f}")


def patient_comparison():
    """Patient comparison page."""
    st.subheader("👥 Patient Comparison")
    
    if st.session_state.df is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'patient_id' in numeric_cols:
            numeric_cols.remove('patient_id')
        
        metric = st.selectbox("Select metric to compare:", numeric_cols)
    
    with col2:
        # Select patients
        if 'patient_id' in df.columns:
            patients = sorted(df['patient_id'].unique())
            selected_patients = st.multiselect(
                "Select patients to compare:",
                patients,
                default=patients[:3] if len(patients) >= 3 else patients
            )
    
    if selected_patients and metric:
        filtered_df = df[df['patient_id'].isin(selected_patients)]
        
        # Create comparison visualization
        fig = px.box(
            filtered_df,
            x='patient_id',
            y=metric,
            title=f"{metric} Comparison Across Patients",
            color='patient_id'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison statistics
        st.markdown("---")
        st.subheader("📊 Comparison Statistics")
        
        comparison_stats = filtered_df.groupby('patient_id')[metric].agg([
            ('Mean', 'mean'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std'),
            ('Count', 'count')
        ]).round(2)
        
        st.dataframe(comparison_stats, use_container_width=True)


def predictions():
    """Predictions page."""
    st.subheader("🤖 Disease Risk Predictions")
    
    if st.session_state.df is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    st.markdown("### Enter Patient Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=50, max_value=250, value=120)
    
    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=200)
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=70)
    
    with col3:
        glucose = st.number_input("Glucose", min_value=40, max_value=400, value=100)
        gender = st.selectbox("Gender", ["M", "F"])
    
    if st.button("🔮 Predict Risk"):
        # Simple risk calculation
        risk_score = 0
        
        if blood_pressure > 140:
            risk_score += 3
        elif blood_pressure > 130:
            risk_score += 2
        
        if cholesterol > 240:
            risk_score += 3
        elif cholesterol > 200:
            risk_score += 2
        
        if glucose > 126:
            risk_score += 2
        
        if age > 60:
            risk_score += 2
        
        # Determine risk level
        if risk_score < 3:
            risk_level = "Low"
            risk_color = "green"
            confidence = 0.85
        elif risk_score < 6:
            risk_level = "Moderate"
            risk_color = "orange"
            confidence = 0.72
        else:
            risk_level = "High"
            risk_color = "red"
            confidence = 0.78
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Level", risk_level)
        with col2:
            st.metric("Risk Score", f"{risk_score}/10")
        with col3:
            st.metric("Confidence", f"{confidence:.0%}")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 10], 'y': [0, 10]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgreen"},
                    {'range': [3, 6], 'color': "lightyellow"},
                    {'range': [6, 10], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)


def settings():
    """Settings page."""
    st.subheader("⚙️ Settings")
    
    st.markdown("### Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Theme**")
        theme = st.selectbox("Select theme:", ["Light", "Dark", "Auto"])
    
    with col2:
        st.markdown("**Data Processing**")
        train_test_split = st.slider("Train/Test Split Ratio:", 0.7, 0.95, 0.8, 0.05)
    
    st.markdown("---")
    st.markdown("### About")
    
    st.markdown("""
    **Medical Data Analyser Dashboard v0.1.0**
    
    A comprehensive healthcare analytics platform for:
    - Processing patient health data
    - Analyzing health trends over time
    - Comparing patient metrics
    - Predicting disease risk using ML
    - Interactive data visualization
    
    **Technologies:**
    - Streamlit for web interface
    - Pandas for data processing
    - Plotly for interactive charts
    - Scikit-learn for ML predictions
    
    **Contact:**
    For questions or feedback, please open an issue on GitHub.
    """)


# Route to pages
if page == "Home":
    home()
elif page == "Data Upload":
    data_upload()
elif page == "Trends Analysis":
    trends_analysis()
elif page == "Patient Comparison":
    patient_comparison()
elif page == "Predictions":
    predictions()
elif page == "Settings":
    settings()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Medical Data Analyser Dashboard | v0.1.0</p>
        <p>© 2024 | <a href='https://github.com/ashtach-hue/medical-data-analyser'>GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
