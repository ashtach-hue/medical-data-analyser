"""Sample patient data generator and loader."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Tuple


def generate_sample_patient_data(n_patients: int = 10, 
                                n_records_per_patient: int = 12,
                                output_path: str = 'data/raw/patients_sample.csv') -> pd.DataFrame:
    """Generate sample patient data for testing and demonstration.
    
    Args:
        n_patients: Number of patients to generate
        n_records_per_patient: Number of records per patient
        output_path: Output file path
        
    Returns:
        Generated DataFrame
    """
    data = []
    np.random.seed(42)
    
    for patient_id in range(1, n_patients + 1):
        # Generate realistic baseline values for this patient
        base_age = np.random.randint(25, 80)
        base_bp = np.random.randint(100, 160)
        base_cholesterol = np.random.randint(150, 280)
        base_heart_rate = np.random.randint(60, 100)
        base_glucose = np.random.randint(70, 150)
        base_weight = np.random.randint(60, 120)
        
        start_date = datetime(2024, 1, 1)
        
        for week in range(n_records_per_patient):
            record_date = start_date + timedelta(days=week * 7)
            
            # Add some realistic variation
            noise_factor = np.random.normal(0, 0.05)  # 5% variation
            
            data.append({
                'patient_id': patient_id,
                'date': record_date.strftime('%Y-%m-%d'),
                'age': base_age,
                'gender': np.random.choice(['M', 'F']),
                'blood_pressure': int(base_bp + base_bp * noise_factor),
                'cholesterol': int(base_cholesterol + base_cholesterol * noise_factor),
                'heart_rate': int(base_heart_rate + base_heart_rate * noise_factor),
                'glucose': int(base_glucose + base_glucose * noise_factor),
                'weight': round(base_weight + base_weight * noise_factor, 1)
            })
    
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records for {n_patients} patients")
    print(f"Saved to {output_path}")
    
    return df


def load_sample_data(filepath: str = 'data/raw/patients_sample.csv') -> pd.DataFrame:
    """Load sample patient data.
    
    Args:
        filepath: Path to data file
        
    Returns:
        DataFrame with patient data
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Generating sample data...")
        return generate_sample_patient_data(output_path=filepath)
    
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_patient_data(n_patients=20, n_records_per_patient=12)
    print("\nFirst few records:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
