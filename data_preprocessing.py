"""
Data Preprocessing Script for Diabetes CGM Data
Prepares data for streaming simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class DiabetesDataPreprocessor:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def load_data(self, filename):
        """Load CGM data from CSV"""
        filepath = os.path.join(self.raw_data_path, filename)
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    
    def preprocess_cgm_data(self, df):
        """
        Preprocess CGM data:
        - Handle missing values
        - Add timestamps if not present
        - Normalize glucose values
        - Extract features
        """
        print("Preprocessing CGM data...")
        
        # Ensure we have timestamp column
        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                # Create synthetic timestamps (5-minute intervals)
                start_time = datetime.now()
                df['timestamp'] = [start_time + timedelta(minutes=5*i) for i in range(len(df))]
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Identify glucose column (common names)
        glucose_col = None
        for col in ['glucose', 'bg', 'blood_glucose', 'glucose_value', 'CGM']:
            if col in df.columns:
                glucose_col = col
                break
        
        if glucose_col is None:
            # If no glucose column found, create synthetic data
            print("Warning: No glucose column found. Creating synthetic data.")
            df['glucose'] = np.random.normal(120, 30, len(df))
            glucose_col = 'glucose'
        else:
            df['glucose'] = df[glucose_col]
        
        # Handle missing glucose values
        df['glucose'] = df['glucose'].fillna(df['glucose'].median())
        
        # Clip glucose values to realistic range (40-400 mg/dL)
        df['glucose'] = df['glucose'].clip(40, 400)
        
        # Feature extraction
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Rolling statistics (if enough data)
        if len(df) > 12:  # Need at least 1 hour of data
            df['glucose_ma_1h'] = df['glucose'].rolling(window=12, min_periods=1).mean()
            df['glucose_std_1h'] = df['glucose'].rolling(window=12, min_periods=1).std()
        else:
            df['glucose_ma_1h'] = df['glucose']
            df['glucose_std_1h'] = 0
        
        # Rate of change (if enough data)
        df['glucose_roc'] = df['glucose'].diff().fillna(0)
        
        # Add activity data (synthetic for now)
        df['steps'] = np.random.randint(0, 100, len(df))
        df['heart_rate'] = np.random.randint(60, 120, len(df))
        
        # Target variable: hypoglycemia risk (glucose < 70)
        df['hypo_risk'] = (df['glucose'] < 70).astype(int)
        
        # Select final columns
        final_columns = [
            'timestamp', 'glucose', 'glucose_ma_1h', 'glucose_std_1h',
            'glucose_roc', 'steps', 'heart_rate', 'hour', 'day_of_week', 'hypo_risk'
        ]
        
        return df[final_columns]
    
    def split_for_institutions(self, df, n_institutions=3):
        """
        Split data to simulate multiple institutions
        Each institution gets different patients/time periods
        """
        print(f"Splitting data for {n_institutions} institutions...")
        
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_size = len(df_shuffled) // n_institutions
        
        institutions = {}
        for i in range(n_institutions):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_institutions - 1 else len(df_shuffled)
            institutions[f'institution_{i+1}'] = df_shuffled.iloc[start_idx:end_idx].copy()
            
            print(f"Institution {i+1}: {len(institutions[f'institution_{i+1}'])} records")
        
        return institutions
    
    def save_for_streaming(self, institutions):
        """
        Save data in JSON format for streaming simulation
        Each record becomes a streaming event
        """
        for inst_name, df in institutions.items():
            output_file = os.path.join(self.output_path, f'{inst_name}_stream.jsonl')
            
            # Convert to JSON lines format
            df['timestamp'] = df['timestamp'].astype(str)
            df.to_json(output_file, orient='records', lines=True)
            
            print(f"Saved {inst_name} streaming data to {output_file}")
    
    def create_batch_data(self, institutions):
        """
        Save data in Parquet format for batch processing
        """
        for inst_name, df in institutions.items():
            output_file = os.path.join(self.output_path, f'{inst_name}_batch.parquet')
            df.to_parquet(output_file, index=False)
            print(f"Saved {inst_name} batch data to {output_file}")
    
    def run_pipeline(self, filename, n_institutions=3):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Load data
        df = self.load_data(filename)
        print(f"Loaded {len(df)} records")
        
        # Preprocess
        df_processed = self.preprocess_cgm_data(df)
        print(f"Preprocessed data shape: {df_processed.shape}")
        
        # Split for institutions
        institutions = self.split_for_institutions(df_processed, n_institutions)
        
        # Save for streaming
        self.save_for_streaming(institutions)
        
        # Save batch data
        self.create_batch_data(institutions)
        
        print("="*60)
        print("Preprocessing Complete!")
        print("="*60)
        
        return institutions


def create_synthetic_data(n_records=10000):
    """Create synthetic diabetes data if real data is not available"""
    print("Creating synthetic diabetes data...")
    
    np.random.seed(42)
    start_time = datetime(2024, 1, 1)
    
    data = {
        'timestamp': [start_time + timedelta(minutes=5*i) for i in range(n_records)],
        'glucose': np.random.normal(130, 40, n_records).clip(50, 350),
        'steps': np.random.randint(0, 150, n_records),
        'heart_rate': np.random.randint(55, 130, n_records),
    }
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "data/raw"
    OUTPUT_PATH = "data/processed"
    N_INSTITUTIONS = 3
    
    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH) or len(os.listdir(RAW_DATA_PATH)) == 0:
        print("No raw data found. Creating synthetic data...")
        df = create_synthetic_data(n_records=10000)
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        df.to_csv(os.path.join(RAW_DATA_PATH, "synthetic_diabetes_data.csv"), index=False)
        filename = "synthetic_diabetes_data.csv"
    else:
        # Use the first CSV file found
        csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
        if csv_files:
            filename = csv_files[0]
        else:
            print("No CSV files found. Creating synthetic data...")
            df = create_synthetic_data(n_records=10000)
            df.to_csv(os.path.join(RAW_DATA_PATH, "synthetic_diabetes_data.csv"), index=False)
            filename = "synthetic_diabetes_data.csv"
    
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor(RAW_DATA_PATH, OUTPUT_PATH)
    
    # Run pipeline
    institutions = preprocessor.run_pipeline(filename, N_INSTITUTIONS)
    
    print(f"\nData prepared for {N_INSTITUTIONS} institutions")
    print("Files created in:", OUTPUT_PATH)