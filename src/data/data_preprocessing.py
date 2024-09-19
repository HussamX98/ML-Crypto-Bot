# src/data/data_preprocessing.py

import pandas as pd

def clean_data(df):
    """
    Clean the DataFrame by removing duplicates and handling missing values.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def preprocess_data(df):
    """
    Perform additional preprocessing steps specific to high-frequency data.
    """
    # Convert data types
    numeric_fields = ['open', 'high', 'low', 'close', 'volume']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    # Handle missing or zero values
    df = df.fillna(0)
    df = df[df['close'] > 0]

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values(by='timestamp')

    return df
