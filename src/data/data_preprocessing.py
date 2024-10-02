# src/data/data_preprocessing.py

import pandas as pd
import numpy as np

def clean_data(df):
    """
    Perform data cleaning tasks.

    Parameters:
    - df (DataFrame): The raw data DataFrame.

    Returns:
    - DataFrame: The cleaned data.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna(subset=['price'])

    # Fill missing volume and liquidity with zeros if necessary
    df['volume'] = df['volume'].fillna(0)
    df['liquidity'] = df['liquidity'].fillna(0)

    # Reset index
    df = df.reset_index(drop=True)

    return df

def preprocess_data(df):
    """
    Perform data preprocessing tasks.

    Parameters:
    - df (DataFrame): The cleaned data DataFrame.

    Returns:
    - DataFrame: The preprocessed data.
    """
    # Sort data by token and timestamp
    df = df.sort_values(by=['token_address', 'timestamp']).reset_index(drop=True)

    # Calculate returns
    df['return'] = df.groupby('token_address')['price'].pct_change()

    # Handle any remaining missing values
    df = df.dropna()

    return df
