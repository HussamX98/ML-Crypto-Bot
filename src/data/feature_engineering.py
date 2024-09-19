# src/data/feature_engineering.py

import pandas as pd
import numpy as np

def add_target_label(df):
    """
    Add a binary target label indicating whether a 500% price increase occurs within the next 15 minutes.

    Parameters:
    - df (DataFrame): The input DataFrame with high-frequency data.

    Returns:
    - DataFrame: DataFrame with the 'success_label' column added.
    """
    df = df.copy()

    # For each row, calculate the maximum future price within the next 15 minutes
    window_size = 15  # Number of minutes
    df['future_max_price'] = df['close'].rolling(window=window_size, min_periods=1).max().shift(-1)

    # Calculate the percentage price increase from the current close to the future max price
    df['price_increase'] = ((df['future_max_price'] - df['close']) / df['close']) * 100

    # Create the success label where the price increase is at least 500%
    df['success_label'] = (df['price_increase'] >= 500).astype(int)

    # Clean up temporary columns
    df = df.drop(columns=['future_max_price', 'price_increase'])

    return df



def add_custom_features(df):
    """
    Add custom features that might be predictive of a big price move within the next 15 minutes.
    """
    df = df.copy()
    # Example technical indicators
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()  # Shorter window
    df['volume_change'] = df['volume'].pct_change()
    df['price_volume_corr'] = df['close'].rolling(window=5).corr(df['volume'])
    
    # Lag features
    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # Moving averages
    df['ma_3'] = df['close'].rolling(window=3).mean()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    
    # Exponential moving averages
    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # RSI with a shorter window
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=7).mean()
    avg_loss = down.rolling(window=7).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

