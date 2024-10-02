# src/data/feature_engineering.py

import pandas as pd
import numpy as np
import pandas_ta as ta

def add_custom_features(df):
    """
    Add custom features to the DataFrame.

    Parameters:
    - df (DataFrame): The preprocessed data.

    Returns:
    - DataFrame: The data with additional features.
    """
    # Ensure data is sorted by token and timestamp
    df = df.sort_values(by=['token_address', 'timestamp']).reset_index(drop=True)

    # Group by token address
    grouped = df.groupby('token_address')

    # Add moving averages and volatility
    df['ma_5'] = grouped['price'].transform(lambda x: x.rolling(window=5).mean())
    df['ma_10'] = grouped['price'].transform(lambda x: x.rolling(window=10).mean())
    df['volatility'] = grouped['return'].transform(lambda x: x.rolling(window=10).std())

    # Add Relative Strength Index (RSI)
    df['rsi'] = grouped.apply(lambda x: x.ta.rsi(length=14)).reset_index(level=0, drop=True)

    # Add Moving Average Convergence Divergence (MACD)
    macd = grouped.apply(lambda x: x.ta.macd()).reset_index(level=0, drop=True)
    df = pd.concat([df, macd], axis=1)

    # Drop rows with NaN values resulting from calculations
    df = df.dropna()

    return df

def add_target_label(df):
    """
    Add a target label to the DataFrame based on 5x price increases within 15 minutes.

    Parameters:
    - df (DataFrame): The data with features.

    Returns:
    - DataFrame: The data with a target label.
    """
    # Ensure data is sorted by token and timestamp
    df = df.sort_values(by=['token_address', 'timestamp']).reset_index(drop=True)

    # Define the time window (15 minutes)
    time_window = pd.Timedelta(minutes=15)

    # Initialize target column
    df['target'] = 0

    # Group by token
    for token, group in df.groupby('token_address'):
        group = group.reset_index(drop=True)
        prices = group['price'].values
        timestamps = group['timestamp']

        for i in range(len(group)):
            current_price = prices[i]
            current_time = timestamps.iloc[i]

            # Find future timestamps within 15 minutes
            future_mask = (timestamps > current_time) & (timestamps <= current_time + time_window)
            future_prices = prices[future_mask.index]

            if len(future_prices) > 0:
                max_future_price = future_prices.max()
                if max_future_price >= 5 * current_price:
                    df.loc[group.index[i], 'target'] = 1

    return df
