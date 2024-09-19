# scripts/predict.py

import pandas as pd
from src.data.data_collection import fetch_high_frequency_data
from src.data.data_preprocessing import clean_data, preprocess_data
from src.data.feature_engineering import add_custom_features
import joblib
import time
import numpy as np

def predict_new_tokens():
    # Load model
    model = joblib.load('models/saved_models/xgboost_model.pkl')

    # Fetch new token listings
    from src.data.data_collection import fetch_new_token_listings
    new_tokens_df = fetch_new_token_listings()

    if new_tokens_df.empty:
        print("No new tokens to analyze.")
        return None

    promising_tokens = []

    for index, row in new_tokens_df.iterrows():
        token_address = row['token_address']
        print(f"Analyzing token: {row['token_name']} ({token_address})")

        # Fetch high-frequency data from the token's listing time until now
        end_timestamp = int(time.time())
        # Assuming token was listed within the last few hours
        start_timestamp = end_timestamp - (6 * 60 * 60)  # Last 6 hours

        df = fetch_high_frequency_data(token_address, start_timestamp, end_timestamp)

        if df.empty or len(df) < 20:  # Ensure enough data points
            continue

        df = clean_data(df)
        df = preprocess_data(df)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df = add_custom_features(df)

        # Use the most recent data point for prediction
        features = [
            'return', 'volatility', 'volume_change', 'price_volume_corr',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
            'ma_3', 'ma_5', 'ma_10', 'ema_3', 'ema_5', 'rsi'
        ]

        X_new = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_new = X_new.tail(1)  # Get the latest data point

        # Check if all required features are present
        if X_new.isnull().values.any():
            continue

        # Predict
        prediction = model.predict(X_new)[0]

        if prediction == 1:
            print(f"Promising token detected: {row['token_name']}")
            promising_tokens.append(row)

    if promising_tokens:
        promising_tokens_df = pd.DataFrame(promising_tokens)
        print("Promising new tokens:")
        print(promising_tokens_df[['token_name', 'token_address']])
        return promising_tokens_df
    else:
        print("No promising tokens found at this time.")
        return None

if __name__ == '__main__':
    predict_new_tokens()
