# src/models/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

def prepare_data(df):
    """
    Prepare data for training and testing.

    Parameters:
    - df (DataFrame): The data with features and target label.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    # Features and target
    features = [
        'price', 'volume', 'liquidity', 'holders',
        'ma_5', 'ma_10', 'volatility',
        'rsi', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    ]
    X = df[features]
    y = df['target']

    # Handle any missing values if necessary
    X = X.fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for future use
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the machine learning model.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.

    Returns:
    - model: The trained model.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=10,  # Adjust for class imbalance
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.

    Parameters:
    - model: The trained model.
    - filepath (str): The file path to save the model.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
