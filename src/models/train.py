# src/models/train.py

import numpy as np
from sklearn.model_selection import train_test_split
from src.models.model import create_xgboost_model
import joblib
from imblearn.over_sampling import RandomOverSampler

def prepare_data(df):
    """
    Prepare data for model training.

    Parameters:
    - df (DataFrame): The input DataFrame with features and target label.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing datasets.
    """
    # Remove rows where 'success_label' is NaN
    df = df.dropna(subset=['success_label'])

    # Define features and target variable
    features = [
        'return', 'volatility', 'volume_change', 'price_volume_corr',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
        'ma_3', 'ma_5', 'ma_10', 'ema_3', 'ema_5', 'rsi'
    ]
    target = 'success_label'

    X = df[features]
    y = df[target]

    # Handle missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Address class imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the XGBoost model.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.

    Returns:
    - model: Trained XGBoost model.
    """
    model = create_xgboost_model()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """
    Save the trained model to a file.

    Parameters:
    - model: Trained model.
    - filename: Path to save the model.
    """
    joblib.dump(model, filename)
