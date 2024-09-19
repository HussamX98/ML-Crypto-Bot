# src/main.py

from src.data.data_collection import fetch_historical_token_data
from src.data.data_preprocessing import clean_data, preprocess_data
from src.data.feature_engineering import add_target_label, add_custom_features
from src.models.train import prepare_data, train_model, save_model
from src.models.evaluate import evaluate_model

def main():
    # Provide a list of token addresses you want to analyze
    token_addresses = [
        'token_address_1',
        'token_address_2',
        # Add more token addresses
    ]

    # Fetch and prepare data
    df = fetch_historical_token_data(token_addresses)
    if df.empty:
        print("No data fetched. Exiting.")
        return

    df = clean_data(df)
    df = preprocess_data(df)
    df = df.sort_values(by=['token_address', 'timestamp']).reset_index(drop=True)
    df = add_custom_features(df)
    df = add_target_label(df)

    # Prepare data for training
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    save_model(model, 'models/saved_models/xgboost_model.pkl')

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
