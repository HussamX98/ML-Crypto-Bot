# src/data/data_collection.py

import requests
import pandas as pd
import yaml
import time

def load_config():
    """
    Load configuration settings from config/config.yaml.
    """
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def fetch_historical_token_data(token_addresses):
    """
    Fetch high-frequency historical data for a list of tokens from their listing time onwards.

    Parameters:
    - token_addresses (list): List of token addresses.

    Returns:
    - DataFrame: Combined DataFrame of historical data for all tokens.
    """
    all_data = []

    for token_address in token_addresses:
        print(f"Fetching data for token: {token_address}")

        # Fetch data from the token's listing time onwards
        df = fetch_token_data_from_listing(token_address)
        if not df.empty:
            df['token_address'] = token_address
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv('data/raw/historical_high_freq_data.csv', index=False)
        return combined_df
    else:
        print("No data collected.")
        return pd.DataFrame()

def fetch_token_data_from_listing(token_address):
    """
    Fetch high-frequency data for a token from its listing time onwards.

    Parameters:
    - token_address (str): The address of the token.

    Returns:
    - DataFrame: DataFrame containing the token's historical data.
    """
    config = load_config()
    api_key = config['api_keys'].get('birdeye')

    if not api_key:
        raise ValueError("Birdeye API key not found in config/config.yaml under 'api_keys: birdeye'.")

    # Set the start time (e.g., token listing time)
    # You may need to fetch the listing time from the API or set a reasonable default (e.g., 7 days ago)
    # For this example, we'll use 7 days ago
    end_timestamp = int(time.time())
    start_timestamp = end_timestamp - (7 * 24 * 60 * 60)  # 7 days ago

    url = f'https://public-api.birdeye.so/public/coin/{token_address}/candlestick'

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    params = {
        'interval': '1m',  # 1-minute intervals
        'start_time': start_timestamp,
        'end_time': end_timestamp,
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        candles = data.get('data', {}).get('candles', [])

        if not candles:
            print(f"No data found for token {token_address} in the specified time frame.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return df

    except Exception as e:
        print(f"An error occurred while fetching data for token {token_address}: {e}")
        return pd.DataFrame()
