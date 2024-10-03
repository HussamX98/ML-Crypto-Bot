# src/data/data_collection.py

import requests
import pandas as pd
import yaml
import time
from datetime import datetime, timedelta, timezone
import logging
import os

logging.basicConfig(level=logging.INFO)

def compute_features(df):
    df['price_change'] = df['close'].pct_change()
    df['price_change_5m'] = df['close'].pct_change(5)
    df['price_change_15m'] = df['close'].pct_change(15)
    df['price_change_1h'] = df['close'].pct_change(60)
    
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume_ratio'] = df['close'] / df['volume'].replace(0, 1)
    else:
        df['volume_change'] = 0
        df['price_volume_ratio'] = 0
    
    return df

def load_config():
    """
    Load configuration settings from config/config.yaml.
    """
    # Get the absolute path to the config directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_token_list():
    """
    Load the list of tokens from token_list.yaml.

    Returns:
    - List: Contains token addresses.
    """
    # Get the absolute path to the token_list.yaml file
    token_list_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'token_list.yaml')
    with open(token_list_path, 'r') as file:
        token_list = yaml.safe_load(file)
    return token_list.get('tokens', [])

def fetch_token_history(address, start_time, end_time, chain, interval, api_key):
    url = f"https://public-api.birdeye.so/defi/ohlcv?address={address}&type={interval}&time_from={start_time}&time_to={end_time}"
    headers = {
        "X-API-KEY": api_key,
        "accept": "application/json",
        "x-chain": chain
    }

    try:
        response = requests.get(url, headers=headers)
        print(f"API response status for token {address}: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if 'data' not in data or 'items' not in data['data'] or not data['data']['items']:
            logging.warning(f"No data available for token {address}")
            return pd.DataFrame()

        # Create DataFrame from the 'items' list
        df = pd.DataFrame(data['data']['items'])
        # 'address' is already included in the data

        # Rename columns to standardize the DataFrame
        df.rename(columns={
            'unixTime': 'timestamp',
            'c': 'close',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'v': 'volume',
            # 'lq': 'liquidity'  # Include if 'lq' exists in the response
        }, inplace=True)

        # Convert 'timestamp' to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.sort_values('datetime').drop_duplicates()

        # Ensure numeric data types
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume', pd.Series(0)), errors='coerce')

        # Replace NaN values in 'volume' column
        df['volume'] = df['volume'].fillna(0)

        # Data validation and cleaning
        df = df[df['datetime'] <= datetime.now(timezone.utc)]
        df = df[df['close'] > 0]
        df.dropna(subset=['close'], inplace=True)

        # Add the token address to the DataFrame
        df['address'] = address

        return compute_features(df)

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred for token {address}: {http_err}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching data for token {address}: {e}")
        return pd.DataFrame()

def fetch_historical_token_data(token_addresses, chain='solana', interval='15m', api_key=None):
    all_token_data = []
    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())  # Fetch last 30 days of data

    for address in token_addresses:
        print(f"Fetching data for token: {address}")
        try:
            token_data = fetch_token_history(address, start_time, end_time, chain, interval, api_key)
            if not token_data.empty:
                print(f"Data fetched for token {address}:")
                print(token_data.head())
                all_token_data.append(token_data)
            else:
                print(f"No data returned for token: {address}")
            time.sleep(1)  # Add a delay to avoid hitting rate limits
        except Exception as e:
            logging.error(f"Error fetching data for token {address}: {str(e)}")

    if all_token_data:
        historical_df = pd.concat(all_token_data, ignore_index=True)
        events_1440min = detect_5x_events(historical_df, window_minutes=1440, min_volume=10000)
        events_60min = detect_5x_events(historical_df, window_minutes=60, min_volume=10000)
        events_15min = detect_5x_events(historical_df, window_minutes=15, min_volume=10000)
        events_5min = detect_5x_events(historical_df, window_minutes=5, min_volume=10000)
        return historical_df, events_1440min, events_60min, events_15min, events_5min
    else:
        logging.warning("No historical data fetched for any token.")
        return pd.DataFrame(), [], [], [], []

def detect_5x_events(df, window_minutes=15, min_volume=10000):
    events = []
    for address, token_data in df.groupby('address'):
        token_data = token_data.sort_values('datetime').reset_index(drop=True)
        
        window_size = pd.Timedelta(minutes=window_minutes)
        
        for i in range(len(token_data)):
            start_time = token_data.loc[i, 'datetime']
            end_time = start_time + window_size

            window_data = token_data[(token_data['datetime'] >= start_time) & (token_data['datetime'] <= end_time)]
            
            if len(window_data) > 1:
                start_price = window_data.iloc[0]['close']
                max_price = window_data['close'].max()
                max_price_index = window_data['close'].idxmax()
                end_time = window_data.loc[max_price_index, 'datetime']
                total_volume = window_data['volume'].sum()

                if max_price >= 5 * start_price and total_volume >= min_volume:
                    events.append({
                        'address': address,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_price': start_price,
                        'end_price': max_price,
                        'increase_factor': max_price / start_price,
                        'window_size': window_minutes,
                        'total_volume': total_volume
                    })
    return events
