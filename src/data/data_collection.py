# src/data/data_collection.py

import requests
import pandas as pd
import yaml
import time
import datetime
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def compute_features(df):
    df['price_change_5m'] = df['value'].pct_change(5)
    df['price_change_15m'] = df['value'].pct_change(15)
    df['volume_change'] = df['volume'].pct_change()
    df['volatility'] = df['value'].rolling(15).std()
    return df

def load_config():
    """
    Load configuration settings from config/config.yaml.
    """
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_token_list():
    """
    Load the list of tokens from token_list.yaml.

    Returns:
    - List: Contains token addresses.
    """
    with open('config/token_list.yaml', 'r') as file:
        token_list = yaml.safe_load(file)
    return token_list.get('tokens', [])

def fetch_historical_price_data(token_address, chain='solana'):
    """
    Fetch historical price data for a specific token, including volume and liquidity if available.

    Parameters:
    - token_address (str): The address of the token.
    - chain (str): The blockchain network (default 'solana').

    Returns:
    - DataFrame: Contains historical price data for the token.
    """
    config = load_config()
    api_key = config['api_keys'].get('birdeye')

    url = 'https://public-api.birdeye.so/defi/history_price'

    headers = {
        'accept': 'application/json',
        'x-api-key': api_key,
        'x-chain': chain,
    }

    # Define the time range for historical data (e.g., last 7 days)
    end_time = int(time.time())
    start_time = end_time - 7 * 24 * 60 * 60  # 7 days ago

    params = {
        'address': token_address,
        'address_type': 'token',
        'type': '1m',  # 1-minute intervals if available
        'time_from': start_time,
        'time_to': end_time,
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        prices = data.get('data', [])

        if not prices:
            print(f"No price data found for token {token_address}.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(prices)

        # Rename columns based on actual response fields
        # Adjust according to actual response
        # Assuming 't' for timestamp, 'c' for price, 'v' for volume, 'l' for liquidity
        df.rename(columns={'t': 'timestamp', 'c': 'price', 'v': 'volume', 'l': 'liquidity'}, inplace=True)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Convert price, volume, and liquidity to numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume', pd.Series()), errors='coerce')
        df['liquidity'] = pd.to_numeric(df.get('liquidity', pd.Series()), errors='coerce')

        # Sort by timestamp
        df = df.sort_values('timestamp')

        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for token {token_address}: {http_err}")
        print(f"Response content: {response.content}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching price data for token {token_address}: {e}")
        return pd.DataFrame()

def fetch_holders_data(token_address, chain='solana'):
    """
    Fetch the number of holders for a specific token.

    Parameters:
    - token_address (str): The address of the token.
    - chain (str): The blockchain network.

    Returns:
    - int: Number of holders.
    """
    # Example using Solana Explorer API (adjust for actual API)
    url = f'https://public-api.solscan.io/token/holders?tokenAddress={token_address}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        holders = data.get('total', 0)
        return holders

    except Exception as e:
        print(f"Error fetching holders data for token {token_address}: {e}")
        return None

def fetch_historical_token_data(token_addresses, chain='solana', interval='1m', api_key=None):
    all_token_data = []
    pre_event_windows = []
    
    end_time = int(datetime.now().timestamp())
    # Fetch data for the last 30 days
    start_time = int((datetime.now() - timedelta(days=30)).timestamp())
    
    for address in token_addresses:
        try:
            logging.info(f"Fetching data for token: {address}")
            
            url = "https://public-api.birdeye.so/defi/history_price"
            
            params = {
                "address": address,
                "address_type": "token",
                "type": interval,
                "time_from": start_time,
                "time_to": end_time
            }
            
            headers = {
                "X-API-KEY": api_key,
                "accept": "application/json"
            }
            
            try:
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                if 'data' in data and 'items' in data['data'] and len(data['data']['items']) > 0:
                    df = pd.DataFrame(data['data']['items'])
                    df['address'] = address
                    df['unixTime'] = df['unixTime'].apply(correct_timestamp)
                    df['datetime'] = pd.to_datetime(df['unixTime'], unit='s', utc=True)
                    
                    if 'volume' not in df.columns:
                        print(f"Warning: Volume data missing for token {address}. Using placeholder values.")
                        df['volume'] = 0  # or some other placeholder value
                    
                    # Compute features
                    df = compute_features(df)
                    
                    # Detect 5x price increases within 15 minutes
                    df['price_ratio'] = df['value'] / df['value'].shift(15)  # 15 periods of 1 minute = 15 minutes
                    five_x_increases = df[df['price_ratio'] >= 5]
                    
                    if not five_x_increases.empty:
                        print(f"Detected {len(five_x_increases)} instances of 5x price increase within 15 minutes for {address}")
                        for _, row in five_x_increases.iterrows():
                            print(f"  5x increase at {row['datetime']}: {row['value']/row['price_ratio']:.8f} to {row['value']:.8f}")
                        
                        # Extract pre-event windows
                        def extract_pre_event_window(df, event_time, window_minutes=60):
                            return df[(df['datetime'] >= event_time - pd.Timedelta(minutes=window_minutes)) & 
                                      (df['datetime'] < event_time)]
                        
                        for _, row in five_x_increases.iterrows():
                            window = extract_pre_event_window(df, row['datetime'])
                            pre_event_windows.append(window)
                    
                    all_token_data.append(df)
                    logging.info(f"Processed {len(df)} data points for token {address}")
                    print(f"Data collected for token {address}: {len(df)} data points")
                    print(f"Date range for {address}: from {df['datetime'].min()} to {df['datetime'].max()}")
                else:
                    print(f"No data available for token {address}")
                    print(f"API Response: {data}")
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching price data for token {address}: {e}")
                print(f"Response content: {response.content}")
            except KeyError as e:
                print(f"Error processing data for token {address}: {e}")
                print(f"API Response: {data}")
        except Exception as e:
            print(f"Error processing data for token {address}: {e}")
            continue  # Skip to the next token instead of breaking the loop
    
    if all_token_data:
        result_df = pd.concat(all_token_data, ignore_index=True)
        print(f"\nTotal data points collected: {len(result_df)}")
        print(f"Overall date range: from {result_df['datetime'].min()} to {result_df['datetime'].max()}")
        return result_df, pre_event_windows
    else:
        print("No data collected for any tokens.")
        return pd.DataFrame(), []

def correct_timestamp(unix_time):
    current_time = int(time.time())
    time_difference = max(unix_time - current_time, 0)
    return current_time - time_difference

def validate_data(df):
    current_time = int(time.time())
    if df['unixTime'].max() > current_time:
        logging.warning(f"Future dates detected for token {df['address'].iloc[0]}")
    if df['value'].min() < 0:
        logging.warning(f"Negative prices detected for token {df['address'].iloc[0]}")
    return df

if __name__ == '__main__':
    # Load token list from token_list.yaml
    print("Loading token list from token_list.yaml...")
    token_addresses = load_token_list()
    print(f"Loaded tokens: {token_addresses}")

    # Fetch historical data for selected tokens
    print("Fetching historical data for selected tokens...")
    historical_df = fetch_historical_token_data(token_addresses, chain='solana')
    if historical_df.empty:
        print("No historical data fetched. Exiting.")
        exit()
    else:
        print("Historical data fetched successfully.")
