# src/main.py

from utils.config import load_config
from data.data_collection import load_token_list, fetch_historical_token_data
from analysis.pattern_recognition import analyze_patterns
from visualization.pattern_visualization import plot_cluster_characteristics, plot_pre_event_windows
import pandas as pd
import logging

def main():
    config = load_config()
    api_key = config['api_keys']['birdeye']
    print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
    
    print("Step 1: Data Collection")
    token_addresses = load_token_list()
    print(f"Loaded tokens: {token_addresses}")

    historical_df, pre_event_windows = fetch_historical_token_data(token_addresses, chain='solana', interval='1m', api_key=api_key)

    if not historical_df.empty:
        print("\nData collection successful!")
        print(f"Total number of data points: {len(historical_df)}")
        print(f"Number of unique tokens: {historical_df['address'].nunique()}")
        
        # Basic analysis
        for address in historical_df['address'].unique():
            token_data = historical_df[historical_df['address'] == address]
            print(f"\nToken: {address}")
            print(f"  Number of data points: {len(token_data)}")
            print(f"  Date range: from {token_data['datetime'].min()} to {token_data['datetime'].max()}")
            print(f"  Price range: {token_data['value'].min():.8f} to {token_data['value'].max():.8f}")
            
            logging.warning(f"Unexpected date range for {address}: {token_data['datetime'].min()} to {token_data['datetime'].max()}")
        
        # Step 2: Pattern Recognition
        print("\nStep 2: Pattern Recognition")
        if pre_event_windows:
            event_clusters, cluster_characteristics, common_patterns = analyze_patterns(pre_event_windows)
            print(f"Identified {len(set(event_clusters))} distinct patterns in the pre-event windows")
            
            # Print common patterns
            print("\nCommon Patterns Identified:")
            for pattern in common_patterns:
                print(f"- {pattern}")
            
            # Visualize the patterns
            plot_cluster_characteristics(cluster_characteristics)
            plot_pre_event_windows(pre_event_windows)
            
            print("\nVisualization complete. Check 'cluster_characteristics.png' and 'pre_event_windows.png' for visual representations of the patterns.")
        else:
            print("No 5x increase events detected for pattern analysis")
        
        # Here you can add more analysis or visualization code
    else:
        print("No historical data fetched or all data was invalid. Please check your data source and processing logic.")

if __name__ == '__main__':
    main()
