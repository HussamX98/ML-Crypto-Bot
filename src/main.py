# src/main.py

import logging
import csv
from datetime import datetime, timedelta, timezone

# Import functions from data_collection.py
from data.data_collection import load_config, load_token_list, fetch_historical_token_data

def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    api_key = config['api_keys']['birdeye']
    print(f"API Key: {api_key[:4]}...{api_key[-4:]}")

    print("Step 1: Data Collection")
    token_addresses = load_token_list()
    print(f"Loaded tokens: {token_addresses}")

    # Set the interval
    interval = '15m'

    # Fetch historical data
    historical_df, events_1440min, events_60min, events_15min, events_5min = fetch_historical_token_data(
        token_addresses, chain='solana', interval=interval, api_key=api_key
    )

    if not historical_df.empty:
        print("\nData collection successful!")

        # Display basic statistics for each token
        for address, token_data in historical_df.groupby('address'):
            print(f"\nToken: {address}")
            print(f"  Number of data points: {len(token_data)}")
            print(f"  Date range: from {token_data['datetime'].min()} to {token_data['datetime'].max()}")
            print(f"  Price range: {token_data['close'].min():.8f} to {token_data['close'].max():.8f}")

        # Compare 24-hour, 60-minute, 15-minute, and 5-minute windows
        print("\nComparison of 24-hour, 60-minute, 15-minute, and 5-minute windows:")
        print(f"Number of 5x events in 24-hour windows: {len(events_1440min)}")
        print(f"Number of 5x events in 60-minute windows: {len(events_60min)}")
        print(f"Number of 5x events in 15-minute windows: {len(events_15min)}")
        print(f"Number of 5x events in 5-minute windows: {len(events_5min)}")

        print(f"\nNumber of tokens with 5x events (24-hour windows): {len(set(e['address'] for e in events_1440min))}")
        print(f"Number of tokens with 5x events (60-minute windows): {len(set(e['address'] for e in events_60min))}")
        print(f"Number of tokens with 5x events (15-minute windows): {len(set(e['address'] for e in events_15min))}")
        print(f"Number of tokens with 5x events (5-minute windows): {len(set(e['address'] for e in events_5min))}")

        # Display top events for each window size
        for window_size, events in [("24-hour", events_1440min), ("60-minute", events_60min), ("15-minute", events_15min), ("5-minute", events_5min)]:
            print(f"\nTop 5 events in {window_size} windows:")
            events_sorted = sorted(events, key=lambda x: x['increase_factor'], reverse=True)
            for event in events_sorted[:5]:
                print(f"  Token: {event['address']}")
                print(f"    Start: {event['start_time']} - Price: {event['start_price']:.8f}")
                print(f"    End: {event['end_time']} - Price: {event['end_price']:.8f}")
                print(f"    Increase: {event['increase_factor']:.2f}x")
                print(f"    Duration: {(event['end_time'] - event['start_time']).total_seconds() / 60:.2f} minutes")
                print(f"    Total Volume: {event['total_volume']:.2f}")

        # Additional statistics
        for window_size, events in [("24-hour", events_1440min), ("60-minute", events_60min), ("15-minute", events_15min), ("5-minute", events_5min)]:
            if events:
                max_increase = max(e['increase_factor'] for e in events)
                avg_increase = sum(e['increase_factor'] for e in events) / len(events)
                print(f"\n{window_size} window statistics:")
                print(f"  Largest price increase: {max_increase:.2f}x")
                print(f"  Average price increase: {avg_increase:.2f}x")
            else:
                print(f"\n{window_size} window statistics:")
                print("  No events detected.")

        # Optional: Save events to CSV files
        for window_size, events in [("24-hour", events_1440min), ("60-minute", events_60min), ("15-minute", events_15min), ("5-minute", events_5min)]:
            if events:
                filename = f"detected_events_{window_size.replace('-', '_')}.csv"
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['address', 'start_time', 'end_time', 'start_price', 'end_price', 'increase_factor', 'duration', 'total_volume']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for event in events:
                        writer.writerow({
                            'address': event['address'],
                            'start_time': event['start_time'],
                            'end_time': event['end_time'],
                            'start_price': event['start_price'],
                            'end_price': event['end_price'],
                            'increase_factor': event['increase_factor'],
                            'duration': (event['end_time'] - event['start_time']).total_seconds() / 60,
                            'total_volume': event['total_volume']
                        })
                print(f"Events saved to {filename}")
    else:
        print("No historical data fetched or all data was invalid. Please check your data source and processing logic.")

if __name__ == '__main__':
    main()
