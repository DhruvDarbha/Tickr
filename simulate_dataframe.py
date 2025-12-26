"""
Simulated DataFrame Example

This script creates a simulated DataFrame showing what the news data
would look like based on actual Marketaux API responses.
"""

import pandas as pd
import pytz
from datetime import datetime

def create_simulated_dataframe():
    """Create a simulated DataFrame based on real API response data."""
    
    # Simulated data based on actual API responses from previous run
    simulated_data = [
        {
            'published_at': pd.to_datetime('2025-12-25T22:07:35.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.5682
        },
        {
            'published_at': pd.to_datetime('2025-12-25T21:07:30.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.93
        },
        {
            'published_at': pd.to_datetime('2025-12-25T19:07:38.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.76595
        },
        {
            'published_at': pd.to_datetime('2025-12-26T08:33:39.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.212
        },
        {
            'published_at': pd.to_datetime('2025-12-26T06:23:02.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.44865
        },
        # Add more examples from the actual response
        {
            'published_at': pd.to_datetime('2024-11-08T01:24:00.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.7783
        },
        {
            'published_at': pd.to_datetime('2024-11-07T23:49:09.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.0
        },
        {
            'published_at': pd.to_datetime('2024-11-07T22:28:28.000000Z', utc=True),
            'ticker': 'AAPL',
            'sentiment': 0.7783
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(simulated_data)
    
    # Ensure published_at is datetime with UTC timezone
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    
    # Sort by timestamp ascending
    df = df.sort_values('published_at').reset_index(drop=True)
    
    # Display the DataFrame
    print("=" * 70)
    print("Simulated News DataFrame (based on real Marketaux API data)")
    print("=" * 70)
    print(f"\nDataFrame Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate Range: {df['published_at'].min()} to {df['published_at'].max()}")
    print(f"Total Articles: {len(df)}")
    print(f"\nSentiment Statistics:")
    print(f"  Min: {df['sentiment'].min():.3f}")
    print(f"  Max: {df['sentiment'].max():.3f}")
    print(f"  Mean: {df['sentiment'].mean():.3f}")
    print(f"  Std: {df['sentiment'].std():.3f}")
    
    print(f"\n{'=' * 70}")
    print("Full DataFrame:")
    print(f"{'=' * 70}")
    print(df.to_string())
    
    print(f"\n{'=' * 70}")
    print("DataFrame Info:")
    print(f"{'=' * 70}")
    print(f"Data Types:")
    print(f"  published_at: {df['published_at'].dtype}")
    print(f"  ticker: {df['ticker'].dtype}")
    print(f"  sentiment: {df['sentiment'].dtype}")
    
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\n{'=' * 70}")
    print("First 3 rows:")
    print(f"{'=' * 70}")
    print(df.head(3).to_string())
    
    print(f"\n{'=' * 70}")
    print("Last 3 rows:")
    print(f"{'=' * 70}")
    print(df.tail(3).to_string())
    
    # Show what it would look like for 2 years of data
    print(f"\n{'=' * 70}")
    print("Projection: What 2 Years of Data Would Look Like")
    print(f"{'=' * 70}")
    print(f"If you collect 2 years of data:")
    print(f"  - 730 days × ~2-5 articles/day = ~1,460 - 3,650 articles")
    print(f"  - Estimated DataFrame size: ~{(len(df) * 730 / len(df) * df.memory_usage(deep=True).sum() / 1024 / 1024):.2f} MB")
    print(f"  - Columns: published_at (datetime64[ns, UTC]), ticker (object), sentiment (float64)")
    
    return df


if __name__ == "__main__":
    df = create_simulated_dataframe()
    
    print(f"\n{'=' * 70}")
    print("✅ DataFrame created successfully!")
    print(f"{'=' * 70}")
    print(f"\nYou can now use this DataFrame structure for training.")
    print(f"To access the DataFrame in your code:")
    print(f"  df = create_simulated_dataframe()")
    print(f"  # Or load from CSV: df = pd.read_csv('news_data.csv', parse_dates=['published_at'])")

