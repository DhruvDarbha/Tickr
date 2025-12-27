"""
Merge All Data Sources for Model Training

This script:
1. Loads news + OHLCV data from MSFT_all_dates_data_with_prices.csv
2. Loads quarterly data from PDFs
3. Merges all data sources
4. Displays the final merged DataFrame ready for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path

from load_data import load_quarterly_from_pdfs
from feature_engineering import align_dataframes

def load_news_and_ohlcv_from_csv(csv_path: str, ticker: str = 'MSFT') -> tuple:
    """
    Load news and OHLCV data from the combined CSV file.
    
    Args:
        csv_path: Path to CSV file
        ticker: Stock ticker symbol
    
    Returns:
        Tuple of (news_df, ohlcv_df)
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract news DataFrame
    news_df = pd.DataFrame({
        'published_at': pd.to_datetime(df['published_at'], utc=True),
        'ticker': df['ticker'],
        'sentiment': pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)
    })
    
    # Sort and deduplicate news
    news_df = news_df.sort_values('published_at').reset_index(drop=True)
    news_df = news_df.drop_duplicates(subset=['published_at', 'ticker'], keep='first')
    
    # Extract OHLCV DataFrame
    # Use price_date_on_published_date for alignment
    ohlcv_df = pd.DataFrame({
        'timestamp': pd.to_datetime(df['price_date_on_published_date'], utc=True),
        'ticker': ticker,
        'open': pd.to_numeric(df['msft_open_on_published_date'], errors='coerce'),
        'high': pd.to_numeric(df['msft_high_on_published_date'], errors='coerce'),
        'low': pd.to_numeric(df['msft_low_on_published_date'], errors='coerce'),
        'close': pd.to_numeric(df['msft_close_on_published_date'], errors='coerce'),
        'volume': pd.to_numeric(df['msft_volume_on_published_date'], errors='coerce')
    })
    
    # Remove rows with missing price data
    ohlcv_df = ohlcv_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    # Sort and deduplicate OHLCV (keep first occurrence per timestamp)
    ohlcv_df = ohlcv_df.sort_values('timestamp').reset_index(drop=True)
    ohlcv_df = ohlcv_df.drop_duplicates(subset=['timestamp'], keep='first')
    
    print(f"  ✓ Loaded {len(news_df)} news articles")
    print(f"  ✓ Loaded {len(ohlcv_df)} OHLCV records")
    print(f"  News date range: {news_df['published_at'].min()} to {news_df['published_at'].max()}")
    print(f"  OHLCV date range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
    
    return news_df, ohlcv_df

def merge_all_data_sources(ticker: str = 'MSFT') -> pd.DataFrame:
    """
    Merge all data sources: news, OHLCV, quarterly reports.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Merged DataFrame ready for model training
    """
    print("="*100)
    print("MERGING ALL DATA SOURCES FOR MODEL TRAINING")
    print("="*100)
    
    # 1. Load news and OHLCV from CSV
    csv_path = f"{ticker}_all_dates_data_with_prices.csv"
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return pd.DataFrame()
    
    news_df, ohlcv_df = load_news_and_ohlcv_from_csv(csv_path, ticker)
    
    # 2. Load quarterly data
    print(f"\nLoading quarterly data for {ticker}...")
    quarterly_df = load_quarterly_from_pdfs(ticker=ticker)
    
    # 3. Create empty prices DataFrame (can be populated later if needed)
    prices_df = pd.DataFrame(columns=['timestamp', 'ticker', 'price'])
    
    # 4. Align all dataframes using feature engineering
    print("\n" + "="*100)
    print("ALIGNING ALL DATAFRAMES")
    print("="*100)
    
    try:
        merged_df = align_dataframes(
            news_df=news_df,
            ohlcv_df=ohlcv_df,
            prices_df=prices_df,
            quarterly_df=quarterly_df,
            target_column='close',
            prediction_horizon=1
        )
        
        print(f"\n✓ Successfully merged all data sources!")
        print(f"  Total samples: {len(merged_df)}")
        print(f"  Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
        
        return merged_df
        
    except Exception as e:
        print(f"\n✗ Error aligning dataframes: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def display_merged_dataframe(df: pd.DataFrame):
    """Display the final merged DataFrame."""
    if df.empty:
        print("\n❌ No data to display!")
        return
    
    print("\n" + "="*100)
    print("FINAL MERGED DATAFRAME (Ready for Model Training)")
    print("="*100)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 50)
    
    # Show first 20 rows
    print("\nFirst 20 rows:")
    print(df.head(20).to_string(index=False))
    
    # Show last 10 rows
    print("\n" + "-"*100)
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    
    # Show summary statistics
    print("\n" + "="*100)
    print("DATAFRAME SUMMARY")
    print("="*100)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n\nData Types:")
    print(df.dtypes)
    
    print(f"\n\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("  No missing values!")
    
    print(f"\n\nSample Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    # Show feature categories
    print("\n" + "="*100)
    print("FEATURE CATEGORIES")
    print("="*100)
    
    tech_indicators = [c for c in df.columns if any(x in c.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
    sentiment_features = [c for c in df.columns if 'sentiment' in c.lower()]
    quarterly_features = [c for c in df.columns if c in ['revenue', 'net_income', 'eps', 'operating_cash_flow', 
                                                          'total_assets', 'total_liabilities', 'stock_impact_score']]
    time_features = [c for c in df.columns if c in ['year', 'month', 'day', 'day_of_week', 'quarter', 'hour']]
    price_features = [c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']]
    
    print(f"\nTechnical Indicators ({len(tech_indicators)}): {tech_indicators[:10]}...")
    print(f"Sentiment Features ({len(sentiment_features)}): {sentiment_features}")
    print(f"Quarterly Features ({len(quarterly_features)}): {quarterly_features}")
    print(f"Time Features ({len(time_features)}): {time_features}")
    print(f"Price Features ({len(price_features)}): {price_features}")
    
    print("\n" + "="*100)
    print("✓ DataFrame is ready for model training!")
    print("="*100)

def main():
    """Main function to merge and display all data."""
    ticker = 'MSFT'
    
    # Merge all data sources
    merged_df = merge_all_data_sources(ticker)
    
    # Display the merged DataFrame
    if not merged_df.empty:
        display_merged_dataframe(merged_df)
    else:
        print("\n❌ Failed to merge data sources. Please check the error messages above.")

if __name__ == "__main__":
    main()

