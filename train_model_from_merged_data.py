"""
Train Trading Model from Merged DataFrame

This script:
1. Loads the final merged DataFrame
2. Trains the trading model
3. Saves the trained model
"""

from merge_all_data import merge_all_data_sources
from train_trading_model import train_trading_model
from load_data import load_news_for_ticker, load_quarterly_from_pdfs
import pandas as pd
import numpy as np
from pathlib import Path

def extract_dataframes_from_merged(merged_df: pd.DataFrame) -> tuple:
    """
    Extract individual dataframes from merged DataFrame for training.
    
    Returns:
        Tuple of (news_df, ohlcv_df, prices_df, quarterly_df)
    """
    # Extract OHLCV data
    ohlcv_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    ohlcv_df = merged_df[ohlcv_cols].copy()
    ohlcv_df = ohlcv_df.rename(columns={'timestamp': 'timestamp'})
    
    # Extract prices (use close as price)
    prices_df = pd.DataFrame({
        'timestamp': merged_df['timestamp'],
        'ticker': merged_df['ticker'],
        'price': merged_df['close']
    })
    
    # Extract news data (create from sentiment features)
    # We'll need to reconstruct this - for now, create minimal news df
    news_df = pd.DataFrame({
        'published_at': merged_df['timestamp'],
        'ticker': merged_df['ticker'],
        'sentiment': merged_df['sentiment_mean'].fillna(0.0)
    })
    
    # Extract quarterly data
    quarterly_cols = ['timestamp', 'ticker', 'revenue', 'net_income', 'eps',
                     'operating_cash_flow', 'total_assets', 'total_liabilities', 
                     'stock_impact_score']
    quarterly_df = merged_df[quarterly_cols].copy()
    quarterly_df = quarterly_df.rename(columns={'timestamp': 'report_date'})
    
    return news_df, ohlcv_df, prices_df, quarterly_df

def main():
    """Main training function."""
    print("="*100)
    print("TRAINING TRADING MODEL FROM MERGED DATAFRAME")
    print("="*100)
    
    ticker = 'MSFT'
    
    # Load merged DataFrame
    print("\n📊 Loading merged DataFrame...")
    merged_df = merge_all_data_sources(ticker)
    
    if merged_df.empty:
        print("❌ Error: Merged DataFrame is empty!")
        return
    
    print(f"✓ Loaded {len(merged_df)} samples with {len(merged_df.columns)} features")
    
    # Extract individual dataframes
    print("\n📦 Extracting individual dataframes...")
    news_df, ohlcv_df, prices_df, quarterly_df = extract_dataframes_from_merged(merged_df)
    
    print(f"  ✓ News: {len(news_df)} records")
    print(f"  ✓ OHLCV: {len(ohlcv_df)} records")
    print(f"  ✓ Prices: {len(prices_df)} records")
    print(f"  ✓ Quarterly: {len(quarterly_df)} records")
    
    # Train model
    print("\n🚀 Starting model training...")
    model = train_trading_model(
        ticker=ticker,
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        model_type='lightgbm',
        max_position_size=100,
        min_profit_threshold=0.001,  # 0.1% for intraday
        max_holding_minutes=240,  # 4 hours
        validation_split=0.2,
        save_path=f'models/trading_model_{ticker.lower()}.pkl'
    )
    
    print("\n" + "="*100)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*100)
    print(f"\nModel saved to: models/trading_model_{ticker.lower()}.pkl")
    print("\nYou can now test the model using the saved file.")
    print("="*100)

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    main()

