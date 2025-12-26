"""
Test script to verify CSV loading and integration with training pipeline.
"""

from load_data import load_news_for_ticker
from feature_engineering import align_dataframes
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

def test_csv_loading():
    """Test loading MSFT news from CSV."""
    print("="*70)
    print("Testing CSV Loading for MSFT")
    print("="*70)
    
    # Load news data
    news_df = load_news_for_ticker('MSFT')
    
    print(f"\n✓ Successfully loaded {len(news_df)} news articles")
    print(f"  Date range: {news_df['published_at'].min()} to {news_df['published_at'].max()}")
    print(f"  Sentiment range: {news_df['sentiment'].min():.3f} to {news_df['sentiment'].max():.3f}")
    print(f"  Average sentiment: {news_df['sentiment'].mean():.3f}")
    
    # Create sample other dataframes (empty for now)
    ohlcv_df = pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    prices_df = pd.DataFrame(columns=['timestamp', 'ticker', 'price'])
    quarterly_df = pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                        'operating_cash_flow', 'total_assets', 'total_liabilities'])
    
    print("\n" + "="*70)
    print("Testing Feature Engineering Integration")
    print("="*70)
    
    # Try to align dataframes (will work even with empty OHLCV)
    try:
        # Create minimal OHLCV data for testing
        dates = pd.date_range(start=news_df['published_at'].min(), 
                            end=news_df['published_at'].max(), 
                            freq='D')
        sample_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'ticker': 'MSFT',
            'open': 400 + np.random.randn(len(dates)) * 10,
            'high': 410 + np.random.randn(len(dates)) * 10,
            'low': 390 + np.random.randn(len(dates)) * 10,
            'close': 400 + np.random.randn(len(dates)) * 10,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        features_df = align_dataframes(
            news_df=news_df,
            ohlcv_df=sample_ohlcv,
            prices_df=prices_df,
            quarterly_df=quarterly_df,
            target_column='close',
            prediction_horizon=1
        )
        
        print(f"\n✓ Successfully created features!")
        print(f"  Total samples: {len(features_df)}")
        print(f"  Features with news sentiment: {features_df['sentiment_count'].sum()}")
        print(f"\n  Sample features:")
        sentiment_cols = [col for col in features_df.columns if 'sentiment' in col]
        print(f"  Sentiment columns: {sentiment_cols}")
        if len(features_df) > 0:
            print(f"\n  Sample row (sentiment features):")
            print(features_df[sentiment_cols].iloc[0])
        
    except Exception as e:
        print(f"\n✗ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Once OHLCV, prices, and quarterly data are ready, update:")
    print("   - train_trading_model.py will automatically use CSV news data")
    print("2. Run training:")
    print("   python train_trading_model.py --ticker MSFT")
    print("3. The model will use real MSFT news data from CSV!")

if __name__ == "__main__":
    test_csv_loading()

