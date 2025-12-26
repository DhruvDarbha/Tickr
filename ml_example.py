"""
Example: Using the ML Model Architecture

This script demonstrates how to use the ML architecture for stock prediction.
It works with placeholder data until all 4 dataframes are ready.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from feature_engineering import align_dataframes, get_feature_columns
from trading_model import TradingModel
from train_trading_model import train_trading_model, get_trading_recommendation, print_trading_recommendation
from load_data import load_news_for_ticker


def create_sample_data():
    """Create sample dataframes for demonstration."""
    
    # Sample dates (last 100 days)
    end_date = datetime.now(pytz.UTC)
    dates = [end_date - timedelta(days=x) for x in range(100, 0, -1)]
    
    # Sample OHLCV data
    np.random.seed(42)
    base_price = 150.0
    prices = []
    for i in range(100):
        change = np.random.normal(0, 2)
        base_price = max(50, base_price + change)
        prices.append(base_price)
    
    ohlcv_data = []
    for i, date in enumerate(dates):
        close = prices[i]
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = close + np.random.normal(0, 0.5)
        volume = np.random.randint(1000000, 10000000)
        
        ohlcv_data.append({
            'timestamp': date,
            'ticker': 'AAPL',
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    ohlcv_df = pd.DataFrame(ohlcv_data)
    
    # Sample news data (some days have news, some don't)
    news_data = []
    for date in dates[::3]:  # Every 3rd day
        sentiment = np.random.uniform(-0.5, 0.8)
        news_data.append({
            'published_at': date,
            'ticker': 'AAPL',
            'sentiment': sentiment
        })
    
    news_df = pd.DataFrame(news_data)
    
    # Sample price data (same as close prices)
    prices_df = pd.DataFrame([{
        'timestamp': dates[-1],
        'ticker': 'AAPL',
        'price': prices[-1]
    }])
    
    # Sample quarterly data (4 reports per year)
    quarterly_data = []
    for i in range(0, len(dates), 25):
        quarterly_data.append({
            'report_date': dates[i],
            'ticker': 'AAPL',
            'revenue': np.random.uniform(80e9, 120e9),
            'net_income': np.random.uniform(15e9, 30e9),
            'eps': np.random.uniform(1.0, 2.5),
            'operating_cash_flow': np.random.uniform(20e9, 40e9),
            'total_assets': np.random.uniform(300e9, 400e9),
            'total_liabilities': np.random.uniform(200e9, 300e9)
        })
    
    quarterly_df = pd.DataFrame(quarterly_data)
    
    return news_df, ohlcv_df, prices_df, quarterly_df


def main():
    """Example usage of ML architecture."""
    
    print("="*70)
    print("ML Model Architecture Example")
    print("="*70)
    
    # Try to load real MSFT news data first
    print("\n1. Loading data...")
    print("   Attempting to load MSFT news from CSV...")
    try:
        news_df = load_news_for_ticker('MSFT')
        if not news_df.empty:
            print(f"   ✓ Loaded {len(news_df)} real MSFT news articles from CSV!")
        else:
            print("   No CSV found, using sample data...")
            news_df, ohlcv_df, prices_df, quarterly_df = create_sample_data()
    except Exception as e:
        print(f"   CSV loading failed: {e}")
        print("   Using sample data instead...")
        news_df, ohlcv_df, prices_df, quarterly_df = create_sample_data()
    
    # If we loaded real news, still need sample data for other sources
    if 'ohlcv_df' not in locals():
        _, ohlcv_df, prices_df, quarterly_df = create_sample_data()
    
    print(f"\n   Data Summary:")
    print(f"   News articles: {len(news_df)}")
    print(f"   OHLCV records: {len(ohlcv_df)}")
    print(f"   Price records: {len(prices_df)}")
    print(f"   Quarterly reports: {len(quarterly_df)}")
    
    # Try to align dataframes and create features
    print("\n2. Engineering features from all dataframes...")
    try:
        features_df = align_dataframes(
            news_df=news_df,
            ohlcv_df=ohlcv_df,
            prices_df=prices_df,
            quarterly_df=quarterly_df,
            prediction_horizon=1,
            target_column='close'
        )
        
        feature_cols = get_feature_columns(features_df)
        print(f"   ✓ Successfully created {len(feature_cols)} features")
        print(f"   ✓ Total samples: {len(features_df)}")
        print(f"\n   Sample features:")
        print(f"   {', '.join(feature_cols[:10])}...")
        
        # Show feature statistics
        print(f"\n   Feature statistics:")
        print(f"   - Technical indicators: {len([c for c in feature_cols if 'sma' in c or 'ema' in c or 'rsi' in c or 'macd' in c])}")
        print(f"   - Sentiment features: {len([c for c in feature_cols if 'sentiment' in c])}")
        print(f"   - Quarterly features: {len([c for c in feature_cols if c in ['revenue', 'net_income', 'eps', 'operating_cash_flow']])}")
        print(f"   - Time features: {len([c for c in feature_cols if c in ['year', 'month', 'day', 'day_of_week', 'quarter']])}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: This is expected if dataframes are empty or malformed")
        return
    
    # Try to train a trading model (if we have enough data)
    if len(features_df) > 20:
        print("\n3. Training Trading Model (LightGBM)...")
        try:
            model = train_trading_model(
                ticker='AAPL',
                news_df=news_df,
                ohlcv_df=ohlcv_df,
                prices_df=prices_df,
                quarterly_df=quarterly_df,
                model_type='lightgbm',
                max_position_size=100,
                min_profit_threshold=0.001,  # 0.1% minimum profit for minute-level
                max_holding_minutes=240,  # 4 hours (240 minutes) for intraday trading
                validation_split=0.2,
                save_path=None  # Don't save for example
            )
            
            print("\n   ✓ Trading model trained successfully!")
            
            # Get trading recommendation
            if 'close' in ohlcv_df.columns:
                current_price = ohlcv_df['close'].iloc[-1]
                recommendation = get_trading_recommendation(
                    model=model,
                    news_df=news_df,
                    ohlcv_df=ohlcv_df,
                    prices_df=prices_df,
                    quarterly_df=quarterly_df,
                    current_price=current_price
                )
                print_trading_recommendation(recommendation)
            
        except Exception as e:
            print(f"   ✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            print("   Note: This may occur if ML libraries aren't installed")
            print("   Install with: pip install lightgbm xgboost scikit-learn")
    else:
        print("\n3. Skipping model training (insufficient data)")
        print("   Need at least 20 samples for training")
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\nTrading Model Output (Minute-Level/Intraday):")
    print("  - BUY/SELL/HOLD signal")
    print("  - Number of shares to buy/sell")
    print("  - Exit timing in MINUTES (when to sell if BUY, when to buy back if SELL)")
    print("  - Expected profit percentage and dollar amount")
    print("  - Max holding period: 4 hours (240 minutes)")
    print("\nNext Steps:")
    print("1. Ensure all 4 dataframes are populated with real data")
    print("2. Install ML dependencies: pip install -r requirements.txt")
    print("3. Train trading model: python train_trading_model.py --ticker AAPL --marketaux_key YOUR_KEY")
    print("4. Optimize hyperparameters once you have sufficient data")


if __name__ == "__main__":
    main()

