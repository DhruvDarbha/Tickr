"""
Training Pipeline for Trading Model

This module trains a trading model that predicts:
- BUY/SELL/HOLD signals
- Position sizes (number of shares)
- Exit/Entry timing (minutes)
- Expected profit

Designed for intraday/minute-level trading.

Usage:
    python train_trading_model.py --ticker AAPL --marketaux_key YOUR_KEY
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import pytz
import argparse
import warnings

from feature_engineering import align_dataframes, get_feature_columns
from trading_model import TradingModel
from data_ingestion import ingest_all_data
from load_data import load_news_for_ticker

warnings.filterwarnings('ignore')


def prepare_trading_data(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame
) -> tuple:
    """
    Prepare data for trading model training.
    
    Returns:
        Tuple of (features_df, prices_series, feature_names)
    """
    print("\n" + "="*70)
    print("Preparing Trading Data")
    print("="*70)
    
    # Align all dataframes (no target needed - we'll create trading targets from prices)
    print("Aligning dataframes and engineering features...")
    features_df = align_dataframes(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        target_column='close',  # Used for alignment, not prediction
        prediction_horizon=1
    )
    
    # Remove target column (we'll create trading targets from prices)
    if 'target' in features_df.columns:
        features_df = features_df.drop('target', axis=1)
    
    print(f"  Total samples: {len(features_df)}")
    print(f"  Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    
    # Get prices for creating trading targets
    if 'close' in features_df.columns:
        prices_series = features_df['close']
    else:
        raise ValueError("No price data available in features")
    
    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    print(f"  Total features: {len(feature_cols)}")
    
    # Separate features
    X = features_df[feature_cols]
    
    # Remove any remaining NaN values
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    prices_series = prices_series[valid_mask]
    
    print(f"  Valid samples after cleaning: {len(X)}")
    
    return X, prices_series, feature_cols


def train_trading_model(
    ticker: str,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    model_type: str = 'lightgbm',
    max_position_size: int = 100,
    min_profit_threshold: float = 0.001,  # 0.1% for minute-level
    max_holding_minutes: int = 240,  # 4 hours for intraday
    validation_split: float = 0.2,
    model_params: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> TradingModel:
    """
    Train trading model.
    
    Args:
        ticker: Stock ticker
        news_df: News sentiment DataFrame
        ohlcv_df: Historical OHLCV DataFrame
        prices_df: Latest price DataFrame
        quarterly_df: Quarterly reports DataFrame
        model_type: Model type ('lightgbm', 'xgboost', 'random_forest')
        max_position_size: Maximum shares per trade
        min_profit_threshold: Minimum profit % to trigger signal
        max_holding_days: Maximum days to hold position
        validation_split: Validation split
        model_params: Custom model parameters
        save_path: Path to save model
    
    Returns:
        Trained TradingModel
    """
    print("\n" + "="*70)
    print(f"Training Trading Model for {ticker}")
    print("="*70)
    print(f"Model Type: {model_type}")
    print(f"Max Position Size: {max_position_size} shares")
    print(f"Min Profit Threshold: {min_profit_threshold*100:.3f}%")
    print(f"Max Holding Minutes: {max_holding_minutes} minutes ({max_holding_minutes/60:.1f} hours)")
    
    # Prepare data
    X, prices, feature_names = prepare_trading_data(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df
    )
    
    if len(X) == 0:
        raise ValueError("No valid training data available.")
    
    # Initialize model
    print(f"\nInitializing {model_type} trading model...")
    model = TradingModel(
        model_type=model_type,
        max_position_size=max_position_size,
        min_profit_threshold=min_profit_threshold,
        max_holding_minutes=max_holding_minutes,
        model_params=model_params
    )
    
    # Get timestamps if available
    timestamps = None
    if isinstance(X, pd.DataFrame) and 'timestamp' in X.index.names:
        # Try to get timestamps from features_df if available
        pass
    # For now, assume minute-level data (each row = 1 minute)
    
    # Train model
    print("\nTraining trading model...")
    model.fit(
        X=X,
        prices=prices,
        timestamps=timestamps,
        validation_split=validation_split,
        verbose=True
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X, prices, timestamps=timestamps, verbose=True)
    
    # Make sample prediction
    print("\n" + "="*70)
    print("Sample Prediction (Latest Data Point)")
    print("="*70)
    latest_pred = model.predict(X.iloc[-1:])
    print(latest_pred.to_string(index=False))
    
    # Save model
    if save_path:
        model.save(save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model


def get_trading_recommendation(
    model: TradingModel,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    current_price: float,
    ticker: Optional[str] = None
) -> Dict:
    """
    Get trading recommendation for current market conditions.
    
    Args:
        model: Trained TradingModel
        news_df: News DataFrame (can load from CSV if needed)
        ohlcv_df: OHLCV DataFrame
        prices_df: Prices DataFrame
        quarterly_df: Quarterly reports DataFrame
        current_price: Current stock price
        ticker: Optional ticker to reload news if needed
    
    Returns:
        Dictionary with trading recommendation
    """
    # If news_df is empty and ticker provided, try loading from CSV
    if news_df.empty and ticker:
        from load_data import load_news_for_ticker
        news_df = load_news_for_ticker(ticker)
    
    # Prepare features
    features_df = align_dataframes(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        target_column='close',
        prediction_horizon=1
    )
    
    if 'target' in features_df.columns:
        features_df = features_df.drop('target', axis=1)
    
    feature_cols = get_feature_columns(features_df)
    latest_features = features_df[feature_cols].iloc[-1:].fillna(0)
    
    # Predict
    prediction = model.predict(latest_features).iloc[0]
    
    # Format recommendation
    signal = prediction['signal']
    position_size = int(prediction['position_size'])
    timing_minutes = int(prediction['timing_minutes'])
    expected_profit = prediction['expected_profit_pct']
    confidence = prediction['signal_confidence']
    
    # Convert minutes to readable format
    hours = timing_minutes // 60
    minutes = timing_minutes % 60
    timing_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
    
    recommendation = {
        'ticker': ohlcv_df['ticker'].iloc[0] if not ohlcv_df.empty else 'UNKNOWN',
        'current_price': current_price,
        'signal': signal,
        'action': 'BUY' if signal == 'BUY' else ('SELL' if signal == 'SELL' else 'HOLD'),
        'shares': position_size,
        'timing': {
            'minutes': timing_minutes,
            'formatted': timing_str,
            'action': 'SELL' if signal == 'BUY' else ('BUY' if signal == 'SELL' else 'N/A')
        },
        'expected_profit_pct': expected_profit,
        'expected_profit_dollars': (current_price * position_size * expected_profit / 100) if position_size > 0 else 0,
        'confidence': confidence
    }
    
    return recommendation


def print_trading_recommendation(recommendation: Dict):
    """Print formatted trading recommendation."""
    print("\n" + "="*70)
    print(f"TRADING RECOMMENDATION: {recommendation['ticker']}")
    print("="*70)
    print(f"Current Price: ${recommendation['current_price']:.2f}")
    print(f"\nSignal: {recommendation['signal']}")
    print(f"Confidence: {recommendation['confidence']*100:.1f}%")
    
    if recommendation['signal'] != 'HOLD':
        print(f"\nAction: {recommendation['action']} {recommendation['shares']} shares")
        print(f"Total Investment: ${recommendation['current_price'] * recommendation['shares']:.2f}")
        
        if recommendation['signal'] == 'BUY':
            print(f"\nExit Strategy:")
            print(f"  → SELL in {recommendation['timing']['formatted']} ({recommendation['timing']['minutes']} minutes)")
            print(f"  → Expected Profit: {recommendation['expected_profit_pct']:.3f}%")
            print(f"  → Expected Profit: ${recommendation['expected_profit_dollars']:.2f}")
        else:  # SELL
            print(f"\nEntry Strategy:")
            print(f"  → BUY back in {recommendation['timing']['formatted']} ({recommendation['timing']['minutes']} minutes)")
            print(f"  → Expected Profit: {recommendation['expected_profit_pct']:.3f}%")
            print(f"  → Expected Profit: ${recommendation['expected_profit_dollars']:.2f}")
    else:
        print("\nRecommendation: HOLD - No action recommended")
        print("Wait for better entry/exit signals")
    
    print("="*70)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train trading model')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    parser.add_argument('--marketaux_key', type=str, help='Marketaux API key')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                       choices=['lightgbm', 'xgboost', 'random_forest'],
                       help='Model type')
    parser.add_argument('--max_position_size', type=int, default=100,
                       help='Maximum shares per trade')
    parser.add_argument('--min_profit_threshold', type=float, default=0.02,
                       help='Minimum profit % to trigger signal (default: 0.02 = 2%%)')
    parser.add_argument('--max_holding_minutes', type=int, default=240,
                       help='Maximum minutes to hold position (default: 240 = 4 hours)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--save_path', type=str, help='Path to save model')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Set dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        end_date = pytz.UTC.localize(end_date)
    else:
        end_date = datetime.now(pytz.UTC)
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        start_date = pytz.UTC.localize(start_date)
    else:
        start_date = end_date - timedelta(days=730)
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    # Try to load news from CSV first
    print(f"Loading news data from CSV for {args.ticker}...")
    news_df = load_news_for_ticker(args.ticker)
    
    # If CSV loading failed or returned empty, try API
    if news_df.empty and args.marketaux_key:
        print(f"CSV not found or empty. Fetching from Marketaux API...")
        data = ingest_all_data(args.ticker, args.marketaux_key)
        news_df = data['news_df']
    elif news_df.empty:
        print("Warning: No news data available (no CSV found and no API key provided).")
        news_df = pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
    
    # Load other data sources (OHLCV, prices, quarterly)
    # For now, these will be empty until data is available
    if args.marketaux_key:
        print(f"Fetching other data for {args.ticker}...")
        data = ingest_all_data(args.ticker, args.marketaux_key)
        # Don't overwrite news_df if we loaded from CSV
        if news_df.empty:
            news_df = data['news_df']
        ohlcv_df = data['prices_df']
        prices_df = data['latest_price_df']
        quarterly_df = data['financials_df']
    else:
        print("Note: Other data sources (OHLCV, prices, quarterly) not yet available.")
        print("Using empty dataframes for these. They will be added soon.")
        ohlcv_df = pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        prices_df = pd.DataFrame(columns=['timestamp', 'ticker', 'price'])
        quarterly_df = pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                            'operating_cash_flow', 'total_assets', 'total_liabilities'])
    
    print(f"\nData Summary:")
    print(f"  News articles: {len(news_df)}")
    print(f"  OHLCV records: {len(ohlcv_df)}")
    print(f"  Price records: {len(prices_df)}")
    print(f"  Quarterly reports: {len(quarterly_df)}")
    
    # Train model
    model = train_trading_model(
        ticker=args.ticker,
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        model_type=args.model_type,
        max_position_size=args.max_position_size,
        min_profit_threshold=args.min_profit_threshold,
        max_holding_minutes=args.max_holding_minutes,
        validation_split=args.validation_split,
        save_path=args.save_path
    )
    
    # Get current recommendation
    if not ohlcv_df.empty and 'close' in ohlcv_df.columns:
        current_price = ohlcv_df['close'].iloc[-1]
    elif not prices_df.empty and 'price' in prices_df.columns:
        current_price = prices_df['price'].iloc[-1]
    else:
        print("\nWarning: No current price available. Cannot generate recommendation.")
        current_price = None
    
    if current_price:
        recommendation = get_trading_recommendation(
            model=model,
            news_df=news_df,
            ohlcv_df=ohlcv_df,
            prices_df=prices_df,
            quarterly_df=quarterly_df,
            current_price=current_price,
            ticker=args.ticker
        )
        print_trading_recommendation(recommendation)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

