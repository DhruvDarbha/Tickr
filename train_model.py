"""
Training Pipeline for Stock Prediction Model

This module provides a complete training pipeline that:
1. Loads all 4 dataframes (news, OHLCV, prices, quarterly)
2. Engineers features from all data sources
3. Trains ML models (LightGBM, XGBoost, or ensemble)
4. Evaluates and saves models
5. Provides prediction interface

Usage:
    python train_model.py --ticker AAPL --model_type lightgbm
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import pytz
import argparse
import warnings

from feature_engineering import align_dataframes, get_feature_columns
from ml_model import StockPredictionModel
from data_ingestion import ingest_all_data

warnings.filterwarnings('ignore')


def prepare_training_data(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    prediction_horizon: int = 1,
    target_column: str = 'close'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare training data from all 4 dataframes.
    
    Args:
        news_df: News sentiment DataFrame
        ohlcv_df: Historical OHLCV DataFrame
        prices_df: Latest price DataFrame
        quarterly_df: Quarterly reports DataFrame
        prediction_horizon: Days ahead to predict
        target_column: Column to predict
    
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    print("\n" + "="*70)
    print("Preparing Training Data")
    print("="*70)
    
    # Align all dataframes
    print("Aligning dataframes and engineering features...")
    features_df = align_dataframes(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        target_column=target_column,
        prediction_horizon=prediction_horizon
    )
    
    print(f"  Total samples: {len(features_df)}")
    print(f"  Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    
    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    print(f"  Total features: {len(feature_cols)}")
    
    # Separate features and target
    X = features_df[feature_cols]
    y = features_df['target']
    
    # Remove any remaining NaN values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"  Valid samples after cleaning: {len(X)}")
    
    return X, y, feature_cols


def train_stock_model(
    ticker: str,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    model_type: str = 'lightgbm',
    use_ensemble: bool = False,
    prediction_horizon: int = 1,
    validation_split: float = 0.2,
    model_params: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> StockPredictionModel:
    """
    Complete training pipeline.
    
    Args:
        ticker: Stock ticker symbol
        news_df: News sentiment DataFrame
        ohlcv_df: Historical OHLCV DataFrame
        prices_df: Latest price DataFrame
        quarterly_df: Quarterly reports DataFrame
        model_type: Type of model ('lightgbm', 'xgboost', 'random_forest', 'ensemble')
        use_ensemble: Whether to use ensemble
        prediction_horizon: Days ahead to predict
        validation_split: Fraction for validation
        model_params: Custom model parameters
        save_path: Path to save trained model
    
    Returns:
        Trained StockPredictionModel
    """
    print("\n" + "="*70)
    print(f"Training Stock Prediction Model for {ticker}")
    print("="*70)
    print(f"Model Type: {model_type}")
    print(f"Ensemble: {use_ensemble}")
    print(f"Prediction Horizon: {prediction_horizon} day(s)")
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        prediction_horizon=prediction_horizon
    )
    
    if len(X) == 0:
        raise ValueError("No valid training data available. Check your dataframes.")
    
    # Initialize model
    print(f"\nInitializing {model_type} model...")
    model = StockPredictionModel(
        model_type=model_type,
        model_params=model_params,
        use_ensemble=use_ensemble
    )
    
    # Train model
    print("\nTraining model...")
    model.fit(
        X=X,
        y=y,
        validation_split=validation_split,
        early_stopping_rounds=10,
        verbose=True
    )
    
    # Evaluate on training data
    print("\nEvaluating model on training data...")
    train_metrics = model.evaluate(X, y, verbose=True)
    
    # Show feature importance
    print("\nTop 20 Most Important Features:")
    print("-" * 70)
    importance_df = model.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))
    
    # Save model if path provided
    if save_path:
        model.save(save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model


def predict_future_price(
    model: StockPredictionModel,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    prediction_horizon: int = 1
) -> float:
    """
    Predict future price using trained model.
    
    Args:
        model: Trained StockPredictionModel
        news_df: News sentiment DataFrame
        ohlcv_df: Historical OHLCV DataFrame
        prices_df: Latest price DataFrame
        quarterly_df: Quarterly reports DataFrame
        prediction_horizon: Days ahead to predict
    
    Returns:
        Predicted price
    """
    # Prepare features for latest data point
    features_df = align_dataframes(
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        target_column='close',
        prediction_horizon=prediction_horizon
    )
    
    # Get latest row (most recent data)
    feature_cols = get_feature_columns(features_df)
    latest_features = features_df[feature_cols].iloc[-1:].fillna(0)
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    
    return prediction


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--marketaux_key', type=str, help='Marketaux API key (for news data)')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                       choices=['lightgbm', 'xgboost', 'random_forest', 'ensemble'],
                       help='Type of ML model')
    parser.add_argument('--use_ensemble', action='store_true',
                       help='Use ensemble of multiple models')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='Days ahead to predict')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of data for validation')
    parser.add_argument('--save_path', type=str, help='Path to save trained model')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Set default dates (2 years back)
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
    
    if args.marketaux_key:
        # Use data ingestion pipeline
        print(f"Fetching data for {args.ticker}...")
        data = ingest_all_data(args.ticker, args.marketaux_key)
        news_df = data['news_df']
        ohlcv_df = data['prices_df']  # Historical OHLCV
        prices_df = data['latest_price_df']
        quarterly_df = data['financials_df']
    else:
        # For now, create empty dataframes (will be populated later)
        print("Warning: No Marketaux API key provided. Using empty dataframes.")
        print("Note: This is a placeholder. Once data is ready, provide the API key.")
        news_df = pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
        ohlcv_df = pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        prices_df = pd.DataFrame(columns=['timestamp', 'ticker', 'price'])
        quarterly_df = pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                            'operating_cash_flow', 'total_assets', 'total_liabilities'])
    
    print(f"  News articles: {len(news_df)}")
    print(f"  OHLCV records: {len(ohlcv_df)}")
    print(f"  Price records: {len(prices_df)}")
    print(f"  Quarterly reports: {len(quarterly_df)}")
    
    # Train model
    model = train_stock_model(
        ticker=args.ticker,
        news_df=news_df,
        ohlcv_df=ohlcv_df,
        prices_df=prices_df,
        quarterly_df=quarterly_df,
        model_type=args.model_type,
        use_ensemble=args.use_ensemble,
        prediction_horizon=args.prediction_horizon,
        validation_split=args.validation_split,
        save_path=args.save_path
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

