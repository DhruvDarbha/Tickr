"""
Feature Engineering Module

This module combines all 4 dataframes (news, OHLCV, prices, quarterly reports)
into a unified feature set for ML model training.

Features:
- Temporal alignment of all data sources
- Technical indicators from OHLCV
- Sentiment aggregation from news
- Fundamental metrics from quarterly reports
- Time-based features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pytz
import warnings

warnings.filterwarnings('ignore')


def align_dataframes(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    target_column: str = 'close',
    prediction_horizon: int = 1
) -> pd.DataFrame:
    """
    Align all dataframes temporally and create target variable.
    
    Args:
        news_df: DataFrame with columns [published_at, ticker, sentiment]
        ohlcv_df: DataFrame with columns [timestamp, ticker, open, high, low, close, volume]
        prices_df: DataFrame with columns [timestamp, ticker, price]
        quarterly_df: DataFrame with columns [report_date, ticker, revenue, net_income, eps, ...]
        target_column: Column to predict (e.g., 'close' for price prediction)
        prediction_horizon: Number of days ahead to predict (default: 1 day)
    
    Returns:
        Aligned DataFrame with all features and target variable
    """
    # Start with OHLCV as base (most frequent data)
    if ohlcv_df.empty:
        raise ValueError("OHLCV dataframe cannot be empty")
    
    # Ensure timestamps are datetime and timezone-aware
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], utc=True)
    ohlcv_df = ohlcv_df.sort_values('timestamp').reset_index(drop=True)
    
    # Create base feature dataframe
    features_df = ohlcv_df[['timestamp', 'ticker']].copy()
    
    # Add OHLCV features
    features_df = features_df.merge(
        ohlcv_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
        on='timestamp',
        how='left'
    )
    
    # Add technical indicators from OHLCV
    features_df = add_technical_indicators(features_df)
    
    # Align and add news sentiment features
    if not news_df.empty:
        news_features = aggregate_news_sentiment(news_df, features_df['timestamp'].unique())
        features_df = features_df.merge(news_features, on='timestamp', how='left')
    else:
        # Add empty news columns
        features_df['sentiment_mean'] = 0.0
        features_df['sentiment_count'] = 0
        features_df['sentiment_std'] = 0.0
        features_df['sentiment_max'] = 0.0
        features_df['sentiment_min'] = 0.0
    
    # Align and add latest price features
    if not prices_df.empty:
        price_features = align_price_data(prices_df, features_df['timestamp'].unique())
        features_df = features_df.merge(price_features, on='timestamp', how='left')
    else:
        features_df['latest_price'] = features_df['close']
    
    # Align and add quarterly report features
    if not quarterly_df.empty:
        quarterly_features = align_quarterly_data(quarterly_df, features_df['timestamp'].unique())
        features_df = features_df.merge(quarterly_features, on='timestamp', how='left')
    else:
        # Add empty quarterly columns
        features_df['revenue'] = np.nan
        features_df['net_income'] = np.nan
        features_df['eps'] = np.nan
        features_df['operating_cash_flow'] = np.nan
        features_df['total_assets'] = np.nan
        features_df['total_liabilities'] = np.nan
    
    # Add time-based features
    features_df = add_time_features(features_df)
    
    # Create target variable (future price)
    if target_column in features_df.columns:
        features_df['target'] = features_df.groupby('ticker')[target_column].shift(-prediction_horizon)
    else:
        features_df['target'] = np.nan
    
    # Remove rows where target is NaN (last N rows where we can't predict)
    features_df = features_df.dropna(subset=['target'])
    
    # Fill missing values
    features_df = features_df.ffill().bfill().fillna(0)
    
    # Sort by timestamp
    features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    
    return features_df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Ensure sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Moving averages
    df['sma_5'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=5).mean())
    df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean())
    df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50).mean())
    
    # Exponential moving averages
    df['ema_12'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12).mean())
    df['ema_26'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=26).mean())
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9).mean())
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df.groupby('ticker')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).mean())
    bb_std = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).std())
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price changes
    df['price_change'] = df.groupby('ticker')['close'].pct_change()
    df['price_change_5d'] = df.groupby('ticker')['close'].pct_change(periods=5)
    df['price_change_10d'] = df.groupby('ticker')['close'].pct_change(periods=10)
    
    # Volume indicators
    df['volume_sma'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility_5d'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=5).std())
    df['volatility_20d'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).std())
    
    # High-Low spread
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['oc_spread'] = (df['close'] - df['open']) / df['open']
    
    return df


def aggregate_news_sentiment(
    news_df: pd.DataFrame,
    timestamps: np.ndarray,
    window_hours: int = 24
) -> pd.DataFrame:
    """
    Aggregate news sentiment for each timestamp.
    
    Args:
        news_df: DataFrame with [published_at, ticker, sentiment]
        timestamps: Array of timestamps to aggregate for
        window_hours: Hours before each timestamp to include news
    
    Returns:
        DataFrame with aggregated sentiment features per timestamp
    """
    if news_df.empty:
        return pd.DataFrame({
            'timestamp': timestamps,
            'sentiment_mean': 0.0,
            'sentiment_count': 0,
            'sentiment_std': 0.0,
            'sentiment_max': 0.0,
            'sentiment_min': 0.0
        })
    
    news_df = news_df.copy()
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
    
    sentiment_features = []
    
    for ts in timestamps:
        ts = pd.to_datetime(ts, utc=True)
        window_start = ts - timedelta(hours=window_hours)
        
        # Get news in window
        window_news = news_df[
            (news_df['published_at'] >= window_start) &
            (news_df['published_at'] < ts)
        ]
        
        if len(window_news) > 0:
            sentiment_features.append({
                'timestamp': ts,
                'sentiment_mean': window_news['sentiment'].mean(),
                'sentiment_count': len(window_news),
                'sentiment_std': window_news['sentiment'].std(),
                'sentiment_max': window_news['sentiment'].max(),
                'sentiment_min': window_news['sentiment'].min()
            })
        else:
            sentiment_features.append({
                'timestamp': ts,
                'sentiment_mean': 0.0,
                'sentiment_count': 0,
                'sentiment_std': 0.0,
                'sentiment_max': 0.0,
                'sentiment_min': 0.0
            })
    
    return pd.DataFrame(sentiment_features)


def align_price_data(
    prices_df: pd.DataFrame,
    timestamps: np.ndarray
) -> pd.DataFrame:
    """
    Align latest price data with timestamps using forward fill.
    
    Args:
        prices_df: DataFrame with [timestamp, ticker, price]
        timestamps: Array of timestamps to align to
    
    Returns:
        DataFrame with price aligned to timestamps
    """
    if prices_df.empty:
        return pd.DataFrame({'timestamp': timestamps, 'latest_price': np.nan})
    
    prices_df = prices_df.copy()
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], utc=True)
    prices_df = prices_df.sort_values('timestamp')
    
    # Create full timestamp range
    price_aligned = pd.DataFrame({'timestamp': pd.to_datetime(timestamps, utc=True)})
    
    # Merge and forward fill
    price_aligned = price_aligned.merge(
        prices_df[['timestamp', 'price']],
        on='timestamp',
        how='left'
    )
    price_aligned['latest_price'] = price_aligned['price'].ffill()
    price_aligned = price_aligned[['timestamp', 'latest_price']]
    
    return price_aligned


def align_quarterly_data(
    quarterly_df: pd.DataFrame,
    timestamps: np.ndarray
) -> pd.DataFrame:
    """
    Align quarterly report data with timestamps using forward fill.
    
    Args:
        quarterly_df: DataFrame with [report_date, ticker, revenue, net_income, eps, ...]
        timestamps: Array of timestamps to align to
    
    Returns:
        DataFrame with quarterly data aligned to timestamps
    """
    if quarterly_df.empty:
        return pd.DataFrame({'timestamp': timestamps})
    
    quarterly_df = quarterly_df.copy()
    quarterly_df['report_date'] = pd.to_datetime(quarterly_df['report_date'], utc=True)
    quarterly_df = quarterly_df.sort_values('report_date')
    
    # Create full timestamp range
    quarterly_aligned = pd.DataFrame({'timestamp': pd.to_datetime(timestamps, utc=True)})
    quarterly_aligned = quarterly_aligned.sort_values('timestamp').reset_index(drop=True)
    
    # Merge and forward fill quarterly data
    quarterly_cols = [col for col in quarterly_df.columns if col not in ['report_date', 'ticker']]
    
    # For each quarterly column, merge using merge_asof (forward fill from report date)
    for col in quarterly_cols:
        temp_df = quarterly_df[['report_date', col]].copy()
        temp_df = temp_df.sort_values('report_date')
        
        # Use merge_asof to forward fill: for each timestamp, get the most recent quarterly value
        quarterly_aligned = pd.merge_asof(
            quarterly_aligned,
            temp_df.rename(columns={'report_date': 'timestamp'}),
            on='timestamp',
            direction='backward'  # Get the most recent quarterly report on or before this timestamp
        )
    
    return quarterly_aligned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Extract time components
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['quarter'] = df['timestamp'].dt.quarter
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Is market open (simplified: Monday-Friday)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding target and metadata).
    
    Args:
        df: Feature DataFrame
    
    Returns:
        List of feature column names
    """
    exclude_cols = ['timestamp', 'ticker', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

