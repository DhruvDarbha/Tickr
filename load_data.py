"""
Data Loading Utilities

This module provides functions to load data from CSV files and convert them
to the format expected by the ML models.
"""

import pandas as pd
import pytz
from typing import Optional
from pathlib import Path


def load_news_from_csv(csv_path: str, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load news data from CSV file and convert to expected format.
    
    Args:
        csv_path: Path to CSV file (e.g., 'MSFT_all_dates_data.csv')
        ticker: Optional ticker to filter by (if None, uses ticker from CSV)
    
    Returns:
        DataFrame with columns: [published_at, ticker, sentiment]
        Ready for use in feature engineering
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['published_at', 'ticker', 'sentiment_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Convert to expected format
    news_df = pd.DataFrame({
        'published_at': df['published_at'],
        'ticker': df['ticker'],
        'sentiment': df['sentiment_score']  # Rename sentiment_score to sentiment
    })
    
    # Filter by ticker if specified
    if ticker:
        news_df = news_df[news_df['ticker'] == ticker.upper()].copy()
    
    # Drop rows with missing published_at
    news_df = news_df.dropna(subset=['published_at'])
    
    # Convert published_at to datetime with UTC timezone
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
    
    # Ensure sentiment is numeric
    news_df['sentiment'] = pd.to_numeric(news_df['sentiment'], errors='coerce').fillna(0.0)
    
    # Deduplicate based on published_at (same timestamp = duplicate)
    news_df = news_df.drop_duplicates(subset=['published_at'], keep='first')
    
    # Sort by published_at ascending
    news_df = news_df.sort_values('published_at').reset_index(drop=True)
    
    print(f"Loaded {len(news_df)} news articles from {csv_path}")
    if len(news_df) > 0:
        print(f"  Date range: {news_df['published_at'].min()} to {news_df['published_at'].max()}")
        print(f"  Sentiment range: {news_df['sentiment'].min():.3f} to {news_df['sentiment'].max():.3f}")
        print(f"  Average sentiment: {news_df['sentiment'].mean():.3f}")
    
    return news_df


def load_quarterly_from_pdfs(ticker: str = 'MSFT', data_dir: str = '.') -> pd.DataFrame:
    """
    Load all quarterly financial data from PDF reports with scoring.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing PDF files
    
    Returns:
        DataFrame with columns: [report_date, ticker, revenue, net_income, eps,
                                operating_cash_flow, total_assets, total_liabilities,
                                stock_impact_score]
    """
    from extract_all_quarterly_data import extract_all_quarterly_data
    
    # Extract all quarterly data with scoring
    quarterly_df = extract_all_quarterly_data()
    
    # Filter by ticker if needed
    if ticker:
        quarterly_df = quarterly_df[quarterly_df['ticker'] == ticker.upper()].copy()
    
    # Select columns needed for feature engineering + stock impact score
    required_cols = ['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                   'operating_cash_flow', 'total_assets', 'total_liabilities', 
                   'stock_impact_score']
    
    # Keep base columns + stock impact score
    quarterly_df_base = quarterly_df[required_cols].copy()
    
    print(f"\nLoaded {len(quarterly_df)} quarterly reports for {ticker}")
    if len(quarterly_df) > 0:
        print(f"  Date range: {quarterly_df['report_date'].min()} to {quarterly_df['report_date'].max()}")
        print(f"  Average Financial Health Score: {quarterly_df['financial_health_score'].mean():.1f}/100")
        print(f"  Latest Stock Impact: {quarterly_df['stock_impact'].iloc[-1]}")
    
    return quarterly_df_base


def load_news_for_ticker(ticker: str, data_dir: str = '.') -> pd.DataFrame:
    """
    Load news data for a specific ticker from CSV file.
    
    Looks for file: {ticker}_all_dates_data.csv
    
    Args:
        ticker: Stock ticker symbol (e.g., 'MSFT')
        data_dir: Directory containing CSV files (default: current directory)
    
    Returns:
        DataFrame with columns: [published_at, ticker, sentiment]
    """
    csv_path = Path(data_dir) / f"{ticker.upper()}_all_dates_data.csv"
    
    if not csv_path.exists():
        print(f"Warning: News CSV file not found: {csv_path}")
        print(f"Returning empty DataFrame. Expected file: {csv_path}")
        return pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
    
    return load_news_from_csv(str(csv_path), ticker=ticker)

