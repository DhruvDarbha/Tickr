"""
Fetch news data for all specific dates and store complete JSON information in DataFrame.
"""

import requests
import pandas as pd
from datetime import datetime
import pytz
import time

def fetch_all_dates_data(ticker: str, api_key: str):
    """
    Fetch news data for all specific dates and store all JSON fields in DataFrame.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        api_key: Marketaux API key
    
    Returns:
        DataFrame with all fields from JSON response
    """
    
    # All 30 specific dates
    specific_dates = [
        '2025-04-09', '2025-05-01', '2025-01-22', '2025-07-31', '2025-04-24',
        '2025-03-05', '2025-01-28', '2024-03-12', '2025-03-14', '2025-01-15',
        '2024-03-14', '2025-05-12', '2024-02-22', '2025-05-27', '2025-05-02',
        '2025-01-30', '2024-10-31', '2024-12-18', '2025-04-16', '2024-07-24',
        '2025-04-04', '2024-05-30', '2025-03-10', '2024-08-05', '2024-04-30',
        '2025-03-28', '2024-03-05', '2025-10-30', '2024-11-15', '2025-12-10'
    ]
    
    base_url = "https://api.marketaux.com/v1/news/all"
    all_articles = []
    
    print(f"Fetching data for {len(specific_dates)} dates...")
    print(f"Ticker: {ticker}\n")
    
    for i, date_str in enumerate(specific_dates, 1):
        print(f"[{i}/{len(specific_dates)}] Fetching data for {date_str}...")
        
        params = {
            'api_token': api_key,
            'symbols': ticker,
            'language': 'en',
            'published_on': date_str,
            'limit': 3,
            'page': 1
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                
                print(f"  Found {len(articles)} articles")
                
                # Process each article - extract ALL fields
                for article in articles:
                    # Extract all top-level fields
                    article_data = {
                        'date_requested': date_str,
                        'uuid': article.get('uuid'),
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'keywords': article.get('keywords'),
                        'snippet': article.get('snippet'),
                        'url': article.get('url'),
                        'image_url': article.get('image_url'),
                        'language': article.get('language'),
                        'published_at': article.get('published_at'),
                        'source': article.get('source'),
                        'ticker': ticker,
                    }
                    
                    # Extract sentiment_score from entity matching our ticker
                    entities = article.get('entities', [])
                    sentiment_score = None
                    for entity in entities:
                        if entity and entity.get('symbol') == ticker:
                            sentiment_score = entity.get('sentiment_score')
                            break
                    
                    # Add sentiment_score (None if not found)
                    article_data['sentiment_score'] = sentiment_score if sentiment_score is not None else None
                    
                    all_articles.append(article_data)
                # Small delay to avoid rate limits
                time.sleep(0.5)
            elif response.status_code == 429:
                print(f"  Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                continue  # Retry same date
            else:
                print(f"  Error {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create DataFrame
    if not all_articles:
        print("\nNo articles found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_articles)
    
    # Convert published_at to datetime
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    
    # Sort by date and published_at
    df = df.sort_values(['date_requested', 'published_at']).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"Total articles collected: {len(df)}")
    print(f"Date range: {df['date_requested'].min()} to {df['date_requested'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*70}\n")
    
    return df


if __name__ == "__main__":
    # Configuration
    ticker = "MSFT"
    api_key = "l0L8d6yPXrWzPn8xVYQd86MELlzFmlSJYkRvki5F"
    
    # Fetch all data
    df = fetch_all_dates_data(ticker, api_key)
    
    # Display DataFrame info
    if not df.empty:
        print("DataFrame Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumn names:")
        print(list(df.columns))
        
        # Save to CSV
        output_file = f"{ticker}_all_dates_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")

