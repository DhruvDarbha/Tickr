"""
Training Data Collection Script

This script efficiently collects 2 years of news data for model training.
It uses date range queries and pagination to maximize data collection
while respecting API rate limits.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import time
import json
import os
from typing import Optional

def fetch_training_news_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    marketaux_api_key: str,
    output_file: Optional[str] = None,
    batch_days: int = 30,
    delay_seconds: float = 1.0,
    max_articles_per_batch: int = 1000
) -> pd.DataFrame:
    """
    Efficiently fetch 2 years of news data for training.
    
    Strategy:
    1. Split 2-year range into monthly batches (30 days each)
    2. For each batch, fetch all articles using pagination
    3. Save incrementally to avoid data loss
    4. Handle rate limits with delays
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date (timezone-aware datetime)
        end_date: End date (timezone-aware datetime)
        marketaux_api_key: Marketaux API key
        output_file: Optional CSV file to save data incrementally
        batch_days: Number of days per batch (default 30 = monthly)
        delay_seconds: Delay between API calls to avoid rate limits
        max_articles_per_batch: Maximum articles to fetch per batch
    
    Returns:
        DataFrame with columns: published_at (UTC), ticker, sentiment
    """
    # Validate inputs
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    if not marketaux_api_key or not isinstance(marketaux_api_key, str) or len(marketaux_api_key.strip()) == 0:
        raise ValueError("marketaux_api_key must be a non-empty string")
    
    ticker = ticker.strip().upper()
    marketaux_api_key = marketaux_api_key.strip()
    
    # Ensure dates are timezone-aware and in UTC
    if start_date.tzinfo is None:
        start_date = pytz.UTC.localize(start_date)
    else:
        start_date = start_date.astimezone(pytz.UTC)
    
    if end_date.tzinfo is None:
        end_date = pytz.UTC.localize(end_date)
    else:
        end_date = end_date.astimezone(pytz.UTC)
    
    if start_date >= end_date:
        raise ValueError(f"start_date must be before end_date")
    
    base_url = "https://api.marketaux.com/v1/news/all"
    all_articles = []
    seen_uuids = set()
    
    # Calculate total days
    total_days = (end_date.date() - start_date.date()).days + 1
    print(f"\n{'='*70}")
    print(f"Training Data Collection for {ticker}")
    print(f"{'='*70}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Total Days: {total_days}")
    print(f"Batch Size: {batch_days} days")
    print(f"Estimated Batches: {(total_days + batch_days - 1) // batch_days}")
    print(f"{'='*70}\n")
    
    # Split into batches
    current_date = start_date.date()
    batch_num = 0
    
    while current_date <= end_date.date():
        batch_num += 1
        batch_end_date = min(current_date + timedelta(days=batch_days - 1), end_date.date())
        
        print(f"\n[Batch {batch_num}] Processing {current_date} to {batch_end_date}")
        print(f"  Progress: {((current_date - start_date.date()).days / total_days * 100):.1f}%")
        
        # Format dates for API (Y-m-d format)
        batch_start_str = current_date.strftime('%Y-%m-%d')
        batch_end_str = batch_end_date.strftime('%Y-%m-%d')
        
        batch_articles = []
        page = 1
        max_pages = 100  # Safety limit
        
        while page <= max_pages:
            params = {
                'api_token': marketaux_api_key,
                'symbols': ticker,
                'published_after': batch_start_str,
                'published_before': batch_end_str,
                'limit': 100,  # Max per page
                'page': page
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check meta info
                    meta = data.get('meta', {})
                    found = meta.get('found', 0)
                    returned = meta.get('returned', 0)
                    
                    if page == 1:
                        print(f"  Found {found} total articles in this batch")
                    
                    articles = data.get('data', [])
                    
                    if not articles:
                        # No more articles
                        break
                    
                    # Process articles
                    for article in articles:
                        article_uuid = article.get('uuid')
                        if not article_uuid or article_uuid in seen_uuids:
                            continue
                        
                        # Parse published_at
                        pub_time_str = article.get('published_at')
                        if not pub_time_str:
                            continue
                        
                        try:
                            pub_time = pd.to_datetime(pub_time_str, utc=True)
                            if pub_time.tzinfo is None:
                                pub_time = pytz.UTC.localize(pub_time.to_pydatetime())
                            else:
                                pub_time = pub_time.astimezone(pytz.UTC)
                        except:
                            continue
                        
                        # Extract sentiment for this ticker
                        sentiment_score = None
                        ticker_found = False
                        entities = article.get('entities', [])
                        
                        for entity in entities:
                            if entity and entity.get('symbol') == ticker:
                                ticker_found = True
                                sentiment_score = entity.get('sentiment_score')
                                break
                        
                        if not ticker_found:
                            continue
                        
                        if sentiment_score is None:
                            sentiment_score = 0.0
                        else:
                            sentiment_score = float(sentiment_score)
                        
                        batch_articles.append({
                            'published_at': pub_time,
                            'ticker': ticker,
                            'sentiment': sentiment_score
                        })
                        
                        seen_uuids.add(article_uuid)
                    
                    print(f"  Page {page}: Retrieved {len(articles)} articles, {len(batch_articles)} unique so far")
                    
                    # Check if more pages available
                    if returned < 100 or page >= (found // 100 + 1):
                        # No more pages
                        break
                    
                    # Check if we've hit max articles for this batch
                    if len(batch_articles) >= max_articles_per_batch:
                        print(f"  Reached max articles limit ({max_articles_per_batch}) for this batch")
                        break
                    
                    page += 1
                    
                    # Delay to avoid rate limits
                    time.sleep(delay_seconds)
                    
                elif response.status_code == 429:
                    # Rate limit - wait longer
                    print(f"  Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue  # Retry same page
                    
                elif response.status_code == 401:
                    raise ValueError("Invalid Marketaux API key")
                    
                else:
                    print(f"  Error {response.status_code}: {response.text[:200]}")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"  Network error: {e}")
                time.sleep(5)
                continue
            
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        # Add batch articles to total
        all_articles.extend(batch_articles)
        print(f"  Batch complete: {len(batch_articles)} articles collected")
        print(f"  Total articles so far: {len(all_articles)}")
        
        # Save incrementally if output file specified
        if output_file and batch_articles:
            batch_df = pd.DataFrame(batch_articles)
            # Append to file (create if doesn't exist)
            if os.path.exists(output_file):
                batch_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                batch_df.to_csv(output_file, mode='w', header=True, index=False)
            print(f"  Saved to {output_file}")
        
        # Move to next batch
        current_date = batch_end_date + timedelta(days=1)
        
        # Delay between batches
        time.sleep(delay_seconds)
    
    # Create final DataFrame
    if not all_articles:
        return pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
    
    news_df = pd.DataFrame(all_articles)
    
    # Clean and sort
    news_df = news_df.drop_duplicates(subset=['published_at'], keep='first')
    news_df = news_df.dropna(subset=['published_at'])
    news_df = news_df.sort_values('published_at').reset_index(drop=True)
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
    news_df['sentiment'] = pd.to_numeric(news_df['sentiment'], errors='coerce').fillna(0.0)
    
    print(f"\n{'='*70}")
    print(f"Collection Complete!")
    print(f"Total Articles: {len(news_df)}")
    print(f"Date Range: {news_df['published_at'].min()} to {news_df['published_at'].max()}")
    print(f"Sentiment Range: {news_df['sentiment'].min():.3f} to {news_df['sentiment'].max():.3f}")
    print(f"Average Sentiment: {news_df['sentiment'].mean():.3f}")
    print(f"{'='*70}\n")
    
    return news_df


def main():
    """Example usage for collecting training data."""
    
    # Configuration
    ticker = "AAPL"
    api_key = "6hV3UI51G9IwfgEZ1Yg7oQBQ6QP5zwCcMF5k3lnR"
    
    # Calculate 2 years back
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=730)
    
    # Output file for incremental saving
    output_file = f"{ticker}_training_news_{start_date.date()}_to_{end_date.date()}.csv"
    
    print(f"Starting training data collection...")
    print(f"Output will be saved to: {output_file}")
    
    # Fetch data
    news_df = fetch_training_news_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        marketaux_api_key=api_key,
        output_file=output_file,
        batch_days=30,  # Monthly batches
        delay_seconds=1.0,  # 1 second delay between calls
        max_articles_per_batch=1000  # Limit per batch
    )
    
    # Final save
    if output_file:
        news_df.to_csv(output_file, index=False)
        print(f"\nFinal data saved to: {output_file}")
    
    # Display summary
    print(f"\nFinal Summary:")
    print(f"  Total Articles: {len(news_df)}")
    print(f"  Unique Dates: {news_df['published_at'].dt.date.nunique()}")
    print(f"  Articles per Day (avg): {len(news_df) / news_df['published_at'].dt.date.nunique():.1f}")
    
    return news_df


if __name__ == "__main__":
    main()

