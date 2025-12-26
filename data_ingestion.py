"""
Phase 1: Data Ingestion Pipeline

This module provides functions to retrieve and clean all raw data needed
for stock analysis and trading recommendations.

All functions return clean Pandas DataFrames with UTC timezone-aware timestamps.
No data leakage, no joins, no feature engineering - just clean raw data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import pytz
import requests
import random
import warnings

warnings.filterwarnings('ignore')


def fetch_historical_prices(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Retrieve historical OHLCV price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date (timezone-aware datetime)
        end_date: End date (timezone-aware datetime)
        interval: Data interval ("1d" for daily)
    
    Returns:
        DataFrame with columns: timestamp, ticker, open, high, low, close, volume
        All timestamps in UTC, sorted ascending, no duplicates, no incomplete bars.
    """
    # Ensure dates are timezone-aware and in UTC
    if start_date.tzinfo is None:
        start_date = pytz.UTC.localize(start_date)
    else:
        start_date = start_date.astimezone(pytz.UTC)
    
    if end_date.tzinfo is None:
        end_date = pytz.UTC.localize(end_date)
    else:
        end_date = end_date.astimezone(pytz.UTC)
    
    # Validate interval
    if interval != "1d":
        raise ValueError(f"Interval '{interval}' not yet supported. Only '1d' is supported.")
    
    # Validate date range
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    
    # Validate ticker
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    try:
        # Fetch data using yfinance (automatically adjusts for splits/dividends)
        stock = yf.Ticker(ticker)
        
        # yfinance expects naive datetime or string dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        hist = stock.history(start=start_str, end=end_str, interval=interval, auto_adjust=True)
        
        if hist.empty:
            raise ValueError(f"No price data found for ticker {ticker} between {start_str} and {end_str}")
        
        # Convert to DataFrame and clean
        prices_df = pd.DataFrame(hist)
        
        # Reset index to get Date as column
        prices_df.reset_index(inplace=True)
        
        # Check if required columns exist (handle different yfinance versions)
        required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        available_cols = set(prices_df.columns)
        
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            raise ValueError(f"Missing required columns from yfinance: {missing}")
        
        # Rename columns to match required format
        prices_df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Ensure timestamp is timezone-aware UTC
        if prices_df['timestamp'].dt.tz is None:
            # yfinance returns dates in market timezone, convert to UTC
            # For US markets, assume Eastern time
            et = pytz.timezone('US/Eastern')
            prices_df['timestamp'] = prices_df['timestamp'].dt.tz_localize(et).dt.tz_convert(pytz.UTC)
        else:
            prices_df['timestamp'] = prices_df['timestamp'].dt.tz_convert(pytz.UTC)
        
        # Add ticker column
        prices_df['ticker'] = ticker
        
        # Select and reorder columns (verify they exist)
        required_output_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_output_cols if col not in prices_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns after processing: {missing_cols}")
        
        prices_df = prices_df[required_output_cols]
        
        # Remove incomplete bars (any NaN values)
        prices_df = prices_df.dropna()
        
        # Check if DataFrame became empty after cleaning
        if prices_df.empty:
            return pd.DataFrame(columns=required_output_cols)
        
        # Remove duplicate timestamps (keep first)
        prices_df = prices_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp ascending
        prices_df = prices_df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure one row = one completed bar (remove any future dates)
        now_utc = datetime.now(pytz.UTC)
        prices_df = prices_df[prices_df['timestamp'] <= now_utc]
        
        # Ensure data types are correct
        prices_df['ticker'] = prices_df['ticker'].astype(str)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')
        
        # Final drop of any rows with NaN after type conversion
        prices_df = prices_df.dropna()
        
        return prices_df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching historical prices for {ticker}: {str(e)}")


def fetch_latest_price(ticker: str) -> pd.DataFrame:
    """
    Retrieve the most recent completed price for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        DataFrame with columns: timestamp (UTC), ticker, price
        Single row with the most recent completed price.
    """
    # Validate ticker
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if info is None or empty
        if info is None or not isinstance(info, dict) or len(info) == 0:
            # Fallback to history
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                timestamp = hist.index[-1]
            else:
                raise ValueError(f"Could not retrieve price for {ticker}: info is empty and history is empty")
        else:
            # Get current price
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        
        if current_price is None:
            # Fallback: get last close from recent history
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                timestamp = hist.index[-1]
            else:
                raise ValueError(f"Could not retrieve price for {ticker}")
        else:
            # Get timestamp from info or use current UTC time
            timestamp = datetime.now(pytz.UTC)
        
        # Ensure timestamp is timezone-aware UTC
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tz is None:
                et = pytz.timezone('US/Eastern')
                timestamp = timestamp.tz_localize(et).tz_convert(pytz.UTC)
            else:
                timestamp = timestamp.tz_convert(pytz.UTC)
            timestamp = timestamp.to_pydatetime()
        elif isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(pytz.UTC)
        
        # Return as DataFrame
        latest_price_df = pd.DataFrame([{
            'timestamp': timestamp,
            'ticker': ticker,
            'price': float(current_price)
        }])
        
        return latest_price_df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching latest price for {ticker}: {str(e)}")


def fetch_company_news(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    marketaux_api_key: str
) -> pd.DataFrame:
    """
    Retrieve company-specific news articles with sentiment scores from Marketaux API.
    
    Fetches 2 articles from each specified date (from images) and articles from 30 random days.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date (timezone-aware datetime)
        end_date: End date (timezone-aware datetime)
        marketaux_api_key: Marketaux API key
    
    Returns:
        DataFrame with columns: published_at (UTC), ticker, sentiment
        All timestamps in UTC, sorted ascending, deduplicated.
        Sentiment scores from Marketaux API.
    """
    # Validate inputs
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    if not marketaux_api_key or not isinstance(marketaux_api_key, str) or len(marketaux_api_key.strip()) == 0:
        raise ValueError("marketaux_api_key must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    # Ensure dates are timezone-aware and in UTC
    if start_date.tzinfo is None:
        start_date = pytz.UTC.localize(start_date)
    else:
        start_date = start_date.astimezone(pytz.UTC)
    
    if end_date.tzinfo is None:
        end_date = pytz.UTC.localize(end_date)
    else:
        end_date = end_date.astimezone(pytz.UTC)
    
    # Validate date range
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    
    # Specific dates from images (combine both lists)
    specific_dates = [
        '2025-04-09', '2025-05-01', '2025-01-22', '2025-07-31', '2025-04-24',
        '2025-03-05', '2025-01-28', '2024-03-12', '2025-03-14', '2025-01-15',
        '2024-03-14', '2025-05-12', '2024-02-22', '2025-05-27', '2025-05-02',
        '2025-01-30', '2024-10-31', '2024-12-18', '2025-04-16', '2024-07-24',
        '2025-04-04', '2024-05-30', '2025-03-10', '2024-08-05', '2024-04-30',
        '2025-03-28', '2024-03-05', '2025-10-30', '2024-11-15', '2025-12-10'
    ]
    
    # Convert to datetime objects and filter to date range
    specific_date_objs = []
    for date_str in specific_dates:
        try:
            # Parse date string to date object
            dt = pd.to_datetime(date_str).date()
            # Check if date is within range
            if start_date.date() <= dt <= end_date.date():
                specific_date_objs.append(dt)
        except Exception as e:
            continue
    
    # Get unique dates and sort
    specific_date_objs = sorted(list(set(specific_date_objs)))
    
    # Generate 30 random days within the date range
    date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
    # Exclude specific dates from random selection
    available_dates = [d.date() for d in date_range if d.date() not in specific_date_objs]
    
    # Handle empty available_dates to avoid random.sample error
    if len(available_dates) == 0:
        random_dates = []
    elif len(available_dates) > 30:
        random_dates = random.sample(available_dates, 30)
    else:
        random_dates = available_dates
    
    # Combine all target dates
    all_target_dates = specific_date_objs + random_dates
    all_target_dates = sorted(list(set(all_target_dates)))  # Remove duplicates and sort
    
    # If no target dates (all filtered out), use a broader approach - fetch from entire date range
    if len(all_target_dates) == 0:
        # Fallback: use the entire date range and sample dates
        date_range_list = [start_date.date() + timedelta(days=x) for x in range((end_date.date() - start_date.date()).days + 1)]
        # Sample up to 60 days if range is too large
        if len(date_range_list) > 60:
            all_target_dates = random.sample(date_range_list, 60)
        else:
            all_target_dates = date_range_list
    
    base_url = "https://api.marketaux.com/v1/news/all"
    all_articles = []
    seen_uuids = set()  # Track article UUIDs to avoid duplicates
    
    try:
        # Fetch articles for each target date
        for target_date in all_target_dates:
            # For specific dates from images, get exactly 2 articles; for random dates, get up to 2
            is_specific_date = target_date in specific_date_objs
            articles_needed = 2  # Always try to get 2 articles per day
            articles_found = 0
            
            # Use published_on for single day queries (more efficient per Marketaux API docs)
            # Format: Y-m-d (e.g., "2024-12-26")
            day_str = target_date.strftime('%Y-%m-%d')
            
            page = 1
            max_pages_per_date = 3  # Limit pages per date to avoid too many API calls
            while articles_found < articles_needed and page <= max_pages_per_date:
                params = {
                    'api_token': marketaux_api_key,
                    'symbols': ticker,
                    'published_on': day_str,  # Use published_on for single day queries
                    'limit': 100,
                    'page': page
                }
                
                # DEBUG: Print request parameters
                print(f"\n[DEBUG] Requesting news for {ticker} on {day_str} (page {page})")
                print(f"  URL: {base_url}")
                print(f"  Params: api_token=***, symbols={ticker}, published_on={day_str}, limit=100, page={page}")
                
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                except requests.exceptions.RequestException as e:
                    # Network error, skip this date
                    print(f"[DEBUG] Network error for {day_str}: {e}")
                    break
                
                if response.status_code != 200:
                    print(f"[DEBUG] Non-200 status code: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    if response.status_code == 401:
                        raise ValueError(f"Invalid Marketaux API key. Please check your API key.")
                    elif response.status_code == 429:
                        # Rate limit - skip this date and continue to next
                        print(f"[DEBUG] Rate limit hit for {day_str}, skipping...")
                        break
                    else:
                        # If error, skip this date and continue
                        print(f"[DEBUG] Error {response.status_code} for {day_str}, skipping...")
                        break
                
                # Check if response is valid JSON
                try:
                    data = response.json()
                    # DEBUG: Print JSON response to verify data is coming
                    print(f"\n[DEBUG] API Response for {target_date} (page {page}):")
                    print(f"  Status: {response.status_code}")
                    print(f"  Full JSON response: {data}")
                    print(f"  Meta found: {data.get('meta', {}).get('found', 'N/A')}")
                    print(f"  Meta returned: {data.get('meta', {}).get('returned', 'N/A')}")
                    print(f"  Articles in data: {len(data.get('data', []))}")
                    if 'data' in data and len(data.get('data', [])) > 0:
                        print(f"  First article title: {data['data'][0].get('title', 'N/A')[:50]}...")
                        print(f"  First article entities: {data['data'][0].get('entities', [])}")
                except ValueError as e:
                    # Response is not valid JSON (might be HTML error page)
                    print(f"\n[DEBUG] Invalid JSON response for {target_date}: {response.text[:200]}")
                    break
                
                if 'data' not in data:
                    print(f"\n[DEBUG] No 'data' key in response for {target_date}")
                    print(f"  Response keys: {list(data.keys())}")
                    break
                
                articles = data.get('data', [])
                if not articles:
                    print(f"\n[DEBUG] Empty articles array for {target_date} (page {page})")
                    # Print full JSON to see what we got
                    print(f"  Full JSON response: {data}")
                    break
                
                print(f"  Processing {len(articles)} articles from response...")
                
                # Process articles
                for article in articles:
                    if articles_found >= articles_needed:
                        break
                    
                    # Check for duplicate UUID
                    article_uuid = article.get('uuid')
                    if not article_uuid or article_uuid in seen_uuids:
                        continue
                    
                    # Parse published_at timestamp (format: "2024-11-08T01:24:00.000000Z")
                    pub_time_str = article.get('published_at')
                    if not pub_time_str:
                        continue
                    
                    # Parse ISO 8601 timestamp (Z indicates UTC)
                    try:
                        # pd.to_datetime handles the Z timezone indicator correctly
                        pub_time = pd.to_datetime(pub_time_str, utc=True)
                        # Ensure it's timezone-aware UTC
                        if pub_time.tzinfo is None:
                            pub_time = pytz.UTC.localize(pub_time.to_pydatetime())
                        else:
                            pub_time = pub_time.astimezone(pytz.UTC)
                    except Exception:
                        continue
                    
                    # Verify it's on the target date (compare date components only)
                    if pub_time.date() != target_date:
                        continue
                    
                    # Extract sentiment score from entities array
                    # JSON structure: entities is an array, each entity has symbol and sentiment_score
                    sentiment_score = None
                    ticker_found = False
                    entities = article.get('entities', [])
                    
                    # Find entity matching our ticker
                    for entity in entities:
                        if entity and entity.get('symbol') == ticker:
                            ticker_found = True
                            sentiment_score = entity.get('sentiment_score')
                            break
                    
                    # If ticker not found in entities, skip this article
                    # (API should filter by symbol, but this is a safety check)
                    if not ticker_found:
                        continue
                    
                    # Handle sentiment_score: can be None, 0 (int), or float like 0.7783
                    # Default to 0.0 if None (article mentions ticker but no sentiment calculated)
                    if sentiment_score is None:
                        sentiment_score = 0.0
                    else:
                        # Convert to float (handles both int 0 and float like 0.7783)
                        sentiment_score = float(sentiment_score)
                    
                    # Add article to results
                    all_articles.append({
                        'published_at': pub_time,
                        'ticker': ticker,
                        'sentiment': sentiment_score
                    })
                    
                    seen_uuids.add(article_uuid)
                    articles_found += 1
                
                # Check pagination
                if 'meta' in data:
                    current_page = data['meta'].get('page', 1)
                    total_pages = data['meta'].get('total_pages', 1)
                    if current_page >= total_pages or articles_found >= articles_needed:
                        break
                    page += 1
                else:
                    break
            
            # If we didn't get enough articles, try to get more from the date range
            # (This helps when specific dates have no articles)
            if articles_found < articles_needed and page == 1:
                # Try one more page if we got some articles but not enough
                pass
        
        # If no articles found from specific/random dates, try fetching from entire date range
        if not all_articles:
            # Fallback: fetch articles from entire date range (up to 100 articles)
            print(f"\n[DEBUG] No articles found from specific dates, trying fallback for entire date range...")
            try:
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                params = {
                    'api_token': marketaux_api_key,
                    'symbols': ticker,
                    'published_after': start_str,
                    'published_before': end_str,
                    'limit': 100,
                    'page': 1
                }
                
                print(f"[DEBUG] Fallback request: symbols={ticker}, published_after={start_str}, published_before={end_str}")
                response = requests.get(base_url, params=params, timeout=30)
                print(f"[DEBUG] Fallback response status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # DEBUG: Print fallback API response
                        print(f"\n[DEBUG] Fallback API Response (entire date range):")
                        print(f"  Status: {response.status_code}")
                        print(f"  Full JSON response: {data}")
                        print(f"  Meta found: {data.get('meta', {}).get('found', 'N/A')}")
                        print(f"  Meta returned: {data.get('meta', {}).get('returned', 'N/A')}")
                        print(f"  Articles in data: {len(data.get('data', []))}")
                        if 'data' in data and len(data.get('data', [])) > 0:
                            print(f"  First article title: {data['data'][0].get('title', 'N/A')[:50]}...")
                            print(f"  First article entities: {data['data'][0].get('entities', [])}")
                        articles = data.get('data', [])
                        
                        for article in articles[:100]:  # Limit to 100 articles
                            article_uuid = article.get('uuid')
                            if not article_uuid or article_uuid in seen_uuids:
                                continue
                            
                            pub_time_str = article.get('published_at')
                            if not pub_time_str:
                                continue
                            
                            try:
                                pub_time = pd.to_datetime(pub_time_str, utc=True)
                                if pub_time.tzinfo is None:
                                    pub_time = pytz.UTC.localize(pub_time.to_pydatetime())
                                else:
                                    pub_time = pub_time.astimezone(pytz.UTC)
                            except Exception:
                                continue
                            
                            # Extract sentiment
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
                            
                            all_articles.append({
                                'published_at': pub_time,
                                'ticker': ticker,
                                'sentiment': sentiment_score
                            })
                            
                            seen_uuids.add(article_uuid)
                    except ValueError:
                        pass  # Invalid JSON, skip
            except Exception:
                pass  # If fallback fails, return empty DataFrame
        
        if not all_articles:
            return pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
        
        news_df = pd.DataFrame(all_articles)
        
        # Deduplicate based on published_at (same timestamp = duplicate)
        news_df = news_df.drop_duplicates(subset=['published_at'], keep='first')
        
        # Drop articles with missing publish times
        news_df = news_df.dropna(subset=['published_at'])
        
        # Sort by published_at ascending
        news_df = news_df.sort_values('published_at').reset_index(drop=True)
        
        # Ensure published_at is datetime
        news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
        
        # Ensure sentiment is numeric
        news_df['sentiment'] = pd.to_numeric(news_df['sentiment'], errors='coerce').fillna(0.0)
        
        return news_df
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error connecting to Marketaux API for {ticker}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error fetching company news for {ticker}: {str(e)}")


def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    """
    Retrieve quarterly financial reports for the last 2 years.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        DataFrame with columns: report_date, ticker, revenue, net_income, eps,
        operating_cash_flow, total_assets, total_liabilities
        All dates in UTC, sorted chronologically.
    """
    # Validate ticker
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get financials
        financials = stock.quarterly_financials
        balance_sheet = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        
        if financials is None or financials.empty:
            return pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                        'operating_cash_flow', 'total_assets', 'total_liabilities'])
        
        # Calculate cutoff date (2 years ago)
        cutoff_date = datetime.now(pytz.UTC) - timedelta(days=730)
        
        # Extract data
        financials_list = []
        
        # Get all available quarters
        quarters = financials.columns
        
        # Check if columns exist
        if len(quarters) == 0:
            return pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                        'operating_cash_flow', 'total_assets', 'total_liabilities'])
        
        for quarter in quarters:
            # Convert quarter to datetime
            if isinstance(quarter, pd.Timestamp):
                report_date = quarter
            else:
                report_date = pd.to_datetime(quarter)
            
            # Ensure timezone-aware UTC
            if report_date.tzinfo is None:
                report_date = pytz.UTC.localize(report_date.to_pydatetime())
            else:
                report_date = report_date.astimezone(pytz.UTC)
            
            # Filter to last 2 years
            if report_date < cutoff_date:
                continue
            
            # Extract values (handle variations in yfinance index names)
            revenue = None
            for rev_key in ['Total Revenue', 'Revenue', 'Revenues']:
                if rev_key in financials.index:
                    revenue = financials.loc[rev_key, quarter]
                    break
            
            net_income = None
            for ni_key in ['Net Income', 'Net Income Common Stockholders']:
                if ni_key in financials.index:
                    net_income = financials.loc[ni_key, quarter]
                    break
            
            eps = None
            for eps_key in ['Basic EPS', 'Earnings Per Share', 'EPS']:
                if eps_key in financials.index:
                    eps = financials.loc[eps_key, quarter]
                    break
            
            # From balance sheet
            total_assets = None
            total_liabilities = None
            if balance_sheet is not None and not balance_sheet.empty:
                for ta_key in ['Total Assets', 'Assets']:
                    if ta_key in balance_sheet.index:
                        total_assets = balance_sheet.loc[ta_key, quarter]
                        break
                
                for tl_key in ['Total Liabilities', 'Liabilities']:
                    if tl_key in balance_sheet.index:
                        total_liabilities = balance_sheet.loc[tl_key, quarter]
                        break
            
            # From cash flow
            operating_cash_flow = None
            if cashflow is not None and not cashflow.empty:
                for ocf_key in ['Operating Cash Flow', 'Total Cash From Operating Activities', 
                               'Cash From Operating Activities', 'Operating Activities']:
                    if ocf_key in cashflow.index:
                        operating_cash_flow = cashflow.loc[ocf_key, quarter]
                        break
            
            financials_list.append({
                'report_date': report_date,
                'ticker': ticker,
                'revenue': revenue if pd.notna(revenue) else None,
                'net_income': net_income if pd.notna(net_income) else None,
                'eps': eps if pd.notna(eps) else None,
                'operating_cash_flow': operating_cash_flow if pd.notna(operating_cash_flow) else None,
                'total_assets': total_assets if pd.notna(total_assets) else None,
                'total_liabilities': total_liabilities if pd.notna(total_liabilities) else None
            })
        
        if not financials_list:
            return pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                        'operating_cash_flow', 'total_assets', 'total_liabilities'])
        
        financials_df = pd.DataFrame(financials_list)
        
        # Sort by report_date ascending
        financials_df = financials_df.sort_values('report_date').reset_index(drop=True)
        
        # Ensure report_date is datetime
        financials_df['report_date'] = pd.to_datetime(financials_df['report_date'], utc=True)
        
        # Convert numeric columns
        numeric_cols = ['revenue', 'net_income', 'eps', 'operating_cash_flow', 'total_assets', 'total_liabilities']
        for col in numeric_cols:
            financials_df[col] = pd.to_numeric(financials_df[col], errors='coerce')
        
        return financials_df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching quarterly financials for {ticker}: {str(e)}")


def ingest_all_data(ticker: str, marketaux_api_key: str) -> Dict:
    """
    Main function to ingest all data for a given ticker.
    
    This function orchestrates all data fetching functions and returns
    clean DataFrames ready for downstream processing.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        marketaux_api_key: Marketaux API key for fetching news sentiment
    
    Returns:
        Dictionary with keys:
            - prices_df: Historical OHLCV data (2 years)
            - latest_price_df: Current price snapshot (DataFrame)
            - news_df: Company news with sentiment scores (DataFrame)
            - financials_df: Quarterly financial reports (DataFrame)
    """
    # Validate inputs
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("ticker must be a non-empty string")
    
    if not marketaux_api_key or not isinstance(marketaux_api_key, str) or len(marketaux_api_key.strip()) == 0:
        raise ValueError("marketaux_api_key must be a non-empty string")
    
    # Calculate date range (2 years from now)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=730)  # Approximately 2 years
    
    # Fetch all data with error handling for each function
    # This allows partial results if one function fails
    prices_df = pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    latest_price_df = pd.DataFrame(columns=['timestamp', 'ticker', 'price'])
    news_df = pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
    financials_df = pd.DataFrame(columns=['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                                          'operating_cash_flow', 'total_assets', 'total_liabilities'])
    
    try:
        prices_df = fetch_historical_prices(ticker, start_date, end_date, interval="1d")
    except Exception as e:
        # Log error but continue with other data sources
        pass
    
    try:
        latest_price_df = fetch_latest_price(ticker)
    except Exception as e:
        # Log error but continue with other data sources
        pass
    
    try:
        news_df = fetch_company_news(ticker, start_date, end_date, marketaux_api_key)
    except Exception as e:
        # Log error but continue with other data sources
        pass
    
    try:
        financials_df = fetch_quarterly_financials(ticker)
    except Exception as e:
        # Log error but continue with other data sources
        pass
    
    return {
        'prices_df': prices_df,
        'latest_price_df': latest_price_df,
        'news_df': news_df,
        'financials_df': financials_df
    }

