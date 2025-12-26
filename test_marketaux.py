"""
Test script to verify Marketaux API is returning JSON data for a single date.
"""

import requests
import json
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Set pandas display options to show full DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

def test_marketaux_single_date():
    """Test Marketaux API with a single date to verify JSON response."""
    
    # API key
    api_key = "6hV3UI51G9IwfgEZ1Yg7oQBQ6QP5zwCcMF5k3lnR"
    
    # Test parameters
    ticker = "AAPL"
    # Use a recent date (yesterday to ensure there's likely news)
    test_date = (datetime.now(pytz.UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    base_url = "https://api.marketaux.com/v1/news/all"
    
    print("=" * 70)
    print("Marketaux API Test - Single Date")
    print("=" * 70)
    print(f"\nTicker: {ticker}")
    print(f"Date: {test_date}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    print("\n" + "-" * 70)
    
    # Test 1: Using published_on parameter (single date)
    print("\n[TEST 1] Using 'published_on' parameter:")
    print("-" * 70)
    
    params1 = {
        'api_token': api_key,
        'symbols': ticker,
        'published_on': test_date,
        'limit': 10,
        'page': 1
    }
    
    print(f"Request URL: {base_url}")
    print(f"Parameters:")
    for key, value in params1.items():
        if key == 'api_token':
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")
    
    try:
        response1 = requests.get(base_url, params=params1, timeout=30)
        print(f"\nResponse Status Code: {response1.status_code}")
        
        if response1.status_code == 200:
            try:
                data1 = response1.json()
                print("\n✅ JSON Response Received!")
                print("\nFull JSON Response:")
                print(json.dumps(data1, indent=2))
                
                print("\n" + "-" * 70)
                print("Response Summary:")
                print("-" * 70)
                if 'meta' in data1:
                    meta = data1['meta']
                    print(f"  Found: {meta.get('found', 'N/A')}")
                    print(f"  Returned: {meta.get('returned', 'N/A')}")
                    print(f"  Limit: {meta.get('limit', 'N/A')}")
                    print(f"  Page: {meta.get('page', 'N/A')}")
                
                if 'data' in data1:
                    articles = data1['data']
                    print(f"\n  Articles in response: {len(articles)}")
                    
                    if len(articles) > 0:
                        print(f"\n  First Article Details:")
                        first = articles[0]
                        print(f"    UUID: {first.get('uuid', 'N/A')}")
                        print(f"    Title: {first.get('title', 'N/A')[:80]}...")
                        print(f"    Published: {first.get('published_at', 'N/A')}")
                        print(f"    Source: {first.get('source', 'N/A')}")
                        print(f"    Entities: {len(first.get('entities', []))}")
                        
                        if 'entities' in first and len(first['entities']) > 0:
                            print(f"\n    Entity Details:")
                            for entity in first['entities']:
                                print(f"      Symbol: {entity.get('symbol', 'N/A')}")
                                print(f"      Name: {entity.get('name', 'N/A')}")
                                print(f"      Sentiment Score: {entity.get('sentiment_score', 'N/A')}")
                                print(f"      Match Score: {entity.get('match_score', 'N/A')}")
                        
                        # Create DataFrame with timestamps
                        print(f"\n  {'='*70}")
                        print(f"  Creating DataFrame with timestamps...")
                        print(f"  {'='*70}")
                        
                        df_data = []
                        for article in articles:
                            # Extract published_at timestamp
                            pub_time_str = article.get('published_at')
                            if pub_time_str:
                                # Parse timestamp
                                try:
                                    pub_time = pd.to_datetime(pub_time_str, utc=True)
                                    if pub_time.tzinfo is None:
                                        pub_time = pytz.UTC.localize(pub_time.to_pydatetime())
                                    else:
                                        pub_time = pub_time.astimezone(pytz.UTC)
                                except:
                                    pub_time = None
                            else:
                                pub_time = None
                            
                            # Extract sentiment for the ticker
                            sentiment_score = None
                            entities = article.get('entities', [])
                            for entity in entities:
                                if entity and entity.get('symbol') == ticker:
                                    sentiment_score = entity.get('sentiment_score')
                                    break
                            
                            if sentiment_score is None:
                                sentiment_score = 0.0
                            else:
                                sentiment_score = float(sentiment_score)
                            
                            df_data.append({
                                'published_at': pub_time,
                                'ticker': ticker,
                                'sentiment': sentiment_score,
                                'title': article.get('title', ''),
                                'source': article.get('source', ''),
                                'url': article.get('url', '')
                            })
                        
                        # Create DataFrame
                        df = pd.DataFrame(df_data)
                        
                        # Ensure published_at is datetime
                        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
                        
                        # Sort by timestamp
                        df = df.sort_values('published_at').reset_index(drop=True)
                        
                        print(f"\n  DataFrame created:")
                        print(f"    Shape: {df.shape}")
                        print(f"    Columns: {list(df.columns)}")
                        print(f"\n  Full DataFrame:")
                        print(df.to_string())
                        
                        # Create a copy with just timestamp, ticker, and sentiment
                        df_copy = df[['published_at', 'ticker', 'sentiment']].copy()
                        print(f"\n  Copied DataFrame (timestamp, ticker, sentiment only):")
                        print(f"    Shape: {df_copy.shape}")
                        print(f"\n  Full Copied DataFrame:")
                        print(df_copy.to_string())
                        
                        # Return the DataFrame
                        return df_copy
                        
                        # Simulate what it would look like with more data (based on previous run)
                        print(f"\n  {'='*70}")
                        print(f"  Simulated DataFrame (based on previous API response):")
                        print(f"  {'='*70}")
                        
                        # Create simulated data based on what we saw in the previous run
                        simulated_data = [
                            {
                                'published_at': pd.to_datetime('2025-12-25T22:07:35.000000Z', utc=True),
                                'ticker': 'AAPL',
                                'sentiment': 0.5682
                            },
                            {
                                'published_at': pd.to_datetime('2025-12-25T21:07:30.000000Z', utc=True),
                                'ticker': 'AAPL',
                                'sentiment': 0.93
                            },
                            {
                                'published_at': pd.to_datetime('2025-12-25T19:07:38.000000Z', utc=True),
                                'ticker': 'AAPL',
                                'sentiment': 0.76595
                            },
                            {
                                'published_at': pd.to_datetime('2025-12-26T08:33:39.000000Z', utc=True),
                                'ticker': 'AAPL',
                                'sentiment': 0.212
                            },
                            {
                                'published_at': pd.to_datetime('2025-12-26T06:23:02.000000Z', utc=True),
                                'ticker': 'AAPL',
                                'sentiment': 0.44865
                            }
                        ]
                        
                        simulated_df = pd.DataFrame(simulated_data)
                        simulated_df['published_at'] = pd.to_datetime(simulated_df['published_at'], utc=True)
                        simulated_df = simulated_df.sort_values('published_at').reset_index(drop=True)
                        
                        print(f"\n  Simulated DataFrame:")
                        print(f"    Shape: {simulated_df.shape}")
                        print(f"    Date Range: {simulated_df['published_at'].min()} to {simulated_df['published_at'].max()}")
                        print(f"    Sentiment Range: {simulated_df['sentiment'].min():.3f} to {simulated_df['sentiment'].max():.3f}")
                        print(f"\n  Full Simulated DataFrame:")
                        print(simulated_df.to_string())
                        
                        print(f"\n  DataFrame Info:")
                        print(f"    Dtypes:")
                        print(f"      published_at: {simulated_df['published_at'].dtype}")
                        print(f"      ticker: {simulated_df['ticker'].dtype}")
                        print(f"      sentiment: {simulated_df['sentiment'].dtype}")
                        print(f"    Memory usage: {simulated_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                        
                    else:
                        print("\n  ⚠️  No articles in response!")
                else:
                    print("\n  ⚠️  No 'data' key in response!")
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON: {e}")
                print(f"Response text: {response1.text[:500]}")
        else:
            print(f"\n❌ Non-200 status code: {response1.status_code}")
            print(f"Response text: {response1.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    
    # Test 2: Using published_after/published_before (date range for single day)
    print("\n\n" + "=" * 70)
    print("[TEST 2] Using 'published_after' and 'published_before' parameters:")
    print("-" * 70)
    
    # Set time range for the day (00:00:00 to 23:59:59 UTC)
    test_datetime = datetime.strptime(test_date, '%Y-%m-%d')
    day_start = pytz.UTC.localize(datetime.combine(test_datetime.date(), datetime.min.time()))
    day_end = pytz.UTC.localize(datetime.combine(test_datetime.date(), datetime.max.time()))
    
    params2 = {
        'api_token': api_key,
        'symbols': ticker,
        'published_after': day_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'published_before': day_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'limit': 10,
        'page': 1
    }
    
    print(f"Request URL: {base_url}")
    print(f"Parameters:")
    for key, value in params2.items():
        if key == 'api_token':
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")
    
    try:
        response2 = requests.get(base_url, params=params2, timeout=30)
        print(f"\nResponse Status Code: {response2.status_code}")
        
        if response2.status_code == 200:
            try:
                data2 = response2.json()
                print("\n✅ JSON Response Received!")
                print("\nFull JSON Response:")
                print(json.dumps(data2, indent=2))
                
                print("\n" + "-" * 70)
                print("Response Summary:")
                print("-" * 70)
                if 'meta' in data2:
                    meta = data2['meta']
                    print(f"  Found: {meta.get('found', 'N/A')}")
                    print(f"  Returned: {meta.get('returned', 'N/A')}")
                
                if 'data' in data2:
                    articles = data2['data']
                    print(f"\n  Articles in response: {len(articles)}")
                else:
                    print("\n  ⚠️  No 'data' key in response!")
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON: {e}")
                print(f"Response text: {response2.text[:500]}")
        else:
            print(f"\n❌ Non-200 status code: {response2.status_code}")
            print(f"Response text: {response2.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    
    # Test 3: Without date filter (just ticker)
    print("\n\n" + "=" * 70)
    print("[TEST 3] Using only 'symbols' parameter (no date filter):")
    print("-" * 70)
    
    params3 = {
        'api_token': api_key,
        'symbols': ticker,
        'limit': 5,
        'page': 1
    }
    
    print(f"Request URL: {base_url}")
    print(f"Parameters:")
    for key, value in params3.items():
        if key == 'api_token':
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")
    
    try:
        response3 = requests.get(base_url, params=params3, timeout=30)
        print(f"\nResponse Status Code: {response3.status_code}")
        
        if response3.status_code == 200:
            try:
                data3 = response3.json()
                print("\n✅ JSON Response Received!")
                print("\nFull JSON Response:")
                print(json.dumps(data3, indent=2))
                
                print("\n" + "-" * 70)
                print("Response Summary:")
                print("-" * 70)
                if 'meta' in data3:
                    meta = data3['meta']
                    print(f"  Found: {meta.get('found', 'N/A')}")
                    print(f"  Returned: {meta.get('returned', 'N/A')}")
                
                if 'data' in data3:
                    articles = data3['data']
                    print(f"\n  Articles in response: {len(articles)}")
                    if len(articles) > 0:
                        print(f"\n  Most recent article published: {articles[0].get('published_at', 'N/A')}")
                else:
                    print("\n  ⚠️  No 'data' key in response!")
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON: {e}")
                print(f"Response text: {response3.text[:500]}")
        else:
            print(f"\n❌ Non-200 status code: {response3.status_code}")
            print(f"Response text: {response3.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return None


if __name__ == "__main__":
    result_df = test_marketaux_single_date()
    if result_df is not None:
        print("\n" + "=" * 70)
        print("RETURNED DATAFRAME:")
        print("=" * 70)
        print(result_df.to_string())
        print(f"\nDataFrame shape: {result_df.shape}")
        print(f"DataFrame columns: {list(result_df.columns)}")
        print(f"DataFrame dtypes:\n{result_df.dtypes}")

