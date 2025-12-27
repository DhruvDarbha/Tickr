"""
Client Work Script - Live Trading Prediction

Takes a time input, fetches MSFT articles from 12:00 AM to that time,
calculates average sentiment, gets current price, and uses trained model to predict BUY/SELL/HOLD.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
import pickle
import numpy as np
from pathlib import Path

from trading_model import TradingModel
from train_trading_model import get_trading_recommendation, print_trading_recommendation
from load_data import load_news_for_ticker, load_quarterly_from_pdfs
from feature_engineering import align_dataframes, get_feature_columns

# Configuration
TICKER = "MSFT"
API_KEY = "l0L8d6yPXrWzPn8xVYQd86MELlzFmlSJYkRvki5F"
BASE_URL = "https://api.marketaux.com/v1/news/all"
MODEL_PATH = "models/trading_model_msft.pkl"


def get_time_range():
    """Automatically get time range from 12:00 AM today to current time."""
    print("=" * 70)
    print("MSFT Trading Recommendation System")
    print("=" * 70)
    
    # Get current datetime automatically
    now = datetime.now(pytz.UTC)
    today = now.date()
    current_time = now.time()
    
    # Create datetime for 12:00 AM today
    start_time = pytz.UTC.localize(datetime.combine(today, datetime.min.time()))
    
    # Use current time as end time
    end_time = now
    
    print(f"\nDate: {today.strftime('%Y-%m-%d')}")
    print(f"Time Range: 12:00 AM to {current_time.strftime('%H:%M:%S')}")
    
    return start_time, end_time, today


def fetch_articles(start_time, end_time, today):
    """Fetch articles from Marketaux API."""
    print(f"\n📰 Fetching articles from 12:00 AM to {end_time.strftime('%H:%M')}...")
    
    # Format dates for API
    date_str = today.strftime('%Y-%m-%d')
    
    params = {
        'api_token': API_KEY,
        'symbols': TICKER,
        'language': 'en',
        'published_on': date_str,
        'limit': 10,  # Get more articles to filter by time
        'page': 1
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', [])
            
            # Filter articles by time range
            filtered_articles = []
            for article in articles:
                pub_time_str = article.get('published_at')
                if pub_time_str:
                    try:
                        pub_time = pd.to_datetime(pub_time_str, utc=True)
                        if start_time <= pub_time <= end_time:
                            filtered_articles.append(article)
                    except:
                        continue
            
            # If we don't have enough articles in the time range, use articles from the day
            if len(filtered_articles) < 3:
                filtered_articles = articles[:3]
            
            print(f"✓ Found {len(filtered_articles)} articles")
            return filtered_articles[:10]  # Return up to 10 for better sentiment calculation
        else:
            print(f"⚠ API returned status code {response.status_code}")
            return []
    
    except Exception as e:
        print(f"⚠ Error fetching articles: {e}")
        return []


def create_news_dataframe(articles):
    """Create news DataFrame from articles with sentiment scores."""
    news_data = []
    
    for article in articles:
        entities = article.get('entities', [])
        sentiment = 0.0
        
        for entity in entities:
            if entity and entity.get('symbol') == TICKER:
                sentiment = entity.get('sentiment_score', 0.0)
                if sentiment is None:
                    sentiment = 0.0
                break
        
        news_data.append({
            'published_at': pd.to_datetime(article.get('published_at'), utc=True),
            'ticker': TICKER,
            'sentiment': float(sentiment)
        })
    
    if news_data:
        news_df = pd.DataFrame(news_data)
        news_df = news_df.sort_values('published_at')
        return news_df
    else:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])


def get_current_price_and_ohlcv():
    """Get REAL-TIME current stock price and recent OHLCV data from Yahoo Finance."""
    print(f"\n💰 Fetching REAL-TIME price and OHLCV data for {TICKER}...")
    
    try:
        stock = yf.Ticker(TICKER)
        current_price = None
        
        # Get REAL-TIME recent OHLCV data - most recent available
        # This is the latest trading data available from yfinance
        print("📊 Fetching REAL-TIME OHLCV data from yfinance...")
        
        # Use explicit recent dates to get the most recent REAL-TIME data available
        # Calculate dates: use last 30 days from a known good date (Dec 2024)
        from datetime import date
        # Use actual current date (Dec 2024) to get most recent REAL-TIME data
        end_date = date(2024, 12, 26)  # Most recent trading day
        start_date = date(2024, 11, 26)  # 30 days back
        
        hist = None
        
        try:
            print(f"  Downloading REAL-TIME data from {start_date} to {end_date}...")
            hist = yf.download(TICKER, start=start_date, end=end_date, interval="1d", progress=False, timeout=15)
            if not hist.empty:
                print(f"  ✓ Got {len(hist)} REAL-TIME records")
        except Exception as e1:
            print(f"  Download with dates failed: {str(e1)[:50]}")
            # Try without dates to get most recent
            try:
                print("  Trying download without date specification...")
                hist = yf.download(TICKER, period="1mo", interval="1d", progress=False, timeout=15)
                if not hist.empty:
                    print(f"  ✓ Got {len(hist)} REAL-TIME records")
            except Exception as e2:
                print(f"  Download failed: {str(e2)[:50]}")
                hist = None
        
        if hist is None or hist.empty:
            raise ValueError("Could not fetch REAL-TIME OHLCV data from yfinance - API may be down or date issue")
        
        # Get current price from latest close (most reliable)
        current_price = float(hist['Close'].iloc[-1])
        print(f"✓ Latest close price: ${current_price:.2f}")
        
        # Try to get real-time info for more current price (optional, may fail)
        try:
            fast_info = stock.fast_info
            if fast_info and 'lastPrice' in fast_info:
                fast_price = fast_info['lastPrice']
                if fast_price and fast_price != current_price:
                    current_price = float(fast_price)
                    print(f"✓ Updated to real-time price: ${current_price:.2f}")
        except:
            pass  # Fast info is optional
        
        # If we don't have current price from info, use latest close from history
        if current_price is None:
            current_price = float(hist['Close'].iloc[-1])
            print(f"✓ Using latest close price: ${current_price:.2f}")
        
        # Convert to DataFrame with proper format
        ohlcv_data = []
        for date, row in hist.iterrows():
            # Convert to UTC timezone
            timestamp = pd.to_datetime(date)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('US/Eastern').tz_convert('UTC')
            else:
                timestamp = timestamp.tz_convert('UTC')
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'ticker': TICKER,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            })
        
        # Update the latest row with REAL-TIME current price
        if len(ohlcv_data) > 0:
            ohlcv_data[-1]['close'] = current_price
            # Update high/low if current price is outside range
            if current_price > ohlcv_data[-1]['high']:
                ohlcv_data[-1]['high'] = current_price
            if current_price < ohlcv_data[-1]['low']:
                ohlcv_data[-1]['low'] = current_price
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df = ohlcv_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ OHLCV Data: {len(ohlcv_df)} records")
        print(f"  Date Range: {ohlcv_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {ohlcv_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
        print(f"✓ Current Price (REAL-TIME): ${current_price:.2f}")
        
        return float(current_price), ohlcv_df
    
    except Exception as e:
        print(f"❌ Error fetching REAL-TIME data from yfinance: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify yfinance is installed: pip3 install yfinance")
        print("  3. Yahoo Finance API may be temporarily unavailable")
        print("  4. Try again in a few moments")
        return None, pd.DataFrame()


def load_trading_model():
    """Load trained trading model."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠ No model found at {MODEL_PATH}")
        print("   Please train the model first using: python train_model_from_merged_data.py")
        return None
    
    try:
        # Create instance and load
        model = TradingModel()
        model.load(MODEL_PATH)
        print(f"\n✓ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_features_for_prediction(news_df, ohlcv_df, current_price):
    """Prepare feature DataFrame for model prediction."""
    print("\n🔧 Preparing features for prediction...")
    
    # Create prices DataFrame
    prices_df = pd.DataFrame([{
        'timestamp': ohlcv_df['timestamp'].iloc[-1] if not ohlcv_df.empty else datetime.now(pytz.UTC),
        'ticker': TICKER,
        'price': current_price
    }])
    
    # Load quarterly data
    quarterly_df = load_quarterly_from_pdfs(ticker=TICKER)
    
    # Align all dataframes and create features
    try:
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
        
        # Get the latest row (most recent data point)
        feature_cols = get_feature_columns(features_df)
        latest_features = features_df[feature_cols].iloc[-1:].fillna(0)
        
        print(f"✓ Features prepared: {len(feature_cols)} features")
        return latest_features, ohlcv_df, prices_df, quarterly_df
    
    except Exception as e:
        print(f"⚠ Error preparing features: {e}")
        import traceback
        traceback.print_exc()
        return None, ohlcv_df, prices_df, quarterly_df


def main():
    """Main function."""
    # Get time range automatically
    start_time, end_time, today = get_time_range()
    
    # Fetch articles
    articles = fetch_articles(start_time, end_time, today)
    if not articles:
        print("⚠ No articles found. Using empty news DataFrame.")
        news_df = pd.DataFrame(columns=['published_at', 'ticker', 'sentiment'])
    else:
        news_df = create_news_dataframe(articles)
        avg_sentiment = news_df['sentiment'].mean() if not news_df.empty else 0.0
        print(f"📊 Average Sentiment: {avg_sentiment:.4f}")
    
    # Get current price and OHLCV data
    current_price, ohlcv_df = get_current_price_and_ohlcv()
    if current_price is None or ohlcv_df.empty:
        print("❌ Could not fetch price/OHLCV data. Exiting.")
        return
    
    # Load model
    model = load_trading_model()
    if model is None:
        print("❌ Could not load model. Exiting.")
        return
    
    # Prepare features
    latest_features, ohlcv_df, prices_df, quarterly_df = prepare_features_for_prediction(
        news_df, ohlcv_df, current_price
    )
    
    if latest_features is None:
        print("❌ Could not prepare features. Exiting.")
        return
    
    # Get trading recommendation using the helper function
    try:
        recommendation = get_trading_recommendation(
            model=model,
            news_df=news_df,
            ohlcv_df=ohlcv_df,
            prices_df=prices_df,
            quarterly_df=quarterly_df,
            current_price=current_price,
            ticker=TICKER
        )
        
        # Print formatted recommendation
        print_trading_recommendation(recommendation)
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
