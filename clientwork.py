"""
Client Work Script

Takes a time input, fetches 3 MSFT articles from 12:00 AM to that time,
calculates average sentiment, gets current price, and uses model to predict BUY/SELL.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
import pickle
import numpy as np

# Configuration
TICKER = "MSFT"
API_KEY = "6hV3UI51G9IwfgEZ1Yg7oQBQ6QP5zwCcMF5k3lnR"
BASE_URL = "https://api.marketaux.com/v1/news/all"


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
    """Fetch 3 articles from Marketaux API."""
    print(f"\nFetching articles from 12:00 AM to {end_time.strftime('%H:%M')}...")
    
    # Format dates for API
    date_str = today.strftime('%Y-%m-%d')
    
    params = {
        'api_token': API_KEY,
        'symbols': TICKER,
        'language': 'en',
        'published_on': date_str,
        'limit': 3,
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
            
            # If we don't have 3 articles in the time range, use the first 3 from the day
            if len(filtered_articles) < 3:
                filtered_articles = articles[:3]
            
            print(f"Found {len(filtered_articles)} articles")
            return filtered_articles[:3]
        else:
            print(f"Error: API returned status code {response.status_code}")
            return []
    
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []


def calculate_average_sentiment(articles):
    """Calculate average sentiment score from articles."""
    sentiment_scores = []
    
    for article in articles:
        entities = article.get('entities', [])
        for entity in entities:
            if entity and entity.get('symbol') == TICKER:
                sentiment = entity.get('sentiment_score')
                if sentiment is not None:
                    sentiment_scores.append(float(sentiment))
                break
    
    if sentiment_scores:
        avg_sentiment = np.mean(sentiment_scores)
        print(f"\nSentiment Scores: {sentiment_scores}")
        print(f"Average Sentiment: {avg_sentiment:.4f}")
        return avg_sentiment
    else:
        print("\nNo sentiment scores found, using 0.0")
        return 0.0


def get_current_price():
    """Get current stock price from Yahoo Finance."""
    print(f"\nFetching current price for {TICKER}...")
    
    try:
        stock = yf.Ticker(TICKER)
        info = stock.info
        
        if info and 'currentPrice' in info:
            price = info['currentPrice']
        elif info and 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
        else:
            # Fallback to history
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                raise ValueError("Could not fetch price")
        
        print(f"Current Price: ${price:.2f}")
        return float(price)
    
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None


def load_model(model_path="trading_model.pkl"):
    """Load trading model if it exists."""
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"\nModel loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"\nNo model found at {model_path}")
        print("Using simple rule-based prediction")
        return None


def predict_with_model(model, avg_sentiment, current_price):
    """Use model to predict BUY/SELL."""
    if model is None:
        # Simple rule-based prediction
        if avg_sentiment > 0.3:
            return "BUY", 0.8
        elif avg_sentiment < -0.3:
            return "SELL", 0.8
        else:
            return "HOLD", 0.5
    else:
        # Use actual model
        try:
            # Create feature DataFrame
            features = pd.DataFrame({
                'sentiment_score': [avg_sentiment],
                'current_price': [current_price]
            })
            
            # Ensure features match model's expected features
            if hasattr(model, 'feature_names_') and model.feature_names_:
                # Add missing features with default values
                for feature in model.feature_names_:
                    if feature not in features.columns:
                        features[feature] = 0.0
                
                # Reorder columns to match model
                features = features[model.feature_names_]
            
            # Predict
            predictions = model.predict(features)
            
            if isinstance(predictions, pd.DataFrame):
                signal = predictions['signal'].iloc[0]
                confidence = predictions.get('signal_confidence', pd.Series([0.5])).iloc[0]
            else:
                # Handle different model types
                signal = predictions[0] if isinstance(predictions, (list, np.ndarray)) else predictions
                confidence = 0.7
            
            return signal, float(confidence)
        
        except Exception as e:
            print(f"Error using model: {e}")
            # Fallback to simple rules
            if avg_sentiment > 0.3:
                return "BUY", 0.8
            elif avg_sentiment < -0.3:
                return "SELL", 0.8
            else:
                return "HOLD", 0.5


def main():
    """Main function."""
    # Get time range automatically
    start_time, end_time, today = get_time_range()
    
    # Fetch articles
    articles = fetch_articles(start_time, end_time, today)
    if not articles:
        print("No articles found. Exiting.")
        return
    
    # Calculate average sentiment
    avg_sentiment = calculate_average_sentiment(articles)
    
    # Get current price
    current_price = get_current_price()
    if current_price is None:
        print("Could not fetch price. Exiting.")
        return
    
    # Load model
    model = load_model()
    
    # Make prediction
    signal, confidence = predict_with_model(model, avg_sentiment, current_price)
    
    # Print results
    print("\n" + "=" * 70)
    print("TRADING RECOMMENDATION")
    print("=" * 70)
    print(f"Ticker: {TICKER}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Average Sentiment: {avg_sentiment:.4f}")
    print(f"Signal: {signal}")
    print(f"Confidence: {confidence:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

