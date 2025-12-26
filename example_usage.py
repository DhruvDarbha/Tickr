"""
Example usage of the Phase 1 data ingestion pipeline.

This script demonstrates how to use the data ingestion functions
to retrieve all required data for a stock ticker.
"""

from data_ingestion import ingest_all_data

def main():
    # Get Marketaux API key from user
    print("=" * 60)
    print("Phase 1: Data Ingestion Pipeline")
    print("=" * 60)
    marketaux_api_key = "xCJ5QFoH9ZagupfWkj7RI2LNpO8QTyZomT1PSEAm"
    # Example: Fetch all data for Apple Inc.
    ticker = "AAPL"
    
    print(f"\nFetching data for {ticker}...")
    print("=" * 60)
    
    # Use the main orchestration function
    data = ingest_all_data(ticker, marketaux_api_key)
    
    # Display results
    print("\n1. HISTORICAL PRICES (OHLCV - 2 years)")
    print("-" * 60)
    prices_df = data['prices_df']
    print(f"Shape: {prices_df.shape}")
    if not prices_df.empty:
        print(f"Date range: {prices_df['timestamp'].min()} to {prices_df['timestamp'].max()}")
        print("\nFirst 5 rows:")
        print(prices_df.head())
        print("\nLast 5 rows:")
        print(prices_df.tail())
    else:
        print("No price data available.")
    
    print("\n\n2. LATEST PRICE")
    print("-" * 60)
    latest_price_df = data['latest_price_df']
    if not latest_price_df.empty:
        row = latest_price_df.iloc[0]
        print(f"Ticker: {row['ticker']}")
        print(f"Price: ${row['price']:.2f}")
        print(f"Timestamp: {row['timestamp']}")
    else:
        print("No latest price available.")
    
    print("\n\n3. COMPANY NEWS (with sentiment scores)")
    print("-" * 60)
    news_df = data['news_df']
    print(f"Total articles: {len(news_df)}")
    if not news_df.empty:
        print("\nFirst 10 articles:")
        print(news_df.head(10))
        print(f"\nSentiment statistics:")
        print(f"  Min: {news_df['sentiment'].min():.3f}")
        print(f"  Max: {news_df['sentiment'].max():.3f}")
        print(f"  Mean: {news_df['sentiment'].mean():.3f}")
        print(f"  Std: {news_df['sentiment'].std():.3f}")
    else:
        print("No news articles found.")
    
    print("\n\n4. QUARTERLY FINANCIALS")
    print("-" * 60)
    financials_df = data['financials_df']
    print(f"Total quarters: {len(financials_df)}")
    if not financials_df.empty:
        print("\nAll quarters:")
        print(financials_df.to_string())
    else:
        print("No financial data available.")
    
    print("\n" + "=" * 60)
    print("Data ingestion complete!")
    print("=" * 60)
    print(data)


if __name__ == "__main__":
    main()

