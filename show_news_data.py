"""
Display MSFT News Data from CSV

This script loads and displays the news dataframe from MSFT_all_dates_data.csv
"""

from load_data import load_news_for_ticker
import pandas as pd

def main():
    print("="*70)
    print("Loading MSFT News Data from CSV")
    print("="*70)
    
    # Load news data
    news_df = load_news_for_ticker('MSFT')
    
    if news_df.empty:
        print("\n❌ No news data found!")
        return
    
    print("\n" + "="*70)
    print("News DataFrame Summary")
    print("="*70)
    print(f"\nTotal Articles: {len(news_df)}")
    print(f"Date Range: {news_df['published_at'].min()} to {news_df['published_at'].max()}")
    print(f"Sentiment Range: {news_df['sentiment'].min():.3f} to {news_df['sentiment'].max():.3f}")
    print(f"Average Sentiment: {news_df['sentiment'].mean():.3f}")
    print(f"Median Sentiment: {news_df['sentiment'].median():.3f}")
    
    print("\n" + "="*70)
    print("DataFrame Info")
    print("="*70)
    print(f"\nColumns: {list(news_df.columns)}")
    print(f"Shape: {news_df.shape}")
    print(f"\nData Types:")
    print(news_df.dtypes)
    
    print("\n" + "="*70)
    print("First 10 Articles")
    print("="*70)
    print(news_df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("Last 10 Articles")
    print("="*70)
    print(news_df.tail(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("Sentiment Statistics")
    print("="*70)
    print(news_df['sentiment'].describe())
    
    print("\n" + "="*70)
    print("Articles by Sentiment Range")
    print("="*70)
    print("Very Negative (< -0.5):", len(news_df[news_df['sentiment'] < -0.5]))
    print("Negative (-0.5 to 0):", len(news_df[(news_df['sentiment'] >= -0.5) & (news_df['sentiment'] < 0)]))
    print("Neutral (0):", len(news_df[news_df['sentiment'] == 0]))
    print("Positive (0 to 0.5):", len(news_df[(news_df['sentiment'] > 0) & (news_df['sentiment'] <= 0.5)]))
    print("Very Positive (> 0.5):", len(news_df[news_df['sentiment'] > 0.5]))
    
    print("\n" + "="*70)
    print("Full DataFrame (All Articles)")
    print("="*70)
    # Set pandas display options to show full data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 100)
    print(news_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ News DataFrame Successfully Loaded!")
    print("="*70)
    print("\nThis dataframe is ready to be used in model training once")
    print("OHLCV, prices, and quarterly dataframes are available.")

if __name__ == "__main__":
    main()

