"""
Display the Final Merged DataFrame

Shows the complete merged DataFrame with all data sources before model training.
"""

from merge_all_data import merge_all_data_sources
import pandas as pd

def main():
    df = merge_all_data_sources('MSFT')
    
    if df.empty:
        print("❌ No data to display!")
        return
    
    print('='*120)
    print('FINAL MERGED DATAFRAME - READY FOR MODEL TRAINING')
    print('='*120)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Show key columns
    key_cols = ['timestamp', 'close', 'volume', 'sentiment_mean', 'sentiment_count', 
                'revenue', 'net_income', 'eps', 'stock_impact_score', 'target']
    
    print('\n📊 COMPLETE DATAFRAME (All rows with key columns):')
    print(df[key_cols].to_string(index=False))
    
    print('\n' + '='*120)
    print('SUMMARY')
    print('='*120)
    print(f'Total Samples: {len(df)}')
    print(f'Total Features: {len(df.columns)}')
    print(f'Date Range: {df["timestamp"].min().strftime("%B %d, %Y")} to {df["timestamp"].max().strftime("%B %d, %Y")}')
    
    print(f'\n📈 Quarterly Data Coverage:')
    print(f'  Revenue: ${df["revenue"].min():,.0f}M to ${df["revenue"].max():,.0f}M')
    print(f'  Net Income: ${df["net_income"].min():,.0f}M to ${df["net_income"].max():,.0f}M')
    print(f'  EPS: ${df["eps"].min():.2f} to ${df["eps"].max():.2f}')
    print(f'  Stock Impact Score: {df["stock_impact_score"].min():.1f} to {df["stock_impact_score"].max():.1f}')
    
    print(f'\n📰 News Data Coverage:')
    print(f'  Articles with sentiment: {df["sentiment_count"].sum():.0f} total')
    if df[df["sentiment_count"] > 0]["sentiment_mean"].count() > 0:
        print(f'  Average sentiment when present: {df[df["sentiment_count"] > 0]["sentiment_mean"].mean():.3f}')
    
    print(f'\n💰 Price Data Coverage:')
    print(f'  Price range: ${df["close"].min():.2f} to ${df["close"].max():.2f}')
    print(f'  Average volume: {df["volume"].mean():,.0f}')
    
    print(f'\n✅ DataFrame is ready to pass to model for training!')
    print('='*120)

if __name__ == "__main__":
    main()

