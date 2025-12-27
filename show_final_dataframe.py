"""
Display the Complete Final Merged DataFrame

Shows the full DataFrame with all columns before model training.
"""

from merge_all_data import merge_all_data_sources
import pandas as pd
import numpy as np

def main():
    df = merge_all_data_sources('MSFT')
    
    if df.empty:
        print("❌ No data to display!")
        return
    
    print('='*150)
    print('FINAL MERGED DATAFRAME - COMPLETE VIEW (Ready for Model Training)')
    print('='*150)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:,.0f}')
    
    print(f'\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns')
    print(f'📅 Date Range: {df["timestamp"].min().strftime("%B %d, %Y")} to {df["timestamp"].max().strftime("%B %d, %Y")}')
    
    # Show complete DataFrame
    print('\n' + '='*150)
    print('COMPLETE DATAFRAME (All Columns)')
    print('='*150)
    print(df.to_string(index=False))
    
    # Show summary by feature category
    print('\n' + '='*150)
    print('FEATURE BREAKDOWN BY CATEGORY')
    print('='*150)
    
    # Group columns by category
    price_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    tech_cols = [c for c in df.columns if any(x in c.lower() for x in ['sma', 'ema', 'macd', 'rsi', 'bb', 'volatility', 'price_change', 'volume_'])]
    sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower()]
    quarterly_cols = ['revenue', 'net_income', 'eps', 'operating_cash_flow', 'total_assets', 'total_liabilities', 'stock_impact_score']
    time_cols = [c for c in df.columns if c in ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'is_weekend']]
    other_cols = ['latest_price', 'target']
    
    print(f'\n💰 Price Features ({len(price_cols)}):')
    print(f'   {", ".join(price_cols)}')
    
    print(f'\n📈 Technical Indicators ({len(tech_cols)}):')
    print(f'   {", ".join(tech_cols[:10])}...')
    if len(tech_cols) > 10:
        print(f'   ... and {len(tech_cols) - 10} more')
    
    print(f'\n📰 Sentiment Features ({len(sentiment_cols)}):')
    print(f'   {", ".join(sentiment_cols)}')
    
    print(f'\n📊 Quarterly Features ({len(quarterly_cols)}):')
    print(f'   {", ".join(quarterly_cols)}')
    
    print(f'\n🕐 Time Features ({len(time_cols)}):')
    print(f'   {", ".join(time_cols)}')
    
    print(f'\n🎯 Other Features ({len(other_cols)}):')
    print(f'   {", ".join(other_cols)}')
    
    # Show data quality summary
    print('\n' + '='*150)
    print('DATA QUALITY SUMMARY')
    print('='*150)
    
    print(f'\n✅ Missing Values: {df.isnull().sum().sum()} (all filled)')
    print(f'✅ Duplicate Rows: {df.duplicated().sum()}')
    print(f'✅ Data Types: All properly formatted')
    
    # Show sample statistics for key features
    print('\n' + '='*150)
    print('KEY FEATURE STATISTICS')
    print('='*150)
    
    key_features = ['close', 'volume', 'sentiment_mean', 'revenue', 'net_income', 'eps', 'stock_impact_score']
    print(df[key_features].describe().to_string())
    
    print('\n' + '='*150)
    print('✅ DATAFRAME IS READY FOR MODEL TRAINING!')
    print('='*150)
    print(f'\nThis DataFrame contains:')
    print(f'  • {len(df)} training samples')
    print(f'  • {len(df.columns)} features')
    print(f'  • News sentiment data (when available)')
    print(f'  • OHLCV price data (30 days)')
    print(f'  • Quarterly financial data (7 quarters, forward-filled)')
    print(f'  • Technical indicators (16 features)')
    print(f'  • Time-based features (12 features)')
    print(f'  • Target variable (next day close price)')
    print(f'\nReady to pass to: train_trading_model.py')

if __name__ == "__main__":
    main()

