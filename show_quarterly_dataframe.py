"""
Display MSFT Quarterly DataFrame with Stock Impact Score

This script shows the quarterly DataFrame with the new stock_impact_score column.
"""

from extract_all_quarterly_data import extract_all_quarterly_data
import pandas as pd

def main():
    print("="*70)
    print("MSFT Quarterly DataFrame with Stock Impact Score")
    print("="*70)
    
    # Extract quarterly data with scoring
    quarterly_df = extract_all_quarterly_data()
    
    if quarterly_df.empty:
        print("\n❌ No quarterly data found!")
        return
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print("\n" + "="*70)
    print("FULL QUARTERLY DATAFRAME (All Columns)")
    print("="*70)
    print(quarterly_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("QUARTERLY DATAFRAME (Key Columns for Analysis)")
    print("="*70)
    key_cols = ['report_date', 'quarter', 'fiscal_year', 'revenue', 'net_income', 
                'eps', 'revenue_growth_yoy', 'net_income_growth_yoy', 
                'financial_health_score', 'stock_impact_score', 'stock_impact', 
                'expected_price_change', 'confidence']
    print(quarterly_df[key_cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("QUARTERLY DATAFRAME (Base Columns for Model Training)")
    print("="*70)
    base_cols = ['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                 'operating_cash_flow', 'total_assets', 'total_liabilities']
    print(quarterly_df[base_cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("STOCK IMPACT SCORE EXPLANATION")
    print("="*70)
    print("stock_impact_score: Numeric score from -100 to +100")
    print("  - +100 to +70: STRONGLY_POSITIVE (expect +5% to +10% price increase)")
    print("  - +69 to +40: POSITIVE (expect +2% to +5% price increase)")
    print("  - +39 to +10: SLIGHTLY_POSITIVE (expect 0% to +2% price increase)")
    print("  - +9 to -9: NEUTRAL (expect -2% to +2% price change)")
    print("  - -10 to -39: NEGATIVE (expect -5% to -2% price decrease)")
    print("  - -40 to -100: STRONGLY_NEGATIVE (expect -10% to -5% price decrease)")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Average Financial Health Score: {quarterly_df['financial_health_score'].mean():.1f}/100")
    print(f"Average Stock Impact Score: {quarterly_df['stock_impact_score'].mean():.1f}/100")
    print(f"Latest Quarter Stock Impact Score: {quarterly_df['stock_impact_score'].iloc[-1]:.1f}/100")
    print(f"Trend: {'IMPROVING' if len(quarterly_df) > 1 and quarterly_df['stock_impact_score'].iloc[-1] > quarterly_df['stock_impact_score'].iloc[0] else 'STABLE' if len(quarterly_df) == 1 else 'DECLINING'}")

if __name__ == "__main__":
    main()

