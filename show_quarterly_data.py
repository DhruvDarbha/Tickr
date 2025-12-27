"""
Display MSFT Quarterly Data with Scoring

This script loads and displays quarterly financial data from PDFs
with financial health scoring and stock impact assessment.
"""

from load_data import load_quarterly_from_pdfs
from extract_all_quarterly_data import extract_all_quarterly_data, print_quarterly_analysis
import pandas as pd

def main():
    print("="*70)
    print("MSFT Quarterly Financial Data with Scoring")
    print("="*70)
    
    # Load quarterly data with scoring
    quarterly_df = extract_all_quarterly_data()
    
    if quarterly_df.empty:
        print("\n❌ No quarterly data found!")
        return
    
    print("\n" + "="*70)
    print("Quarterly DataFrame (All Columns)")
    print("="*70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(quarterly_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("Quarterly DataFrame (Base Columns for Model)")
    print("="*70)
    base_cols = ['report_date', 'ticker', 'revenue', 'net_income', 'eps',
                 'operating_cash_flow', 'total_assets', 'total_liabilities']
    print(quarterly_df[base_cols].to_string(index=False))
    
    # Print detailed analysis
    print_quarterly_analysis(quarterly_df)
    
    print("\n" + "="*70)
    print("✓ Quarterly DataFrame Ready for Model Training")
    print("="*70)
    print(f"\nThis dataframe can now be used in model training along with:")
    print(f"  - News data (from CSV)")
    print(f"  - OHLCV data (when available)")
    print(f"  - Prices data (when available)")

if __name__ == "__main__":
    main()

