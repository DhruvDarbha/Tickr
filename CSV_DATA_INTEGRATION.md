# CSV Data Integration Guide

## Overview

The training pipeline now supports loading news data from CSV files. This allows you to use pre-collected news data without needing to make API calls every time.

## CSV Format

The CSV file should be named: `{TICKER}_all_dates_data.csv` (e.g., `MSFT_all_dates_data.csv`)

### Required Columns:
- `published_at`: Timestamp in format "YYYY-MM-DD HH:MM:SS+00:00" (UTC timezone)
- `ticker`: Stock ticker symbol (e.g., "MSFT")
- `sentiment_score`: Sentiment score (float, typically -1.0 to 1.0)

### Example CSV Structure:
```csv
date_requested,uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,ticker,sentiment_score
2024-02-22,69b60748-...,OpenAI CEO...,...,...,...,...,...,en,2024-02-22 23:28:30+00:00,cnbc.com,MSFT,0.7845
```

## Usage

### Automatic Loading

The training pipeline automatically looks for CSV files when you specify a ticker:

```bash
python train_trading_model.py --ticker MSFT
```

The pipeline will:
1. First try to load news from `MSFT_all_dates_data.csv`
2. If CSV not found or empty, fall back to Marketaux API (if API key provided)
3. If neither available, use empty DataFrame

### Manual Loading in Code

```python
from load_data import load_news_for_ticker

# Load news for a specific ticker
news_df = load_news_for_ticker('MSFT')

# Or load from specific CSV file
from load_data import load_news_from_csv
news_df = load_news_from_csv('MSFT_all_dates_data.csv', ticker='MSFT')
```

## Current Status

### ✅ Completed:
- **News Data (MSFT)**: `MSFT_all_dates_data.csv` - 90 articles loaded successfully
  - Date range: 2024-02-22 to 2025-12-10
  - Sentiment range: -0.670 to 0.920
  - Average sentiment: 0.329

### ⏳ Pending:
- **OHLCV Data**: Historical price data (Open, High, Low, Close, Volume)
- **Stock Prices**: Latest price snapshots
- **Quarterly Reports**: Financial data

## Integration with Training Pipeline

The `train_trading_model.py` script has been updated to:

1. **Automatically detect CSV files** for the specified ticker
2. **Load news data from CSV** if available
3. **Fall back to API** if CSV not found (requires API key)
4. **Handle missing data gracefully** (other dataframes can be empty)

### Example Training Command:

```bash
# Train with CSV news data (no API key needed if CSV exists)
python train_trading_model.py --ticker MSFT --model_type lightgbm

# Train with API fallback (if CSV missing)
python train_trading_model.py --ticker MSFT --marketaux_key YOUR_KEY --model_type lightgbm
```

## Data Flow

```
CSV File (MSFT_all_dates_data.csv)
    ↓
load_news_for_ticker('MSFT')
    ↓
news_df [published_at, ticker, sentiment]
    ↓
align_dataframes() - Feature Engineering
    ↓
Trading Model Training
```

## Testing

Run the test script to verify CSV loading:

```bash
python test_csv_loading.py
```

This will:
- Load MSFT news from CSV
- Test feature engineering integration
- Verify data format and structure

## Next Steps

Once the other 3 dataframes are ready:

1. **OHLCV Data**: Create `MSFT_ohlcv_data.csv` with columns:
   - `timestamp`, `ticker`, `open`, `high`, `low`, `close`, `volume`

2. **Prices Data**: Create `MSFT_prices_data.csv` with columns:
   - `timestamp`, `ticker`, `price`

3. **Quarterly Data**: Create `MSFT_quarterly_data.csv` with columns:
   - `report_date`, `ticker`, `revenue`, `net_income`, `eps`, `operating_cash_flow`, `total_assets`, `total_liabilities`

4. **Update load_data.py** to add loading functions for these CSV files

5. **Update train_trading_model.py** to load all CSV files automatically

## Notes

- CSV files should be in the same directory as the scripts (or specify `data_dir` parameter)
- All timestamps must be UTC timezone-aware
- The pipeline handles missing data gracefully (empty DataFrames are fine)
- News data is automatically deduplicated and sorted by timestamp

