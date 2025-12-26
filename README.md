# Tickr

A data-driven stock analysis and trading recommendation system.

## Phase 1: Data Ingestion

This phase implements a modular data ingestion pipeline that retrieves all raw data needed for stock analysis.

### Features

- **Historical Price Data (OHLCV)**: 2 years of daily price data with automatic split/dividend adjustments
- **Latest Price**: Current price snapshot for live recommendations
- **Company News**: News articles with sentiment scores from Marketaux API
- **Quarterly Financials**: Fundamental data from quarterly reports

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Quick Start

```python
from data_ingestion import ingest_all_data

# Get Marketaux API key (required for news sentiment)
marketaux_api_key = "your_api_key_here"

# Fetch all data for a ticker
data = ingest_all_data("AAPL", marketaux_api_key)

# Access the data
prices_df = data['prices_df']          # Historical OHLCV
latest_price_df = data['latest_price_df']  # Current price (DataFrame)
news_df = data['news_df']              # News with sentiment scores
financials_df = data['financials_df']  # Quarterly reports
```

#### Individual Functions

```python
from data_ingestion import (
    fetch_historical_prices,
    fetch_latest_price,
    fetch_company_news,
    fetch_quarterly_financials
)
from datetime import datetime, timedelta
import pytz

# Marketaux API key (required for news)
marketaux_api_key = "your_api_key_here"

# Historical prices
end_date = datetime.now(pytz.UTC)
start_date = end_date - timedelta(days=730)
prices_df = fetch_historical_prices("AAPL", start_date, end_date, interval="1d")

# Latest price (returns DataFrame)
latest_price_df = fetch_latest_price("AAPL")

# Company news with sentiment from Marketaux
news_df = fetch_company_news("AAPL", start_date, end_date, marketaux_api_key)

# Financials
financials_df = fetch_quarterly_financials("AAPL")
```

### Data Outputs

All functions return clean Pandas DataFrames with:

- **prices_df**: Historical OHLCV (timestamp, ticker, open, high, low, close, volume)
- **latest_price_df**: Current price (timestamp, ticker, price)
- **news_df**: News articles with sentiment (published_at, ticker, sentiment)
- **financials_df**: Quarterly reports (report_date, ticker, revenue, net_income, eps, operating_cash_flow, total_assets, total_liabilities)

All DataFrames have:
- **UTC timezone-aware timestamps**
- **No duplicates**
- **No incomplete data**
- **Chronologically sorted**
- **Ready for downstream processing**

### Marketaux API

This pipeline uses the Marketaux API for fetching news articles with sentiment scores. You'll need to:
1. Sign up for a Marketaux API key at https://www.marketaux.com/
2. Provide the API key when calling `ingest_all_data()` or `fetch_company_news()`

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Example

Run the example script:

```bash
python example_usage.py
```

This will demonstrate fetching and displaying all data types for a sample ticker.