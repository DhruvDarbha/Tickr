# Trading Model Architecture

## Overview

The trading model architecture has been redesigned for **intraday/minute-level trading**. The model provides:

1. **Trading Signal**: BUY, SELL, or HOLD
2. **Position Size**: Number of shares to buy/sell
3. **Exit/Entry Timing**: When to sell (if BUY) or buy back (if SELL) - **in MINUTES**
4. **Expected Profit**: Predicted profit percentage and dollar amount

## Model Output Example

```
TRADING RECOMMENDATION: AAPL
======================================================================
Current Price: $129.23

Signal: BUY
Confidence: 99.5%

Action: BUY 100 shares
Total Investment: $12923.07

Exit Strategy:
  → SELL in 19m (19 minutes)
  → Expected Profit: 6.159%
  → Expected Profit: $795.87
```

## Architecture Components

### 1. Multi-Model System (`trading_model.py`)

The trading model uses **4 separate models**:

- **Signal Model** (Classification): Predicts BUY/SELL/HOLD
- **Position Model** (Regression): Predicts number of shares
- **Timing Model** (Regression): Predicts **minutes** until exit/entry
- **Profit Model** (Regression): Predicts expected profit %

### 2. Trading Target Generation

The model creates trading targets by:
- Looking ahead in price data (up to 240 minutes / 4 hours)
- Finding optimal entry/exit points within minutes
- Calculating maximum profit opportunities
- Only generating signals when profit threshold is met (default: 0.1% for minute-level)

### 3. Intraday/Minute-Level Focus

- **Max Holding Minutes**: 240 minutes (4 hours) - configurable
- **Min Profit Threshold**: 0.1% (configurable, optimized for minute-level trading)
- **Position Sizing**: Based on confidence and expected profit
- **Time Horizon**: Minutes, not days - perfect for day trading

## Usage

### Training the Model

```bash
python train_trading_model.py \
    --ticker AAPL \
    --marketaux_key YOUR_API_KEY \
    --model_type lightgbm \
    --max_position_size 100 \
    --min_profit_threshold 0.001 \
    --max_holding_minutes 240 \
    --save_path models/aapl_trading_model.pkl
```

### Using the Model

```python
from trading_model import TradingModel
from train_trading_model import get_trading_recommendation, print_trading_recommendation

# Load model
model = TradingModel()
model.load('models/aapl_trading_model.pkl')

# Get recommendation
recommendation = get_trading_recommendation(
    model=model,
    news_df=news_df,
    ohlcv_df=ohlcv_df,
    prices_df=prices_df,
    quarterly_df=quarterly_df,
    current_price=150.0
)

# Print formatted recommendation
print_trading_recommendation(recommendation)
```

## Model Parameters

### Signal Generation
- **BUY**: When model predicts price will increase by at least `min_profit_threshold`
- **SELL**: When model predicts price will decrease (opportunity to buy back lower)
- **HOLD**: When no clear profit opportunity exists

### Position Sizing
- Scales with expected profit magnitude
- Maximum position size: `max_position_size` (default: 100 shares)
- Formula: `position_size = max_position_size * confidence`

### Exit/Entry Timing
- **If BUY**: Predicts days until optimal SELL point
- **If SELL**: Predicts days until optimal BUY back point
- **If HOLD**: No timing (0 days)

## Example Output Interpretation

### BUY Signal
```
Signal: BUY
Action: BUY 100 shares
Exit Strategy: SELL in 19m (19 minutes)
Expected Profit: 6.159% ($795.87)
```
**Meaning**: Buy 100 shares now, sell in 19 minutes for ~6.16% profit.

### SELL Signal
```
Signal: SELL
Action: SELL 50 shares
Entry Strategy: BUY back in 15m (15 minutes)
Expected Profit: 0.5% ($32.30)
```
**Meaning**: Sell 50 shares now, buy back in 15 minutes at lower price for ~0.5% profit.

### HOLD Signal
```
Signal: HOLD
Recommendation: HOLD - No action recommended
```
**Meaning**: Wait for better entry/exit signals.

## Placeholder Data

The `ml_example.py` uses placeholder data:
- **OHLCV**: 100 days of simulated price data
- **News**: 34 articles with random sentiment scores
- **Prices**: 1 latest price record
- **Quarterly**: 4 quarterly reports

This allows testing the architecture before real data is available.

## Model Evaluation Metrics

- **Signal Accuracy**: % of correct BUY/SELL/HOLD predictions
- **Position Size MAE**: Mean absolute error in share predictions
- **Timing MAE**: Mean absolute error in **minutes**
- **Profit MAE**: Mean absolute error in profit %

## Configuration

### Key Parameters

```python
model = TradingModel(
    model_type='lightgbm',          # 'lightgbm', 'xgboost', 'random_forest'
    max_position_size=100,          # Max shares per trade
    min_profit_threshold=0.001,     # 0.1% minimum profit (for minute-level)
    max_holding_minutes=240        # Max 240 minutes (4 hours) for intraday
)
```

## Files

- `trading_model.py`: Core trading model architecture
- `train_trading_model.py`: Training pipeline
- `ml_example.py`: Example with placeholder data
- `feature_engineering.py`: Feature engineering (unchanged)
- `data_ingestion.py`: Data loading (unchanged)

## Next Steps

1. **Populate Real Data**: Once all 4 dataframes are ready with real data
2. **Train Model**: Use `train_trading_model.py` with real data
3. **Optimize**: Tune hyperparameters for better performance
4. **Backtest**: Test on historical data
5. **Deploy**: Use for live trading recommendations

## Notes

- Model is designed for **intraday/minute-level trading** (max 240 minutes / 4 hours)
- Requires minimum profit threshold (0.1%) to avoid noise
- Position sizing scales with confidence
- All predictions are based on maximizing profit opportunities within minutes
- Perfect for day trading and scalping strategies
- Timing predictions are in **minutes**, not days

