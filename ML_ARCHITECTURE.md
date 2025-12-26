# ML Model Architecture

## Overview

This ML architecture is designed to predict stock prices using 4 different data sources:
1. **News Sentiment** - Marketaux API news articles with sentiment scores
2. **Historical OHLCV** - Open, High, Low, Close, Volume data
3. **Stock Prices** - Latest price snapshots
4. **Quarterly Reports** - Fundamental financial data

## Architecture Components

### 1. Feature Engineering (`feature_engineering.py`)

The feature engineering module combines all 4 dataframes into a unified feature set:

- **Temporal Alignment**: Aligns all data sources by timestamp
- **Technical Indicators**: 
  - Moving averages (SMA 5, 10, 20, 50)
  - Exponential moving averages (EMA 12, 26)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Price changes and volatility metrics
  - Volume indicators

- **News Sentiment Aggregation**:
  - Mean, count, std, max, min of sentiment scores
  - Aggregated over configurable time windows (default: 24 hours)

- **Quarterly Data Alignment**:
  - Forward-fills quarterly data to daily timestamps
  - Includes revenue, net income, EPS, cash flow, assets, liabilities

- **Time Features**:
  - Year, month, day, day of week, quarter
  - Cyclical encoding (sin/cos) for periodic patterns
  - Weekend indicators

### 2. ML Models (`ml_model.py`)

Supports multiple model types:

#### LightGBM (Recommended)
- Fast, efficient gradient boosting
- Handles large datasets well
- Good for structured/tabular data
- Built-in feature importance

#### XGBoost
- Robust gradient boosting
- Good regularization
- Handles missing values well

#### Random Forest
- Ensemble of decision trees
- Good baseline model
- Less prone to overfitting

#### Ensemble
- Combines all available models
- Averages predictions for robustness
- Better generalization

### 3. Training Pipeline (`train_model.py`)

Complete training workflow:
- Data loading and validation
- Feature engineering
- Train/validation split (time-aware)
- Model training with early stopping
- Evaluation metrics (MSE, RMSE, MAE, R², MAPE)
- Feature importance analysis
- Model saving/loading

## Usage

### Basic Training

```python
from train_model import train_stock_model
from data_ingestion import ingest_all_data

# Load data
data = ingest_all_data("AAPL", marketaux_api_key)
news_df = data['news_df']
ohlcv_df = data['prices_df']
prices_df = data['latest_price_df']
quarterly_df = data['financials_df']

# Train model
model = train_stock_model(
    ticker="AAPL",
    news_df=news_df,
    ohlcv_df=ohlcv_df,
    prices_df=prices_df,
    quarterly_df=quarterly_df,
    model_type='lightgbm',
    prediction_horizon=1,
    save_path='models/aapl_model.pkl'
)
```

### Command Line Training

```bash
python train_model.py \
    --ticker AAPL \
    --marketaux_key YOUR_API_KEY \
    --model_type lightgbm \
    --prediction_horizon 1 \
    --save_path models/aapl_model.pkl
```

### Using Trained Model for Predictions

```python
from ml_model import StockPredictionModel
from feature_engineering import align_dataframes, get_feature_columns

# Load model
model = StockPredictionModel()
model.load('models/aapl_model.pkl')

# Prepare latest features
features_df = align_dataframes(news_df, ohlcv_df, prices_df, quarterly_df)
feature_cols = get_feature_columns(features_df)
latest_features = features_df[feature_cols].iloc[-1:]

# Predict
prediction = model.predict(latest_features)[0]
print(f"Predicted price: ${prediction:.2f}")
```

## Model Selection Guide

### When to Use LightGBM
- **Recommended default choice**
- Large datasets (>10k samples)
- Need fast training
- Want feature importance
- Structured/tabular data

### When to Use XGBoost
- Need strong regularization
- Have missing values in data
- Want robust performance
- Similar to LightGBM but more conservative

### When to Use Random Forest
- Small datasets (<5k samples)
- Want interpretability
- Need baseline comparison
- Less computational resources

### When to Use Ensemble
- Want maximum robustness
- Have computational resources
- Production deployment
- Need best generalization

## Hyperparameter Optimization

The architecture is designed to be easily optimized. Key parameters to tune:

### LightGBM
```python
model_params = {
    'num_leaves': 31,          # Tree complexity
    'learning_rate': 0.05,      # Step size
    'n_estimators': 100,       # Number of trees
    'feature_fraction': 0.9,   # Feature sampling
    'bagging_fraction': 0.8,   # Data sampling
    'max_depth': -1            # Tree depth (-1 = unlimited)
}
```

### XGBoost
```python
model_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

## Evaluation Metrics

- **MSE** (Mean Squared Error): Penalizes large errors
- **RMSE** (Root Mean Squared Error): In same units as target
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Feature Importance

The model provides feature importance scores showing which features contribute most to predictions. Use this to:
- Understand model behavior
- Identify most predictive signals
- Remove irrelevant features
- Guide feature engineering

## Future Optimizations

Once all 4 dataframes are ready, consider:

1. **Hyperparameter Tuning**: Grid search or Bayesian optimization
2. **Feature Selection**: Remove low-importance features
3. **Cross-Validation**: Time-series cross-validation
4. **Ensemble Stacking**: Stack multiple models
5. **LSTM Integration**: Add LSTM for temporal patterns
6. **Feature Engineering**: Domain-specific indicators
7. **Regularization**: Prevent overfitting
8. **Model Monitoring**: Track performance over time

## Data Requirements

### News DataFrame
- Columns: `published_at`, `ticker`, `sentiment`
- UTC timezone-aware timestamps
- Sentiment scores: -1 to 1 (or 0 to 1)

### OHLCV DataFrame
- Columns: `timestamp`, `ticker`, `open`, `high`, `low`, `close`, `volume`
- Daily frequency recommended
- UTC timezone-aware timestamps

### Prices DataFrame
- Columns: `timestamp`, `ticker`, `price`
- Latest price snapshots
- UTC timezone-aware timestamps

### Quarterly DataFrame
- Columns: `report_date`, `ticker`, `revenue`, `net_income`, `eps`, `operating_cash_flow`, `total_assets`, `total_liabilities`
- Quarterly frequency
- UTC timezone-aware timestamps

## Notes

- All timestamps must be UTC timezone-aware
- Dataframes should be sorted by timestamp
- Missing data is handled via forward-fill and default values
- Model architecture is flexible and can be extended
- Designed for optimization once all data is available

