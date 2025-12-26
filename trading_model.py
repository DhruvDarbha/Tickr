"""
Trading Model Architecture

This module provides a trading-focused ML architecture that predicts:
1. Trading Signal: BUY, SELL, or HOLD
2. Position Size: Number of shares to buy/sell
3. Exit Timing: Minutes until optimal exit (if BUY) or entry (if SELL)
4. Expected Profit: Predicted profit percentage

Designed for intraday/minute-level trading horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TradingModel:
    """
    Multi-output trading model that predicts:
    - Trading signal (BUY/SELL/HOLD)
    - Position size (shares)
    - Exit/Entry timing (minutes)
    - Expected profit (%)
    
    Designed for intraday/minute-level trading.
    """
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        max_position_size: int = 100,
        min_profit_threshold: float = 0.001,  # 0.1% minimum profit for minute-level
        max_holding_minutes: int = 240,  # Max 4 hours (240 minutes) for intraday trading
        model_params: Optional[Dict] = None
    ):
        """
        Initialize trading model.
        
        Args:
            model_type: 'lightgbm', 'xgboost', or 'random_forest'
            max_position_size: Maximum number of shares per trade
            min_profit_threshold: Minimum profit % to trigger signal (0.1% for minute-level)
            max_holding_minutes: Maximum minutes to hold position (default: 240 = 4 hours)
            model_params: Model-specific parameters
        """
        self.model_type = model_type.lower()
        self.max_position_size = max_position_size
        self.min_profit_threshold = min_profit_threshold
        self.max_holding_minutes = max_holding_minutes
        
        # Default parameters
        if model_params is None:
            model_params = self._get_default_params()
        self.model_params = model_params
        
        # Initialize models (one for each prediction task)
        self.signal_model = None  # Classification: BUY/SELL/HOLD
        self.position_model = None  # Regression: number of shares
        self.timing_model = None  # Regression: minutes until exit/entry
        self.profit_model = None  # Regression: expected profit %
        
        self.feature_names_ = None
        self._initialize_models()
    
    def _get_default_params(self) -> Dict:
        """Get default parameters."""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'multiclass',  # For signal classification
                'num_class': 3,  # BUY, SELL, HOLD
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        else:
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def _initialize_models(self):
        """Initialize all models."""
        # Signal model (classification)
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            signal_params = {k: v for k, v in self.model_params.items() 
                           if k not in ['objective', 'num_class', 'metric']}
            self.signal_model = lgb.LGBMClassifier(**signal_params)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            signal_params = {k: v for k, v in self.model_params.items() 
                           if k not in ['objective', 'num_class', 'eval_metric']}
            self.signal_model = xgb.XGBClassifier(**signal_params)
        elif SKLEARN_AVAILABLE:
            self.signal_model = RandomForestClassifier(**self.model_params)
        else:
            raise ImportError("No ML libraries available")
        
        # Position, timing, and profit models (regression)
        reg_params = {k: v for k, v in self.model_params.items() 
                     if k not in ['objective', 'num_class', 'metric', 'eval_metric']}
        
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.position_model = lgb.LGBMRegressor(**reg_params)
            self.timing_model = lgb.LGBMRegressor(**reg_params)
            self.profit_model = lgb.LGBMRegressor(**reg_params)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.position_model = xgb.XGBRegressor(**reg_params)
            self.timing_model = xgb.XGBRegressor(**reg_params)
            self.profit_model = xgb.XGBRegressor(**reg_params)
        elif SKLEARN_AVAILABLE:
            self.position_model = RandomForestRegressor(**self.model_params)
            self.timing_model = RandomForestRegressor(**self.model_params)
            self.profit_model = RandomForestRegressor(**self.model_params)
    
    def _create_trading_targets(
        self,
        prices: pd.Series,
        timestamps: Optional[pd.Series] = None,
        max_lookahead_minutes: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Create trading targets from price data (minute-level).
        
        Args:
            prices: Series of prices
            timestamps: Series of timestamps (optional, for minute calculation)
            max_lookahead_minutes: Maximum minutes to look ahead (default: max_holding_minutes)
        
        Returns:
            Tuple of (signals, position_sizes, timings_minutes, profits)
        """
        if max_lookahead_minutes is None:
            max_lookahead_minutes = self.max_holding_minutes
        
        signals = []
        position_sizes = []
        timings_minutes = []
        profits = []
        
        # If timestamps provided, calculate actual minute differences
        # Otherwise, assume each step is 1 minute
        use_actual_timing = timestamps is not None and len(timestamps) == len(prices)
        
        for i in range(len(prices)):
            current_price = prices.iloc[i]
            
            # Look ahead to find best exit/entry
            best_profit = 0
            best_minutes = 0
            best_signal = 1  # HOLD (0=SELL, 1=HOLD, 2=BUY)
            
            # Check future prices (look ahead up to max_lookahead_minutes)
            max_steps = min(max_lookahead_minutes, len(prices) - i - 1)
            
            for step_ahead in range(1, max_steps + 1):
                if i + step_ahead >= len(prices):
                    break
                    
                future_price = prices.iloc[i + step_ahead]
                
                # Calculate minutes ahead
                if use_actual_timing:
                    time_diff = timestamps.iloc[i + step_ahead] - timestamps.iloc[i]
                    minutes_ahead = int(time_diff.total_seconds() / 60)
                    if minutes_ahead > max_lookahead_minutes:
                        break
                else:
                    minutes_ahead = step_ahead  # Assume 1 minute per step
                
                # Calculate profit for BUY
                buy_profit = (future_price - current_price) / current_price
                if buy_profit > best_profit and buy_profit >= self.min_profit_threshold:
                    best_profit = buy_profit
                    best_minutes = minutes_ahead
                    best_signal = 2  # BUY
                
                # Calculate profit for SELL (if we had position)
                # For SELL signal, we're looking for when to buy back (lower price)
                sell_profit = (current_price - future_price) / current_price
                if sell_profit > best_profit and sell_profit >= self.min_profit_threshold:
                    best_profit = sell_profit
                    best_minutes = minutes_ahead
                    best_signal = 0  # SELL
            
            signals.append(best_signal)
            
            # Position size based on confidence (profit magnitude)
            if best_signal != 1:  # Not HOLD
                # Scale position size by expected profit (more profit = more shares)
                # For minute-level, normalize by 0.01 (1%) instead of 0.1 (10%)
                confidence = min(abs(best_profit) / 0.01, 1.0)  # Normalize to 0-1
                position_size = int(self.max_position_size * confidence)
                position_sizes.append(position_size)
                timings_minutes.append(best_minutes)
                profits.append(best_profit * 100)  # Convert to percentage
            else:
                position_sizes.append(0)
                timings_minutes.append(0)
                profits.append(0.0)
        
        return (
            pd.Series(signals, name='signal'),
            pd.Series(position_sizes, name='position_size'),
            pd.Series(timings_minutes, name='timing_minutes'),
            pd.Series(profits, name='profit')
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        prices: pd.Series,
        timestamps: Optional[pd.Series] = None,
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        """
        Train all models.
        
        Args:
            X: Feature DataFrame
            prices: Price series for creating targets
            timestamps: Optional timestamp series for minute calculation
            validation_split: Validation split
            verbose: Print progress
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create trading targets
        if verbose:
            print("Creating trading targets from price data (minute-level)...")
        signals, position_sizes, timings_minutes, profits = self._create_trading_targets(
            prices, timestamps=timestamps
        )
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_signal_train = signals.iloc[:split_idx]
        y_signal_val = signals.iloc[split_idx:]
        y_position_train = position_sizes.iloc[:split_idx]
        y_position_val = position_sizes.iloc[split_idx:]
        y_timing_train = timings_minutes.iloc[:split_idx]
        y_timing_val = timings_minutes.iloc[split_idx:]
        y_profit_train = profits.iloc[:split_idx]
        y_profit_val = profits.iloc[split_idx:]
        
        # Train signal model (only on non-HOLD samples for better learning)
        if verbose:
            print("\nTraining signal model (BUY/SELL/HOLD)...")
        signal_mask_train = y_signal_train != 1
        if signal_mask_train.sum() > 0:
            self.signal_model.fit(
                X_train[signal_mask_train],
                y_signal_train[signal_mask_train]
            )
        else:
            self.signal_model.fit(X_train, y_signal_train)
        
        # Train position size model (only on non-zero positions)
        if verbose:
            print("Training position size model...")
        position_mask_train = y_position_train > 0
        if position_mask_train.sum() > 0:
            self.position_model.fit(
                X_train[position_mask_train],
                y_position_train[position_mask_train]
            )
        else:
            self.position_model.fit(X_train, y_position_train)
        
        # Train timing model (only on non-zero timings)
        if verbose:
            print("Training timing model...")
        timing_mask_train = y_timing_train > 0
        if timing_mask_train.sum() > 0:
            self.timing_model.fit(
                X_train[timing_mask_train],
                y_timing_train[timing_mask_train]
            )
        else:
            self.timing_model.fit(X_train, y_timing_train)
        
        # Train profit model (only on non-zero profits)
        if verbose:
            print("Training profit prediction model...")
        profit_mask_train = y_profit_train != 0
        if profit_mask_train.sum() > 0:
            self.profit_model.fit(
                X_train[profit_mask_train],
                y_profit_train[profit_mask_train]
            )
        else:
            self.profit_model.fit(X_train, y_profit_train)
        
        if verbose:
            print("\n✓ All models trained successfully!")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make trading predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with columns: signal, position_size, timing_days, expected_profit
        """
        # Predict signal
        signal_pred = self.signal_model.predict(X)
        signal_proba = self.signal_model.predict_proba(X)
        
        # Predict position size
        position_pred = self.position_model.predict(X)
        position_pred = np.clip(position_pred, 0, self.max_position_size).astype(int)
        
        # Predict timing (in minutes)
        timing_pred = self.timing_model.predict(X)
        timing_pred = np.clip(timing_pred, 1, self.max_holding_minutes).astype(int)
        
        # Predict profit
        profit_pred = self.profit_model.predict(X)
        
        # Adjust predictions based on signal
        # If HOLD, set position and timing to 0
        hold_mask = signal_pred == 1
        position_pred[hold_mask] = 0
        timing_pred[hold_mask] = 0
        profit_pred[hold_mask] = 0.0
        
        # Map signal numbers to labels
        signal_labels = np.where(signal_pred == 0, 'SELL',
                                np.where(signal_pred == 1, 'HOLD', 'BUY'))
        
        results = pd.DataFrame({
            'signal': signal_labels,
            'position_size': position_pred,
            'timing_minutes': timing_pred,
            'expected_profit_pct': profit_pred,
            'signal_confidence': np.max(signal_proba, axis=1)
        })
        
        return results
    
    def evaluate(self, X: pd.DataFrame, prices: pd.Series, timestamps: Optional[pd.Series] = None, verbose: bool = True) -> Dict:
        """Evaluate model performance."""
        # Create true targets
        signals_true, position_sizes_true, timings_true, profits_true = self._create_trading_targets(
            prices, timestamps=timestamps
        )
        
        # Predict
        predictions = self.predict(X)
        
        # Map true signals to numbers
        signal_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
        signals_true_num = signals_true.map({0: 0, 1: 1, 2: 2})
        signals_pred_num = predictions['signal'].map(signal_map)
        
        # Signal accuracy
        signal_accuracy = accuracy_score(signals_true_num, signals_pred_num)
        
        # Position size MAE (only for non-HOLD)
        non_hold_mask = signals_true_num != 1
        if non_hold_mask.sum() > 0:
            position_mae = np.mean(np.abs(
                position_sizes_true[non_hold_mask].values - 
                predictions['position_size'].values[non_hold_mask]
            ))
        else:
            position_mae = 0
        
        # Timing MAE (only for non-HOLD) - in minutes
        if non_hold_mask.sum() > 0:
            timing_mae = np.mean(np.abs(
                timings_true[non_hold_mask].values - 
                predictions['timing_minutes'].values[non_hold_mask]
            ))
        else:
            timing_mae = 0
        
        # Profit MAE
        if non_hold_mask.sum() > 0:
            profit_mae = np.mean(np.abs(
                profits_true[non_hold_mask].values - 
                predictions['expected_profit_pct'].values[non_hold_mask]
            ))
        else:
            profit_mae = 0
        
        metrics = {
            'signal_accuracy': signal_accuracy,
            'position_mae': position_mae,
            'timing_mae': timing_mae,
            'profit_mae': profit_mae
        }
        
        if verbose:
            print("\n" + "="*70)
            print("Trading Model Evaluation")
            print("="*70)
            print(f"Signal Accuracy:           {signal_accuracy:.4f} ({signal_accuracy*100:.2f}%)")
            print(f"Position Size MAE:        {position_mae:.2f} shares")
            print(f"Timing MAE:                {timing_mae:.2f} minutes")
            print(f"Profit Prediction MAE:     {profit_mae:.2f}%")
            print("="*70)
            
            # Confusion matrix
            print("\nSignal Confusion Matrix:")
            print(confusion_matrix(signals_true_num, signals_pred_num, labels=[0, 1, 2]))
            print("\nClassification Report:")
            print(classification_report(signals_true_num, signals_pred_num, 
                                      target_names=['SELL', 'HOLD', 'BUY']))
        
        return metrics
    
    def save(self, filepath: str):
        """Save model."""
        model_data = {
            'model_type': self.model_type,
            'max_position_size': self.max_position_size,
            'min_profit_threshold': self.min_profit_threshold,
            'max_holding_minutes': self.max_holding_minutes,
            'model_params': self.model_params,
            'feature_names_': self.feature_names_,
            'signal_model': self.signal_model,
            'position_model': self.position_model,
            'timing_model': self.timing_model,
            'profit_model': self.profit_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Trading model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.max_position_size = model_data['max_position_size']
        self.min_profit_threshold = model_data['min_profit_threshold']
        # Handle backward compatibility
        if 'max_holding_minutes' in model_data:
            self.max_holding_minutes = model_data['max_holding_minutes']
        else:
            # Convert old days to minutes (30 days = 43200 minutes, but use 240 for intraday)
            self.max_holding_minutes = model_data.get('max_holding_days', 30) * 1440  # days to minutes
        self.model_params = model_data['model_params']
        self.feature_names_ = model_data.get('feature_names_')
        self.signal_model = model_data['signal_model']
        self.position_model = model_data['position_model']
        self.timing_model = model_data['timing_model']
        self.profit_model = model_data['profit_model']
        
        print(f"Trading model loaded from {filepath}")

