"""
ML Model Architecture for Stock Prediction

This module provides a flexible ML architecture that supports:
- LightGBM (primary recommendation)
- XGBoost
- Ensemble models
- Hyperparameter optimization ready

The architecture is designed to work with the 4 dataframes:
1. News sentiment
2. Historical OHLCV
3. Stock prices
4. Quarterly reports
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class StockPredictionModel:
    """
    Flexible ML model for stock price prediction.
    Supports multiple algorithms and ensemble methods.
    """
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        model_params: Optional[Dict] = None,
        use_ensemble: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'random_forest', 'ensemble')
            model_params: Dictionary of model-specific parameters
            use_ensemble: Whether to use ensemble of multiple models
        """
        self.model_type = model_type.lower()
        self.use_ensemble = use_ensemble
        self.models = []
        self.feature_importance_ = None
        self.feature_names_ = None
        
        # Default parameters for each model type
        if model_params is None:
            model_params = self._get_default_params()
        
        self.model_params = model_params
        
        # Initialize model(s)
        if self.use_ensemble or self.model_type == 'ensemble':
            self._initialize_ensemble()
        else:
            self._initialize_single_model()
    
    def _get_default_params(self) -> Dict:
        """Get default parameters for each model type."""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
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
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            return {}
    
    def _initialize_single_model(self):
        """Initialize a single model based on model_type."""
        if self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
            self.model = lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
            self.model = xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'random_forest':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is not installed. Install with: pip install scikit-learn")
            self.model = RandomForestRegressor(**self.model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _initialize_ensemble(self):
        """Initialize ensemble of multiple models."""
        self.models = []
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_params = self._get_default_params() if self.model_type == 'ensemble' else self.model_params
            lgb_params['model_type'] = 'lightgbm'
            self.models.append(('lightgbm', lgb.LGBMRegressor(**{k: v for k, v in lgb_params.items() 
                                                                  if k != 'model_type'})))
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_params = self._get_default_params() if self.model_type == 'ensemble' else self.model_params
            xgb_params['model_type'] = 'xgboost'
            self.models.append(('xgboost', xgb.XGBRegressor(**{k: v for k, v in xgb_params.items() 
                                                               if k != 'model_type'})))
        
        # Random Forest
        if SKLEARN_AVAILABLE:
            rf_params = self._get_default_params() if self.model_type == 'ensemble' else self.model_params
            rf_params['model_type'] = 'random_forest'
            self.models.append(('random_forest', RandomForestRegressor(**{k: v for k, v in rf_params.items() 
                                                                        if k != 'model_type'})))
        
        if not self.models:
            raise ImportError("No ML libraries available. Install at least one: lightgbm, xgboost, or scikit-learn")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            early_stopping_rounds: Early stopping rounds (for LightGBM/XGBoost)
            verbose: Whether to print training progress
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split data (time-aware split for time series)
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = X, X
            y_train, y_val = y, y
        
        if self.use_ensemble or self.model_type == 'ensemble':
            # Train ensemble models
            for name, model in self.models:
                if verbose:
                    print(f"Training {name}...")
                
                if name == 'lightgbm' and early_stopping_rounds > 0:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
                    )
                elif name == 'xgboost' and early_stopping_rounds > 0:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
            
            # Calculate feature importance (average across models)
            importances = []
            for name, model in self.models:
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            if importances:
                self.feature_importance_ = np.mean(importances, axis=0)
        else:
            # Train single model
            if verbose:
                print(f"Training {self.model_type}...")
            
            if self.model_type == 'lightgbm' and early_stopping_rounds > 0:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
                )
            elif self.model_type == 'xgboost' and early_stopping_rounds > 0:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance_ = self.model.feature_importances_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if self.use_ensemble or self.model_type == 'ensemble':
            # Average predictions from all models
            predictions = []
            for name, model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        else:
            return self.model.predict(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True target values
            verbose: Whether to print metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        if verbose:
            print("\n" + "="*50)
            print("Model Evaluation Metrics")
            print("="*50)
            print(f"Mean Squared Error (MSE):     {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE):     {mae:.4f}")
            print(f"R-squared (R²):               {r2:.4f}")
            print(f"Mean Absolute % Error (MAPE): {mape:.2f}%")
            print("="*50)
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if self.feature_names_ is None:
            self.feature_names_ = [f'feature_{i}' for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model_type': self.model_type,
            'use_ensemble': self.use_ensemble,
            'model_params': self.model_params,
            'feature_names_': self.feature_names_,
            'feature_importance_': self.feature_importance_
        }
        
        if self.use_ensemble or self.model_type == 'ensemble':
            model_data['models'] = self.models
        else:
            model_data['model'] = self.model
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.use_ensemble = model_data['use_ensemble']
        self.model_params = model_data['model_params']
        self.feature_names_ = model_data.get('feature_names_')
        self.feature_importance_ = model_data.get('feature_importance_')
        
        if self.use_ensemble or self.model_type == 'ensemble':
            self.models = model_data['models']
        else:
            self.model = model_data['model']
        
        print(f"Model loaded from {filepath}")

