"""
Tree-based models for portfolio optimization.

Implements Random Forest, XGBoost, and LightGBM approaches.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import Config
from ..data.features import FeatureSet
from .base import BasePortfolioModel


class RandomForestPortfolioModel(BasePortfolioModel):
    """
    Random Forest model for portfolio weight prediction.
    
    Trains a separate regressor for each asset's expected return,
    then converts predictions to portfolio weights.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "random_forest"):
        super().__init__(config, name)
        self.models: Dict[str, Any] = {}  # One model per asset
        self.params = self.config.RANDOM_FOREST_PARAMS.copy()
    
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        **kwargs
    ) -> 'RandomForestPortfolioModel':
        """
        Train Random Forest models for each asset.
        
        Args:
            features: FeatureSet with training data
            asset_names: List of asset names
            **kwargs: Override default parameters
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Update params with any overrides
        params = {**self.params, **kwargs}
        
        self.asset_names = asset_names
        self.feature_names = features.features.columns.tolist()
        
        X = features.features.values
        
        if features.target is None:
            raise ValueError("FeatureSet must include target for training")
        
        print(f"\n🌲 Training Random Forest models for {len(asset_names)} assets...")
        
        for i, asset in enumerate(asset_names):
            if asset not in features.target.columns:
                print(f"   ⚠️ Skipping {asset}: not in target")
                continue
            
            y = features.target[asset].values
            
            # Remove NaN rows
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            if len(y_valid) < 100:
                print(f"   ⚠️ Skipping {asset}: insufficient data ({len(y_valid)} samples)")
                continue
            
            model = RandomForestRegressor(**params)
            model.fit(X_valid, y_valid)
            self.models[asset] = model
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Trained {i + 1}/{len(asset_names)} models")
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_assets': len(self.models),
            'params': params
        }
        
        print(f"   ✅ Training complete: {len(self.models)} models fitted")
        return self
    
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """
        Generate expected return predictions for each asset.
        """
        X = features.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = np.zeros(len(self.asset_names))
        
        for i, asset in enumerate(self.asset_names):
            if asset in self.models:
                predictions[i] = self.models[asset].predict(X)[0]
            else:
                predictions[i] = 0.0  # Neutral prediction for missing models
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get averaged feature importance across all asset models."""
        if not self.is_fitted or not self.models:
            return None
        
        importances = np.zeros(len(self.feature_names))
        
        for model in self.models.values():
            importances += model.feature_importances_
        
        importances /= len(self.models)
        
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {'models': self.models, 'params': self.params}
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.models = state['models']
        self.params = state['params']


class XGBoostPortfolioModel(BasePortfolioModel):
    """
    XGBoost model for portfolio weight prediction.
    
    Uses gradient boosting for potentially better performance
    than Random Forest, with built-in regularization.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "xgboost"):
        super().__init__(config, name)
        self.models: Dict[str, Any] = {}
        self.params = self.config.XGBOOST_PARAMS.copy()
    
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        early_stopping_rounds: int = 10,
        validation_fraction: float = 0.2,
        **kwargs
    ) -> 'XGBoostPortfolioModel':
        """
        Train XGBoost models with early stopping.
        
        Args:
            features: FeatureSet with training data
            asset_names: List of asset names
            early_stopping_rounds: Rounds for early stopping
            validation_fraction: Fraction for validation
            **kwargs: Override default parameters
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        params = {**self.params, **kwargs}
        
        self.asset_names = asset_names
        self.feature_names = features.features.columns.tolist()
        
        X = features.features.values
        n_samples = len(X)
        
        # Train/validation split
        split_idx = int(n_samples * (1 - validation_fraction))
        X_train, X_val = X[:split_idx], X[split_idx:]
        
        if features.target is None:
            raise ValueError("FeatureSet must include target for training")
        
        print(f"\n🚀 Training XGBoost models for {len(asset_names)} assets...")
        print(f"   Train samples: {split_idx}, Validation samples: {n_samples - split_idx}")
        
        for i, asset in enumerate(asset_names):
            if asset not in features.target.columns:
                continue
            
            y = features.target[asset].values
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Remove NaN
            train_valid = ~np.isnan(y_train)
            val_valid = ~np.isnan(y_val)
            
            if train_valid.sum() < 100:
                print(f"   ⚠️ Skipping {asset}: insufficient data")
                continue
            
            model = xgb.XGBRegressor(**params)
            
            model.fit(
                X_train[train_valid], 
                y_train[train_valid],
                eval_set=[(X_val[val_valid], y_val[val_valid])] if val_valid.sum() > 0 else None,
                verbose=False
            )
            
            self.models[asset] = model
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Trained {i + 1}/{len(asset_names)} models")
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'n_assets': len(self.models),
            'params': params,
            'early_stopping_rounds': early_stopping_rounds
        }
        
        print(f"   ✅ Training complete: {len(self.models)} models fitted")
        return self
    
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """Generate expected return predictions."""
        X = features.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = np.zeros(len(self.asset_names))
        
        for i, asset in enumerate(self.asset_names):
            if asset in self.models:
                predictions[i] = self.models[asset].predict(X)[0]
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get averaged feature importance (gain-based)."""
        if not self.is_fitted or not self.models:
            return None
        
        importances = np.zeros(len(self.feature_names))
        
        for model in self.models.values():
            importances += model.feature_importances_
        
        importances /= len(self.models)
        
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {'models': self.models, 'params': self.params}
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.models = state['models']
        self.params = state['params']


class LightGBMPortfolioModel(BasePortfolioModel):
    """
    LightGBM model for portfolio weight prediction.
    
    Fast gradient boosting with categorical feature support.
    Good for large feature sets and quick iteration.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "lightgbm"):
        super().__init__(config, name)
        self.models: Dict[str, Any] = {}
        self.params = {
            'n_estimators': 100,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.config.RANDOM_SEED,
            'verbose': -1
        }
    
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        **kwargs
    ) -> 'LightGBMPortfolioModel':
        """Train LightGBM models."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        params = {**self.params, **kwargs}
        
        self.asset_names = asset_names
        self.feature_names = features.features.columns.tolist()
        
        X = features.features.values
        
        if features.target is None:
            raise ValueError("FeatureSet must include target for training")
        
        print(f"\n⚡ Training LightGBM models for {len(asset_names)} assets...")
        
        for i, asset in enumerate(asset_names):
            if asset not in features.target.columns:
                continue
            
            y = features.target[asset].values
            valid_mask = ~np.isnan(y)
            
            if valid_mask.sum() < 100:
                continue
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X[valid_mask], y[valid_mask])
            self.models[asset] = model
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Trained {i + 1}/{len(asset_names)} models")
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_assets': len(self.models),
            'params': params
        }
        
        print(f"   ✅ Training complete: {len(self.models)} models fitted")
        return self
    
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """Generate expected return predictions."""
        X = features.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = np.zeros(len(self.asset_names))
        
        for i, asset in enumerate(self.asset_names):
            if asset in self.models:
                predictions[i] = self.models[asset].predict(X)[0]
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get averaged feature importance."""
        if not self.is_fitted or not self.models:
            return None
        
        importances = np.zeros(len(self.feature_names))
        
        for model in self.models.values():
            importances += model.feature_importances_
        
        importances /= len(self.models)
        
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {'models': self.models, 'params': self.params}
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.models = state['models']
        self.params = state['params']
