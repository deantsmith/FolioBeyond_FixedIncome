"""
Base model interface for portfolio optimization.

All portfolio models must inherit from BasePortfolioModel and implement
the required methods to ensure compatibility with the ensemble layer.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from ..config import Config
from ..interface.model_output import ModelOutput, ModelOutputTimeSeries
from ..data.loader import PortfolioData
from ..data.features import FeatureSet


class BasePortfolioModel(ABC):
    """
    Abstract base class for portfolio optimization models.
    
    All models must implement:
    - fit(): Train the model
    - predict(): Generate portfolio weights
    - save()/load(): Model persistence
    
    The predict() method must return a ModelOutput object to ensure
    compatibility with the ensemble layer.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "base_model"):
        """
        Initialize model.
        
        Args:
            config: Configuration object
            name: Model name for identification
        """
        self.config = config or Config()
        self.name = name
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.asset_names: Optional[List[str]] = None
        self.training_metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        **kwargs
    ) -> 'BasePortfolioModel':
        """
        Train the model on historical data.
        
        Args:
            features: FeatureSet containing training features and targets
            asset_names: List of asset names in the portfolio
            **kwargs: Additional model-specific parameters
            
        Returns:
            self (for chaining)
        """
        pass
    
    @abstractmethod
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """
        Generate raw predictions (before post-processing).
        
        Args:
            features: Feature values for prediction
            date: Date for the prediction
            
        Returns:
            Raw model output (e.g., expected returns per asset)
        """
        pass
    
    def predict(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp,
        apply_constraints: bool = True
    ) -> ModelOutput:
        """
        Generate portfolio weights with full post-processing.
        
        This method:
        1. Calls predict_raw() to get model predictions
        2. Converts predictions to portfolio weights
        3. Applies constraints (max weight, budget, non-negativity)
        4. Computes confidence score
        5. Returns standardized ModelOutput
        
        Args:
            features: Feature values for prediction
            date: Date for the prediction
            apply_constraints: Whether to apply portfolio constraints
            
        Returns:
            ModelOutput with portfolio weights
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get raw predictions
        raw_predictions = self.predict_raw(features, date)
        
        # Convert to weights
        weights = self._predictions_to_weights(raw_predictions)
        
        # Apply constraints if requested
        if apply_constraints:
            weights = self._apply_constraints(weights)
        
        # Compute confidence
        confidence = self._compute_confidence(features, raw_predictions)
        
        # Create output
        weights_series = pd.Series(weights, index=self.asset_names)
        
        return ModelOutput(
            date=date,
            weights=weights_series,
            confidence=confidence,
            metadata={
                'model': self.name,
                'raw_predictions': raw_predictions.tolist(),
                'constraints_applied': apply_constraints
            }
        )
    
    def _predictions_to_weights(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert raw predictions to portfolio weights.
        
        Default implementation uses softmax for long-only portfolios.
        Override for different weight schemes.
        
        Args:
            predictions: Raw model predictions (e.g., expected returns)
            
        Returns:
            Portfolio weights (sum to 1, non-negative)
        """
        # Softmax transformation for long-only
        # Shift to prevent overflow
        shifted = predictions - np.max(predictions)
        exp_preds = np.exp(shifted)
        weights = exp_preds / exp_preds.sum()
        
        return weights
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply portfolio constraints to weights.
        
        Constraints:
        - Budget: weights sum to 1
        - Max weight: no single position > MAX_WEIGHT_PER_ASSET
        - Non-negativity: all weights >= 0 (long-only)
        
        Args:
            weights: Unconstrained weights
            
        Returns:
            Constrained weights
        """
        # Ensure non-negative (long-only)
        weights = np.maximum(weights, self.config.MIN_WEIGHT_PER_ASSET)
        
        # Cap maximum weight
        max_weight = self.config.MAX_WEIGHT_PER_ASSET
        weights = np.minimum(weights, max_weight)
        
        # Renormalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback to equal weight
            weights = np.ones(len(weights)) / len(weights)
        
        # Re-apply max weight cap after normalization (iterative)
        for _ in range(5):  # Max iterations
            excess = np.maximum(weights - max_weight, 0)
            if excess.sum() < 1e-6:
                break
            weights = np.minimum(weights, max_weight)
            deficit = 1.0 - weights.sum()
            # Redistribute deficit to non-capped positions
            uncapped_mask = weights < max_weight
            if uncapped_mask.sum() > 0:
                weights[uncapped_mask] += deficit / uncapped_mask.sum()
        
        # Final normalization
        weights = weights / weights.sum()
        
        return weights
    
    def _compute_confidence(
        self, 
        features: pd.DataFrame, 
        predictions: np.ndarray
    ) -> float:
        """
        Compute model confidence in the prediction.
        
        Default implementation based on prediction dispersion.
        Override for model-specific confidence metrics.
        
        Args:
            features: Input features
            predictions: Raw predictions
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on prediction spread
        # Higher spread = more confident about relative rankings
        pred_std = np.std(predictions)
        pred_range = np.max(predictions) - np.min(predictions)
        
        # Normalize to 0-1 range (heuristic)
        if pred_range > 0:
            # More spread = higher confidence (up to a point)
            confidence = min(pred_range * 10, 1.0)
        else:
            confidence = 0.5  # Uncertain
        
        return confidence
    
    def backtest(
        self, 
        features: FeatureSet,
        returns: pd.DataFrame,
        rebalance_frequency: int = 21
    ) -> ModelOutputTimeSeries:
        """
        Run backtest over historical period.
        
        Args:
            features: FeatureSet with historical features
            returns: Historical returns for performance calculation
            rebalance_frequency: Days between rebalancing
            
        Returns:
            ModelOutputTimeSeries with all predictions
        """
        outputs = ModelOutputTimeSeries(model_name=self.name)
        
        dates = features.features.index
        rebalance_dates = dates[::rebalance_frequency]
        
        for date in rebalance_dates:
            try:
                # Get features for this date
                date_features = features.features.loc[[date]]
                
                # Generate prediction
                output = self.predict(date_features, date)
                outputs.add(output)
                
            except Exception as e:
                print(f"Warning: Prediction failed for {date}: {e}")
                continue
        
        return outputs
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        state = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'asset_names': self.asset_names,
            'training_metadata': self.training_metadata,
            'config': {
                'MAX_WEIGHT_PER_ASSET': self.config.MAX_WEIGHT_PER_ASSET,
                'MIN_WEIGHT_PER_ASSET': self.config.MIN_WEIGHT_PER_ASSET
            }
        }
        
        # Add model-specific state
        model_state = self._get_model_state()
        state['model_state'] = model_state
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, config: Optional[Config] = None) -> 'BasePortfolioModel':
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            config: Optional config override
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(config=config, name=state['name'])
        model.is_fitted = state['is_fitted']
        model.feature_names = state['feature_names']
        model.asset_names = state['asset_names']
        model.training_metadata = state['training_metadata']
        
        # Load model-specific state
        model._set_model_state(state['model_state'])
        
        print(f"Model loaded from {path}")
        return model
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for serialization."""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model-specific state from deserialization."""
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores.
        
        Returns:
            Series of importance scores indexed by feature name,
            or None if not available
        """
        return None  # Override in subclasses that support this
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
