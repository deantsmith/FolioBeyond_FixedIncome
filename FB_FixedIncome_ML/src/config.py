"""
Configuration module for ML portfolio optimization.

Centralizes all configuration parameters and provides environment-aware
path resolution.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    """Centralized configuration for ML portfolio optimization."""
    
    # ==========================================================================
    # Time Constants
    # ==========================================================================
    TRADING_DAYS_PER_YEAR: int = 253
    SQRT_TRADING_DAYS: float = field(default_factory=lambda: np.sqrt(253))
    
    # ==========================================================================
    # Data Parameters
    # ==========================================================================
    SCALE_FACTOR_FOR_RETURNS: int = 100  # Returns in data are scaled by 100
    
    # Training windows
    LOOKBACK_WINDOW_DAYS: int = 730      # ~2 years for training
    VALIDATION_WINDOW_DAYS: int = 63     # ~3 months for validation
    TEST_WINDOW_DAYS: int = 63           # ~3 months for testing
    
    # Rebalancing
    REBALANCE_FREQUENCY_DAYS: int = 21   # Monthly rebalancing
    
    # ==========================================================================
    # Portfolio Constraints
    # ==========================================================================
    MAX_WEIGHT_PER_ASSET: float = 0.30   # 30% max per asset
    MIN_WEIGHT_PER_ASSET: float = 0.00   # Long-only by default
    
    # Volatility targets (annualized)
    VOLATILITY_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'Low': 0.025,
        'Moderate': 0.035,
        'High': 0.045,
        'RRS': 0.15
    })
    
    DEFAULT_VOLATILITY_TARGET: str = 'Moderate'
    
    # Stress test
    STRESS_TEST_MAX_LOSS: float = 15.0   # Maximum allowed stress loss %
    
    # ==========================================================================
    # Feature Engineering
    # ==========================================================================
    # Lookback periods for feature calculation (in trading days)
    FEATURE_LOOKBACKS: List[int] = field(default_factory=lambda: [5, 21, 63, 126, 252])
    
    # Moving average parameters
    MA_SHORT_LENGTH: int = 5
    MA_LONG_LENGTH: int = 120
    
    # ==========================================================================
    # Model Parameters
    # ==========================================================================
    # Tree models
    XGBOOST_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    })
    
    RANDOM_FOREST_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    })
    
    # Neural network
    NEURAL_PARAMS: Dict = field(default_factory=lambda: {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    })
    
    LSTM_PARAMS: Dict = field(default_factory=lambda: {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 21,  # ~1 month of history
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    })
    
    # ==========================================================================
    # Training Parameters
    # ==========================================================================
    TRAIN_TEST_SPLIT: float = 0.8
    CROSS_VAL_FOLDS: int = 5
    RANDOM_SEED: int = 42
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    DEFAULT_BASE_PATH: str = '/Users/deansmith/FolioBeyond Dropbox/Dean Smith/SharedFolioBeyond'
    
    def get_base_path(self) -> Path:
        """
        Get base data path from environment or default.
        
        Priority:
        1. FOLIOBEYOND_BASE_PATH environment variable
        2. Default path
        """
        base_path = os.environ.get('FOLIOBEYOND_BASE_PATH')
        if base_path:
            return Path(base_path)
        return Path(self.DEFAULT_BASE_PATH)
    
    def get_model_output_path(self) -> Path:
        """Get path for saving model artifacts."""
        output_path = os.environ.get('MODEL_OUTPUT_PATH')
        if output_path:
            return Path(output_path)
        return Path(__file__).parent.parent / 'models'
    
    def get_torch_device(self) -> str:
        """Get PyTorch device from environment or auto-detect."""
        device = os.environ.get('TORCH_DEVICE')
        if device:
            return device
        
        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    def get_volatility_target(self, portfolio_type: str) -> float:
        """Get volatility target for portfolio type."""
        return self.VOLATILITY_TARGETS.get(
            portfolio_type, 
            self.VOLATILITY_TARGETS[self.DEFAULT_VOLATILITY_TARGET]
        )


# Global config instance
config = Config()
