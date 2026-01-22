"""
Standardized model output interface for ensemble integration.

This module defines the contract that all portfolio models must follow
to integrate with the FolioBeyond ensemble layer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import json


@dataclass
class ModelOutput:
    """
    Standardized output contract for portfolio optimization models.
    
    All models (MVO, ML, etc.) should produce this output format to enable
    seamless integration with the ensemble layer.
    
    Attributes:
        date: The date for which these weights are valid
        weights: Portfolio weights indexed by asset name (must sum to ~1.0 for long-only)
        confidence: Model's confidence in this allocation (0.0 to 1.0)
        metadata: Model-specific diagnostic information
        
    Example:
        >>> output = ModelOutput(
        ...     date=pd.Timestamp('2024-01-15'),
        ...     weights=pd.Series({'TLT': 0.3, 'IEF': 0.25, 'LQD': 0.20, 'HYG': 0.15, 'EMB': 0.10}),
        ...     confidence=0.85,
        ...     metadata={'model': 'xgboost', 'train_sharpe': 1.2}
        ... )
    """
    
    date: pd.Timestamp
    weights: pd.Series
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate output after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate the model output meets contract requirements."""
        # Ensure date is Timestamp
        if not isinstance(self.date, pd.Timestamp):
            self.date = pd.Timestamp(self.date)
        
        # Ensure weights is Series
        if not isinstance(self.weights, pd.Series):
            self.weights = pd.Series(self.weights)
        
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Check weights sum (allow some tolerance for long-only)
        weight_sum = self.weights.sum()
        if abs(weight_sum - 1.0) > 0.1:  # 10% tolerance for long/short strategies
            import warnings
            warnings.warn(f"Weights sum to {weight_sum:.4f}, expected ~1.0")
    
    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio."""
        return len(self.weights)
    
    @property
    def gross_exposure(self) -> float:
        """Sum of absolute weights."""
        return self.weights.abs().sum()
    
    @property
    def net_exposure(self) -> float:
        """Sum of weights (long - short)."""
        return self.weights.sum()
    
    @property
    def long_exposure(self) -> float:
        """Sum of positive weights."""
        return self.weights[self.weights > 0].sum()
    
    @property
    def short_exposure(self) -> float:
        """Sum of negative weights (as positive number)."""
        return abs(self.weights[self.weights < 0].sum())
    
    @property
    def effective_n_assets(self) -> float:
        """
        Effective number of assets (inverse HHI).
        Higher = more diversified.
        """
        w_squared = (self.weights ** 2).sum()
        if w_squared == 0:
            return 0.0
        return 1.0 / w_squared
    
    @property
    def max_weight(self) -> float:
        """Maximum absolute weight."""
        return self.weights.abs().max()
    
    @property
    def active_assets(self) -> List[str]:
        """List of assets with non-zero weights."""
        return self.weights[self.weights.abs() > 1e-6].index.tolist()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date.isoformat(),
            'weights': self.weights.to_dict(),
            'confidence': self.confidence,
            'metadata': self.metadata,
            'summary': {
                'n_assets': self.n_assets,
                'gross_exposure': self.gross_exposure,
                'net_exposure': self.net_exposure,
                'effective_n_assets': self.effective_n_assets,
                'max_weight': self.max_weight
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelOutput':
        """Create ModelOutput from dictionary."""
        return cls(
            date=pd.Timestamp(data['date']),
            weights=pd.Series(data['weights']),
            confidence=data['confidence'],
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelOutput':
        """Create ModelOutput from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return (
            f"ModelOutput(date={self.date.strftime('%Y-%m-%d')}, "
            f"n_assets={self.n_assets}, "
            f"confidence={self.confidence:.2f}, "
            f"net_exposure={self.net_exposure:.2%})"
        )


@dataclass
class ModelOutputTimeSeries:
    """
    Collection of ModelOutput objects over time.
    
    Useful for backtesting and ensemble integration.
    """
    
    outputs: List[ModelOutput] = field(default_factory=list)
    model_name: str = "unknown"
    
    def add(self, output: ModelOutput):
        """Add a new output to the series."""
        self.outputs.append(output)
        # Keep sorted by date
        self.outputs.sort(key=lambda x: x.date)
    
    @property
    def dates(self) -> pd.DatetimeIndex:
        """All dates in the series."""
        return pd.DatetimeIndex([o.date for o in self.outputs])
    
    @property
    def weights_df(self) -> pd.DataFrame:
        """All weights as a DataFrame (dates x assets)."""
        if not self.outputs:
            return pd.DataFrame()
        return pd.DataFrame(
            {o.date: o.weights for o in self.outputs}
        ).T.sort_index()
    
    @property
    def confidence_series(self) -> pd.Series:
        """All confidence values as a Series."""
        return pd.Series(
            {o.date: o.confidence for o in self.outputs}
        ).sort_index()
    
    def get_output(self, date: pd.Timestamp) -> Optional[ModelOutput]:
        """Get output for a specific date."""
        for output in self.outputs:
            if output.date == date:
                return output
        return None
    
    def get_latest(self) -> Optional[ModelOutput]:
        """Get the most recent output."""
        if not self.outputs:
            return None
        return self.outputs[-1]
    
    def __len__(self) -> int:
        return len(self.outputs)
    
    def __repr__(self) -> str:
        if not self.outputs:
            return f"ModelOutputTimeSeries(model={self.model_name}, n_outputs=0)"
        return (
            f"ModelOutputTimeSeries(model={self.model_name}, "
            f"n_outputs={len(self.outputs)}, "
            f"date_range={self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')})"
        )
