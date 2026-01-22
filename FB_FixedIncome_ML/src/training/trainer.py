"""
Training pipeline for portfolio optimization models.

Handles walk-forward training, cross-validation, and model selection.
"""

from typing import Optional, List, Dict, Any, Type
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from ..config import Config
from ..data.loader import DataLoader, PortfolioData
from ..data.features import FeatureEngineer, FeatureSet
from ..models.base import BasePortfolioModel
from ..interface.model_output import ModelOutputTimeSeries


class Trainer:
    """
    Training pipeline for portfolio models.
    
    Supports:
    - Single train/test split
    - Walk-forward validation (rolling retraining)
    - Cross-validation for hyperparameter tuning
    """
    
    def __init__(
        self, 
        model: BasePortfolioModel,
        config: Optional[Config] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model instance to train
            config: Configuration object
        """
        self.model = model
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
    
    def train(
        self,
        data: PortfolioData,
        validation_split: float = 0.2,
        target_type: str = 'returns',
        **model_kwargs
    ) -> BasePortfolioModel:
        """
        Train model on portfolio data.
        
        Args:
            data: PortfolioData object
            validation_split: Fraction for validation
            target_type: Target variable type
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model.name} on {data.metadata.get('strategy_name', 'unknown')}")
        print(f"{'='*60}")
        
        # Generate features
        features = self.feature_engineer.generate_features(
            data,
            include_target=True,
            target_type=target_type
        )
        
        # Train model
        self.model.fit(
            features,
            data.etf_order,
            validation_fraction=validation_split,
            **model_kwargs
        )
        
        return self.model
    
    def train_walk_forward(
        self,
        data: PortfolioData,
        initial_train_days: int = 730,
        retrain_frequency: int = 63,
        target_type: str = 'returns',
        **model_kwargs
    ) -> ModelOutputTimeSeries:
        """
        Walk-forward training with periodic retraining.
        
        This simulates realistic model deployment where the model
        is retrained periodically on expanding or rolling data.
        
        Args:
            data: PortfolioData object
            initial_train_days: Initial training period
            retrain_frequency: Days between retraining
            target_type: Target variable type
            **model_kwargs: Additional model parameters
            
        Returns:
            ModelOutputTimeSeries with all predictions
        """
        print(f"\n{'='*60}")
        print(f"Walk-Forward Training: {self.model.name}")
        print(f"Initial training: {initial_train_days} days")
        print(f"Retrain every: {retrain_frequency} days")
        print(f"{'='*60}")
        
        # Generate all features upfront
        all_features = self.feature_engineer.generate_features(
            data,
            include_target=True,
            target_type=target_type
        )
        
        # Get trading dates
        dates = all_features.features.index
        
        # Initial training period
        train_end_idx = initial_train_days
        if train_end_idx >= len(dates):
            raise ValueError(f"Insufficient data for initial training: {len(dates)} < {initial_train_days}")
        
        outputs = ModelOutputTimeSeries(model_name=self.model.name)
        last_train_idx = 0
        
        print(f"\nStarting walk-forward from {dates[train_end_idx].strftime('%Y-%m-%d')}")
        
        for i in range(train_end_idx, len(dates), self.config.REBALANCE_FREQUENCY_DAYS):
            current_date = dates[i]
            
            # Check if we need to retrain
            if i - last_train_idx >= retrain_frequency or last_train_idx == 0:
                print(f"\n   📚 Retraining at {current_date.strftime('%Y-%m-%d')}...")
                
                # Get training data up to current date
                train_features = FeatureSet(
                    features=all_features.features.iloc[:i],
                    feature_names=all_features.feature_names,
                    target=all_features.target.iloc[:i] if all_features.target is not None else None
                )
                
                # Train model
                self.model.fit(
                    train_features,
                    data.etf_order,
                    **model_kwargs
                )
                
                last_train_idx = i
            
            # Generate prediction
            try:
                current_features = all_features.features.iloc[[i]]
                output = self.model.predict(current_features, current_date)
                outputs.add(output)
            except Exception as e:
                print(f"   ⚠️ Prediction failed at {current_date}: {e}")
        
        print(f"\n✅ Walk-forward complete: {len(outputs)} predictions")
        return outputs
    
    def cross_validate(
        self,
        data: PortfolioData,
        n_folds: int = 5,
        target_type: str = 'returns',
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Time-series cross-validation.
        
        Uses expanding window (not k-fold) to respect temporal order.
        
        Args:
            data: PortfolioData object
            n_folds: Number of validation folds
            target_type: Target variable type
            **model_kwargs: Additional model parameters
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        print(f"\n{'='*60}")
        print(f"Cross-Validation: {self.model.name} ({n_folds} folds)")
        print(f"{'='*60}")
        
        # Generate features
        features = self.feature_engineer.generate_features(
            data,
            include_target=True,
            target_type=target_type
        )
        
        X = features.features.values
        y = features.target.values if features.target is not None else None
        
        if y is None:
            raise ValueError("Target required for cross-validation")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n   Fold {fold + 1}/{n_folds}...")
            print(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
            
            # Create fold-specific feature sets
            train_features = FeatureSet(
                features=pd.DataFrame(X[train_idx], columns=features.feature_names.keys()),
                feature_names=features.feature_names,
                target=pd.DataFrame(y[train_idx], columns=data.etf_order)
            )
            
            # Train fresh model instance
            fold_model = type(self.model)(config=self.config, name=f"{self.model.name}_fold{fold}")
            fold_model.fit(train_features, data.etf_order, **model_kwargs)
            
            # Validate
            val_predictions = []
            for idx in val_idx:
                val_features = pd.DataFrame(
                    X[idx:idx+1], 
                    columns=features.feature_names.keys()
                )
                pred = fold_model.predict_raw(val_features, pd.Timestamp.now())
                val_predictions.append(pred)
            
            val_predictions = np.array(val_predictions)
            val_actual = y[val_idx]
            
            # Calculate metrics
            mse = np.mean((val_predictions - val_actual) ** 2)
            mae = np.mean(np.abs(val_predictions - val_actual))
            
            # Correlation of rankings (more relevant for portfolio)
            rank_corrs = []
            for i in range(len(val_idx)):
                from scipy.stats import spearmanr
                corr, _ = spearmanr(val_predictions[i], val_actual[i])
                if not np.isnan(corr):
                    rank_corrs.append(corr)
            
            avg_rank_corr = np.mean(rank_corrs) if rank_corrs else 0.0
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'mse': mse,
                'mae': mae,
                'rank_correlation': avg_rank_corr
            })
            
            print(f"   MSE: {mse:.6f}, MAE: {mae:.6f}, Rank Corr: {avg_rank_corr:.4f}")
        
        # Aggregate results
        results = {
            'n_folds': n_folds,
            'fold_results': fold_results,
            'mean_mse': np.mean([f['mse'] for f in fold_results]),
            'mean_mae': np.mean([f['mae'] for f in fold_results]),
            'mean_rank_correlation': np.mean([f['rank_correlation'] for f in fold_results]),
            'std_mse': np.std([f['mse'] for f in fold_results]),
            'std_rank_correlation': np.std([f['rank_correlation'] for f in fold_results])
        }
        
        print(f"\n{'='*40}")
        print(f"CV Results:")
        print(f"   Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
        print(f"   Mean Rank Corr: {results['mean_rank_correlation']:.4f} ± {results['std_rank_correlation']:.4f}")
        
        return results
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save trained model.
        
        Args:
            path: Save path (default: models/{model_name}_{timestamp}.pkl)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            output_dir = self.config.get_model_output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = output_dir / f"{self.model.name}_{timestamp}.pkl"
        
        self.model.save(path)
        return path


class ModelSelector:
    """
    Model selection via hyperparameter search.
    
    Supports grid search and random search over model hyperparameters.
    """
    
    def __init__(
        self,
        model_class: Type[BasePortfolioModel],
        param_grid: Dict[str, List[Any]],
        config: Optional[Config] = None
    ):
        """
        Initialize model selector.
        
        Args:
            model_class: Model class to tune
            param_grid: Dictionary of parameter names to lists of values
            config: Configuration object
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.config = config or Config()
        self.results: List[Dict[str, Any]] = []
    
    def grid_search(
        self,
        data: PortfolioData,
        n_cv_folds: int = 3,
        metric: str = 'rank_correlation'
    ) -> Dict[str, Any]:
        """
        Grid search over hyperparameters.
        
        Args:
            data: PortfolioData object
            n_cv_folds: Number of CV folds per configuration
            metric: Metric to optimize ('mse', 'mae', 'rank_correlation')
            
        Returns:
            Best parameters and results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = list(product(*param_values))
        
        print(f"\n{'='*60}")
        print(f"Grid Search: {len(all_combinations)} configurations")
        print(f"{'='*60}")
        
        best_score = float('-inf') if metric == 'rank_correlation' else float('inf')
        best_params = None
        
        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))
            print(f"\n[{i+1}/{len(all_combinations)}] Testing: {params}")
            
            # Create model with these parameters
            model = self.model_class(config=self.config)
            trainer = Trainer(model, self.config)
            
            # Cross-validate
            cv_results = trainer.cross_validate(data, n_folds=n_cv_folds, **params)
            
            score = cv_results[f'mean_{metric}']
            
            self.results.append({
                'params': params,
                'score': score,
                'cv_results': cv_results
            })
            
            # Check if best
            if metric == 'rank_correlation':
                is_better = score > best_score
            else:
                is_better = score < best_score
            
            if is_better:
                best_score = score
                best_params = params
        
        print(f"\n{'='*40}")
        print(f"Best Parameters: {best_params}")
        print(f"Best {metric}: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }
