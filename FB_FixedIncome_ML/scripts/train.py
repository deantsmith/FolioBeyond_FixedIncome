#!/usr/bin/env python3
"""
CLI entry point for training portfolio optimization models.

Usage:
    python scripts/train.py --model xgboost --strategy "Moderate Int.20121101.current"
    python scripts/train.py --model lstm --strategy "Moderate Int.20121101.current" --walk-forward
    python scripts/train.py --model random_forest --evaluate
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.loader import DataLoader
from src.training.trainer import Trainer
from src.training.evaluation import Evaluator
from src.models import (
    RandomForestPortfolioModel,
    XGBoostPortfolioModel,
    LightGBMPortfolioModel,
    FeedforwardPortfolioModel,
    LSTMPortfolioModel
)


MODEL_CLASSES = {
    'random_forest': RandomForestPortfolioModel,
    'xgboost': XGBoostPortfolioModel,
    'lightgbm': LightGBMPortfolioModel,
    'feedforward': FeedforwardPortfolioModel,
    'lstm': LSTMPortfolioModel
}


def main():
    parser = argparse.ArgumentParser(
        description='Train ML portfolio optimization models'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(MODEL_CLASSES.keys()),
        default='xgboost',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='Moderate Int.20121101.current',
        help='Strategy name (e.g., "Moderate Int.20121101.current")'
    )
    
    parser.add_argument(
        '--walk-forward', '-wf',
        action='store_true',
        help='Use walk-forward training (periodic retraining)'
    )
    
    parser.add_argument(
        '--cross-validate', '-cv',
        action='store_true',
        help='Run cross-validation'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Evaluate model after training'
    )
    
    parser.add_argument(
        '--save', 
        action='store_true',
        help='Save trained model'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for saved model'
    )
    
    parser.add_argument(
        '--target-type',
        type=str,
        choices=['returns', 'sharpe'],
        default='returns',
        help='Target variable type'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split fraction'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FolioBeyond ML Portfolio Optimization")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Target: {args.target_type}")
    
    # Initialize
    config = Config()
    loader = DataLoader(config)
    
    # Load data
    print(f"\n📊 Loading data...")
    try:
        data = loader.load_portfolio_data(args.strategy)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nMake sure your FOLIOBEYOND_BASE_PATH is set correctly.")
        print("You can set it in a .env file or as an environment variable.")
        sys.exit(1)
    
    # Create model
    model_class = MODEL_CLASSES[args.model]
    model = model_class(config=config)
    trainer = Trainer(model, config)
    
    # Train
    if args.cross_validate:
        # Cross-validation mode
        results = trainer.cross_validate(
            data,
            n_folds=args.cv_folds,
            target_type=args.target_type
        )
        print(f"\n✅ Cross-validation complete")
        
    elif args.walk_forward:
        # Walk-forward mode
        outputs = trainer.train_walk_forward(
            data,
            target_type=args.target_type
        )
        
        if args.evaluate:
            print(f"\n📈 Evaluating...")
            evaluator = Evaluator(config)
            result = evaluator.evaluate(outputs, data)
            print(result.summary())
            
    else:
        # Standard training
        trainer.train(
            data,
            validation_split=args.validation_split,
            target_type=args.target_type
        )
        
        if args.evaluate:
            print(f"\n📈 Running backtest evaluation...")
            # Generate features for backtest
            from src.data.features import FeatureEngineer
            fe = FeatureEngineer(config)
            features = fe.generate_features(data, include_target=False)
            
            # Backtest
            outputs = model.backtest(
                features, 
                data.returns,
                rebalance_frequency=config.REBALANCE_FREQUENCY_DAYS
            )
            
            evaluator = Evaluator(config)
            result = evaluator.evaluate(outputs, data)
            print(result.summary())
    
    # Save model
    if args.save:
        output_path = Path(args.output_dir) if args.output_dir else None
        save_path = trainer.save_model(output_path)
        print(f"\n💾 Model saved to: {save_path}")
    
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
