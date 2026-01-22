# FolioBeyond Fixed Income ML Portfolio Optimization

A machine learning approach to fixed income portfolio optimization, designed as a modular component of the FolioBeyond ensemble system.

## Architecture Overview

This repository implements ML-based portfolio optimization using either:
- **Tree-based models**: Random Forest, XGBoost, LightGBM
- **Neural networks**: Feedforward, LSTM, Transformer-based

```
FB_FixedIncome_ML/
├── src/
│   ├── config.py              # Configuration & constants
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── features.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── base.py            # Base model interface (ModelOutput contract)
│   │   ├── tree_models.py     # Tree-based approaches
│   │   └── neural_models.py   # Neural network approaches
│   ├── training/
│   │   ├── trainer.py         # Training pipeline
│   │   └── evaluation.py      # Backtesting & performance metrics
│   └── interface/
│       └── model_output.py    # Standardized output for ensemble integration
├── notebooks/                  # Experimentation notebooks
├── scripts/
│   └── train.py               # CLI entry point
└── models/                     # Saved model checkpoints
```

## Ensemble Integration

This module produces standardized `ModelOutput` objects that can be consumed by the ensemble layer:

```python
from src.interface.model_output import ModelOutput

# All models produce this standardized output
output = ModelOutput(
    date=pd.Timestamp('2024-01-15'),
    weights=pd.Series({'TLT': 0.3, 'IEF': 0.25, ...}),
    confidence=0.85,
    metadata={'model': 'xgboost', 'features_used': [...]}
)
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Environment Configuration

```bash
cp .env.example .env
# Edit .env with your data paths
```

## Quick Start

### Training a Model

```python
from src.config import Config
from src.data.loader import DataLoader
from src.models.tree_models import XGBoostPortfolioModel
from src.training.trainer import Trainer

# Load data
config = Config()
loader = DataLoader(config)
data = loader.load_portfolio_data("Moderate Int.20121101.current")

# Initialize model
model = XGBoostPortfolioModel(config)

# Train
trainer = Trainer(model, config)
trainer.train(data, validation_split=0.2)

# Generate predictions
output = model.predict(data.get_latest_features())
print(f"Weights: {output.weights}")
print(f"Confidence: {output.confidence}")
```

### Command Line

```bash
# Train XGBoost model
python scripts/train.py --model xgboost --strategy "Moderate Int.20121101.current"

# Train neural network
python scripts/train.py --model lstm --strategy "Moderate Int.20121101.current"

# Evaluate against MVO baseline
python scripts/train.py --model xgboost --evaluate --compare-mvo
```

## Model Approaches

### Tree-Based Models

| Model | Description | Best For |
|-------|-------------|----------|
| Random Forest | Ensemble of decision trees | Baseline, interpretability |
| XGBoost | Gradient boosted trees | Performance, feature importance |
| LightGBM | Fast gradient boosting | Large datasets, speed |

### Neural Network Models

| Model | Description | Best For |
|-------|-------------|----------|
| Feedforward | Dense layers | Simple patterns |
| LSTM | Recurrent architecture | Temporal dependencies |
| Transformer | Attention-based | Complex temporal patterns |

## Feature Engineering

The system generates features from:

1. **Price-based**: Returns, volatility, momentum indicators
2. **Yield-based**: YTM, yield curve features, spreads
3. **Technical**: Moving averages, RSI, Bollinger bands
4. **Cross-sectional**: Relative value, z-scores across assets
5. **Macro**: (Optional) VIX, rates, credit spreads

## Training Methodology

### Problem Formulation

Portfolio optimization as ML can be framed as:

1. **Regression**: Predict optimal weights directly
2. **Classification**: Predict overweight/underweight signals
3. **Reinforcement Learning**: Learn allocation policy via rewards

This implementation supports regression and classification approaches.

### Walk-Forward Validation

```
|-------- Training --------|--- Val ---|--- Test ---|
                           t-252       t-63         t

# Rolling window approach to prevent look-ahead bias
for each rebalance_date:
    train on [t-730, t-63]
    validate on [t-63, t]
    predict weights for t+1
```

## Performance Metrics

- **Return metrics**: Annualized return, Sharpe ratio, Sortino ratio
- **Risk metrics**: Volatility, VaR, CVaR, max drawdown
- **ML metrics**: MSE (for weight prediction), accuracy (for signals)
- **Comparison**: Alpha vs MVO baseline, tracking error

## Configuration

Key parameters in `src/config.py`:

```python
# Training
LOOKBACK_WINDOW = 730        # ~2 years of history
VALIDATION_WINDOW = 63       # ~3 months
REBALANCE_FREQUENCY = 21     # Monthly

# Model
VOLATILITY_TARGET = 0.035    # 3.5% annualized
MAX_WEIGHT = 0.30            # 30% max per asset

# Features
FEATURE_LOOKBACKS = [5, 21, 63, 126, 252]  # Multiple timeframes
```

## Related Repositories

- `FB_FixedIncome_LongOnly`: MVO-based optimization (production)
- `FB_Ensemble` (future): Ensemble combination layer

## License

Proprietary - FolioBeyond
