# ML Features Extension - Implementation Blueprint

## Overview
Transform the portfolio optimization system to use machine learning for enhanced return/risk predictions while maintaining the mean-variance optimization framework (Option A: ML predictions → traditional optimization).

---

## Phase 1: Data Infrastructure (Week 1-2)

### 1.1 Bloomberg Data Integration

**Objective**: Create robust Bloomberg API integration for comprehensive data access

**Components**:
```python
class BloombergDataConnector:
    """
    Bloomberg Terminal API integration for FI data
    Uses blpapi (Bloomberg Python API)
    """
    # Bond ETF data
    - Price history (intraday & daily)
    - Volume & liquidity metrics
    - Bid-ask spreads

    # Fixed Income specific
    - Yield curves (Treasury, corporate, muni)
    - Credit spreads (IG, HY by rating)
    - Option-adjusted spreads (OAS)
    - Duration & convexity
    - Sector spreads

    # Macro & rates
    - Fed funds rate & expectations
    - SOFR & term rates
    - Inflation expectations (TIPS breakevens)
    - Economic indicators (GDP, employment, CPI, etc.)
    - Central bank policy indicators

    # Market microstructure
    - Trading volumes by sector
    - New issuance calendar
    - Redemption flows
    - Fund flows (ETF creation/redemption)
```

**Files to Create**:
- `data/bloomberg_connector.py` - Bloomberg API wrapper
- `data/data_cache.py` - Local caching to minimize API calls
- `data/data_validator.py` - Data quality checks

**Bloomberg Tickers for Fixed Income**:
```python
CORE_TICKERS = {
    # ETFs (existing)
    'AGG', 'BND', 'VCIT', 'VGIT', 'VTEB', ...

    # Treasury Curve Points
    'USGG2YR Index',   # 2Y Treasury
    'USGG5YR Index',   # 5Y Treasury
    'USGG10YR Index',  # 10Y Treasury
    'USGG30YR Index',  # 30Y Treasury

    # Credit Spreads
    'LUACOAS Index',   # IG Corporate OAS
    'LF98OAS Index',   # HY Corporate OAS

    # Volatility
    'MOVE Index',      # Bond market volatility (MOVE)
    'VIX Index',       # Equity vol (for correlation)

    # Macro
    'FDTR Index',      # Fed Funds Target Rate
    'USYC2Y10 Index',  # 2s10s curve slope
    'USGGBE10 Index',  # 10Y inflation breakeven
}
```

### 1.2 Feature Store Architecture

**Objective**: Versioned, validated feature storage

```python
class FeatureStore:
    """
    Centralized feature storage with versioning
    - Stores raw and engineered features
    - Tracks feature definitions and transformations
    - Enables reproducibility and backtesting
    """

    def __init__(self, storage_path: str):
        self.raw_data_path = storage_path / 'raw'
        self.features_path = storage_path / 'features'
        self.metadata_path = storage_path / 'metadata'

    def save_feature_set(self, name: str, features: pd.DataFrame,
                        version: str, metadata: dict):
        """Save feature set with version and metadata"""

    def load_feature_set(self, name: str, version: str = 'latest'):
        """Load specific feature set version"""

    def list_versions(self, name: str):
        """List all versions of a feature set"""
```

---

## Phase 2: Feature Engineering Library (Week 3-4)

### 2.1 Technical Features (Price-based)

```python
class TechnicalFeatureEngineer:
    """Generate technical indicators from price/return data"""

    def calculate_momentum_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Multiple horizon momentum:
        - 5, 10, 20, 60, 120, 252 day returns
        - Acceleration (change in momentum)
        - Cross-sectional momentum ranking
        """

    def calculate_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility measures:
        - Realized volatility (multiple windows: 20, 60, 252 days)
        - EWMA volatility (decay factors: 0.94, 0.97)
        - Volatility-of-volatility
        - Up vs down volatility
        - Garman-Klass volatility (uses OHLC)
        """

    def calculate_trend_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Trend indicators:
        - Moving averages (5, 10, 20, 50, 120, 200 day)
        - MA crossovers (short/long ratios)
        - MACD indicators
        - ADX (trend strength)
        - Parabolic SAR
        """

    def calculate_mean_reversion_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Mean reversion signals:
        - Bollinger Bands (position & width)
        - RSI (14, 30, 60 day)
        - Z-scores vs moving averages
        - Distance from highs/lows
        """

    def calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Liquidity measures:
        - Volume trends
        - Amihud illiquidity measure
        - Bid-ask spreads
        - Roll's implicit spread
        - Turnover ratios
        """
```

**Output**: ~25 technical features per ETF

### 2.2 Macro & Rates Features

```python
class MacroFeatureEngineer:
    """Generate macro and fixed income specific features"""

    def calculate_curve_features(self, yields: pd.DataFrame) -> pd.DataFrame:
        """
        Yield curve features:
        - Curve level (average of key points)
        - Curve slope (10Y - 2Y, 30Y - 2Y)
        - Curve curvature/butterfly (2*10Y - 5Y - 30Y)
        - Rate of change in curve shape
        - PCA factors (level, slope, curvature)
        """

    def calculate_spread_features(self, spreads: pd.DataFrame) -> pd.DataFrame:
        """
        Credit spread features:
        - IG spread level & changes
        - HY spread level & changes
        - IG-HY spread differential
        - Spread percentiles (historical)
        - Spread momentum
        - Cross-sector spread relationships
        """

    def calculate_policy_features(self, rates: pd.DataFrame) -> pd.DataFrame:
        """
        Monetary policy indicators:
        - Fed funds rate level & trajectory
        - Policy rate expectations (futures)
        - Deviation from Taylor rule
        - Days since last Fed meeting
        - Surprise component (actual vs expected)
        - Forward guidance sentiment
        """

    def calculate_inflation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inflation indicators:
        - Breakeven inflation rates (5Y, 10Y)
        - Change in breakevens
        - Real vs nominal yield spread
        - CPI/PPI trends
        - Inflation surprise index
        """

    def calculate_economic_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Economic indicators:
        - GDP growth (current & forward)
        - Employment indicators (payrolls, unemployment)
        - Manufacturing/services PMI
        - Consumer/business sentiment
        - Leading economic indicators
        - Surprise indices (Citi Economic Surprise)
        """
```

**Output**: ~30 macro features

### 2.3 Cross-Sectional & Relative Value Features

```python
class CrossSectionalFeatureEngineer:
    """Generate relative value and cross-sectional features"""

    def calculate_relative_performance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional rankings:
        - Return rank (vs other ETFs)
        - Sharpe ratio rank
        - Relative to benchmark
        - Sector leadership
        """

    def calculate_correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Correlation dynamics:
        - Rolling correlations (pairwise)
        - Average correlation to portfolio
        - Correlation stability
        - Correlation regime changes
        - PCA loadings
        """

    def calculate_dispersion_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Dispersion measures:
        - Cross-sectional return dispersion
        - Volatility dispersion
        - Range (max - min returns)
        - Concentration (HHI of returns)
        """
```

**Output**: ~15 cross-sectional features

### 2.4 Temporal & Calendar Features

```python
class TemporalFeatureEngineer:
    """Time-based and calendar features"""

    def calculate_calendar_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calendar effects:
        - Month of year
        - Day of week
        - Days to month-end
        - Days to quarter-end
        - Fed meeting proximity
        - FOMC week indicator
        - Treasury auction calendar
        - Options expiration proximity
        - Holiday effects
        """

    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Market regime indicators:
        - Volatility regime (high/low)
        - Trend regime (up/down/sideways)
        - Credit regime (tightening/widening)
        - Liquidity regime
        """
```

**Output**: ~10 temporal features

### 2.5 Feature Summary

**Total Feature Count**: ~80-100 features
- 25 technical features
- 30 macro/rates features
- 15 cross-sectional features
- 10 temporal features

---

## Phase 3: ML Model Development (Week 5-6)

### 3.1 Model Architecture

```python
class MLPortfolioPredictor:
    """
    Ensemble ML model for return and risk prediction
    Option A: Predict inputs to mean-variance optimization
    """

    def __init__(self, config: MLConfig):
        # Return prediction models
        self.return_model = ReturnPredictionEnsemble()

        # Risk prediction models
        self.volatility_model = VolatilityPredictionModel()
        self.correlation_model = CorrelationPredictionModel()

        # Feature importance tracking
        self.feature_importance = {}

    class ReturnPredictionEnsemble:
        """
        Ensemble for expected return prediction
        Combines multiple algorithms
        """
        def __init__(self):
            self.models = {
                'xgboost': xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8
                ),
                'lightgbm': lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20
                ),
                'ridge': Ridge(alpha=1.0)
            }
            self.ensemble_weights = None  # Learned via validation

        def fit(self, X: pd.DataFrame, y: pd.Series,
                sample_weight: Optional[np.ndarray] = None):
            """Train all models and learn ensemble weights"""

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """Generate ensemble prediction"""

    class VolatilityPredictionModel:
        """
        Volatility forecasting (GARCH family + ML)
        """
        def __init__(self):
            # GARCH(1,1) baseline
            self.garch_models = {}

            # ML augmentation
            self.ml_model = xgb.XGBRegressor()

        def fit(self, returns: pd.DataFrame, features: pd.DataFrame):
            """Fit GARCH + ML hybrid model"""

        def predict(self, features: pd.DataFrame) -> pd.Series:
            """Predict next-period volatility"""

    class CorrelationPredictionModel:
        """
        Dynamic correlation forecasting
        """
        def __init__(self):
            # DCC-GARCH for correlations
            # Factor models for dimension reduction
            # ML for regime-dependent correlations
            pass
```

### 3.2 Training Pipeline

```python
class MLTrainingPipeline:
    """Walk-forward training and validation"""

    def __init__(self,
                 feature_engineer: FeatureEngineer,
                 model: MLPortfolioPredictor,
                 config: MLConfig):
        self.feature_engineer = feature_engineer
        self.model = model
        self.config = config

    def run_walk_forward_validation(self,
                                   start_date: str,
                                   end_date: str,
                                   train_window: int = 1260,  # 5 years
                                   test_window: int = 252):   # 1 year
        """
        Walk-forward cross-validation
        - Train on rolling window
        - Test on next period
        - Track predictions vs actuals
        """

    def calculate_model_metrics(self, predictions: pd.DataFrame,
                               actuals: pd.DataFrame) -> dict:
        """
        Evaluation metrics:
        - Return prediction: R², MSE, direction accuracy, IC
        - Volatility prediction: RMSE, MAPE
        - Correlation prediction: Frobenius norm
        """

    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Feature importance analysis:
        - SHAP values
        - Permutation importance
        - Grouped feature importance
        """

    def hyperparameter_tuning(self, param_grid: dict):
        """Bayesian optimization for hyperparameters"""
```

### 3.3 Target Variables

```python
# Forward-looking targets for training

# Return targets (multiple horizons)
forward_returns = {
    '1d': prices.pct_change().shift(-1),   # Next day
    '5d': prices.pct_change(5).shift(-5),  # Next week
    '20d': prices.pct_change(20).shift(-20), # Next month
}

# Volatility targets
realized_volatility = returns.rolling(20).std().shift(-20)

# Correlation targets
rolling_correlation = returns.rolling(60).corr()  # Rolling correlation matrix
```

---

## Phase 4: Integration with Optimization (Week 7)

### 4.1 ML-Enhanced Optimization

```python
class MLEnhancedOptimizer:
    """
    Combines ML predictions with mean-variance optimization
    """

    def __init__(self,
                 ml_predictor: MLPortfolioPredictor,
                 portfolio: Portfolio,
                 config: Config):
        self.ml_predictor = ml_predictor
        self.portfolio = portfolio
        self.config = config

    def optimize_with_ml_inputs(self,
                                date: pd.Timestamp,
                                features: pd.DataFrame) -> Dict[str, Any]:
        """
        1. Generate ML predictions
        2. Use predictions as inputs to mean-variance optimization
        3. Apply all constraints (stress test, etc.)
        """

        # Generate ML predictions
        predicted_returns = self.ml_predictor.predict_returns(features)
        predicted_volatility = self.ml_predictor.predict_volatility(features)
        predicted_correlation = self.ml_predictor.predict_correlation(features)

        # Construct covariance matrix
        vol_matrix = np.diag(predicted_volatility)
        cov_matrix = vol_matrix @ predicted_correlation @ vol_matrix

        # Traditional mean-variance optimization with ML inputs
        weights = self._optimize_mean_variance(
            expected_returns=predicted_returns,
            covariance_matrix=cov_matrix,
            constraints=self._build_constraints()
        )

        return {
            'weights': weights,
            'ml_predictions': {
                'returns': predicted_returns,
                'volatility': predicted_volatility,
                'correlation': predicted_correlation
            },
            'optimization_result': result
        }

    def blend_ml_and_traditional(self,
                                 ml_inputs: dict,
                                 traditional_inputs: dict,
                                 blend_weight: float = 0.5) -> dict:
        """
        Blend ML predictions with traditional estimates
        - blend_weight: 0 = pure traditional, 1 = pure ML
        """

        blended_returns = (
            blend_weight * ml_inputs['returns'] +
            (1 - blend_weight) * traditional_inputs['returns']
        )

        # Similar for covariance

        return {'returns': blended_returns, 'covariance': blended_cov}
```

### 4.2 Model Monitoring & Retraining

```python
class ModelMonitor:
    """Monitor ML model performance in production"""

    def track_prediction_accuracy(self,
                                 predictions: pd.DataFrame,
                                 actuals: pd.DataFrame):
        """Track rolling prediction metrics"""

    def detect_model_drift(self) -> bool:
        """Detect when model performance degrades"""

    def trigger_retraining(self, threshold: float = 0.1):
        """Automatically retrain if drift detected"""
```

---

## Phase 5: Testing & Validation (Week 8)

### 5.1 Backtesting Framework

```python
def backtest_ml_strategy(
    start_date: str,
    end_date: str,
    use_ml: bool = True,
    blend_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Compare ML-enhanced vs traditional optimization

    Returns:
        Performance comparison including:
        - Returns (annualized, cumulative)
        - Sharpe ratios
        - Drawdowns
        - Turnover
        - Win rates
    """
```

### 5.2 Performance Metrics

- Information Coefficient (IC): Correlation between predictions and actuals
- Rank IC: Spearman correlation
- Direction accuracy: % correct direction
- Hit ratio: % predictions within error bands
- Sharpe ratio improvement vs baseline

---

## Implementation Files Structure

```
FB_FixedIncome_LongOnly/
├── folio_beyond_fixed_income.py        # Main file (existing)
├── ml/
│   ├── __init__.py
│   ├── data/
│   │   ├── bloomberg_connector.py      # Bloomberg API integration
│   │   ├── data_cache.py               # Caching layer
│   │   └── data_validator.py           # Data quality checks
│   ├── features/
│   │   ├── feature_store.py            # Feature storage & versioning
│   │   ├── technical_features.py       # Technical indicators
│   │   ├── macro_features.py           # Macro & rates features
│   │   ├── cross_sectional_features.py # Relative value features
│   │   └── temporal_features.py        # Time-based features
│   ├── models/
│   │   ├── return_predictor.py         # Return prediction models
│   │   ├── volatility_predictor.py     # Volatility models
│   │   ├── correlation_predictor.py    # Correlation models
│   │   └── ensemble.py                 # Model ensembling
│   ├── training/
│   │   ├── pipeline.py                 # Training pipeline
│   │   ├── validation.py               # Walk-forward validation
│   │   └── hyperparameter_tuning.py    # Optimization
│   ├── integration/
│   │   ├── ml_optimizer.py             # ML-enhanced optimization
│   │   ├── blending.py                 # ML/traditional blending
│   │   └── monitoring.py               # Model monitoring
│   └── utils/
│       ├── metrics.py                  # Evaluation metrics
│       └── visualization.py            # Plotting & analysis
├── tests/
│   └── ml/                             # Unit tests
└── config/
    └── ml_config.yaml                  # ML configuration
```

---

## Dependencies

```
# Add to requirements.txt
xgboost>=2.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
shap>=0.42.0              # Feature importance
optuna>=3.3.0             # Hyperparameter tuning
blpapi                    # Bloomberg API (requires license)
pyarrow                   # Fast data storage
joblib                    # Model persistence
```

---

## Success Criteria

### Minimum Viable Product (MVP):
1. ✓ Bloomberg data integration functional
2. ✓ 50+ engineered features
3. ✓ Return prediction R² > 0.10 (out-of-sample)
4. ✓ Volatility prediction RMSE < historical baseline
5. ✓ ML-optimized portfolio Sharpe ≥ traditional Sharpe

### Stretch Goals:
- Information Coefficient (IC) > 0.05 (monthly returns)
- Top quartile selection accuracy > 55%
- Sharpe ratio improvement > 10% vs baseline

---

## Risk Mitigation

1. **Overfitting**: Walk-forward validation, regularization, ensemble methods
2. **Data leakage**: Strict temporal splits, feature lag verification
3. **Model instability**: Ensemble methods, blend with traditional estimates
4. **Computational cost**: Feature caching, model compression, parallel processing
5. **Bloomberg dependencies**: Fallback to free data sources for testing

---

## Next Steps

1. Review and approve blueprint
2. Set up Bloomberg API credentials and test connection
3. Begin Phase 1: Data infrastructure implementation
4. Parallel work: Start feature engineering while data pipeline builds

---

**Questions for Discussion:**
1. Bloomberg Terminal access confirmed?
2. Preferred ML framework (XGBoost vs LightGBM vs both)?
3. GPU availability for neural network experiments?
4. Preferred retraining frequency (monthly, quarterly)?
5. Model interpretability requirements (SHAP analysis)?
