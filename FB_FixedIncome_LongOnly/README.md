# FolioBeyond Fixed Income Portfolio Optimization System

A sophisticated fixed income portfolio optimization system supporting both long-only and long/short strategies with leverage. The system implements mean-variance optimization with comprehensive risk constraints, point-in-time stress testing, and transaction cost awareness.

## Key Features

### Portfolio Optimization Modes

| Mode | Description | Gross Exposure | Net Exposure |
|------|-------------|----------------|--------------|
| **Long-Only** | Traditional long positions only | 100% | 100% |
| **Long/Short** | Allows short positions with leverage | Up to 180% | 100% (target) |

### Core Capabilities

- **Mean-Variance Optimization**: SLSQP-based optimizer with theoretically grounded risk aversion
- **Multiple Volatility Targets**: Low (2.5%), Moderate (3.5%), High (4.5%), RRS (15%)
- **Point-in-Time Stress Testing**: Eliminates look-ahead bias using rolling maximum drawdown
- **Transaction Cost Awareness**: TC penalty in optimizer prevents signal churning
- **Ledoit-Wolf Shrinkage**: Automatic covariance matrix regularization for numerical stability
- **Momentum Signals**: Applied to expected returns (not covariance) for proper signal implementation
- **Comprehensive Risk Analytics**: VaR, CVaR, drawdown analysis, risk decomposition

### Long/Short Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Long Weight | 50% | Maximum long position per asset |
| Max Short Weight | -30% | Maximum short position per asset |
| Gross Exposure Limit | 180% | Maximum sum of absolute weights |
| Net Exposure Target | 100% | Target sum of weights (long bias) |
| Leverage Cost | 2.00% annual | Cost of financing leverage |
| Borrow Cost | 0.50% annual | Cost to borrow for shorts |

## Architecture

```
folio_beyond_fixed_income.py
├── SECTION 1: Configuration & Imports
│   ├── Config class - Long-only parameters
│   └── LongShortConfig class - Long/short parameters
├── SECTION 2: Core Portfolio Class
│   └── Portfolio class - Data management, path handling
├── SECTION 3: Data Loading & Validation
│   ├── load_portfolio_data() - Main data loader
│   ├── load_time_series() - Generic time series loader
│   └── _calculate_pit_stress_values() - Point-in-time stress testing
├── SECTION 4: Optimization Engine
│   ├── optimize_portfolio() - Main optimization loop
│   └── _optimize_single_date() - Single-date optimizer with all constraints
├── SECTION 5: Performance Analysis
│   ├── calculate_portfolio_performance() - Full performance analysis
│   ├── calculate_long_short_metrics() - Long/short specific metrics
│   └── compare_with_benchmark() - Benchmark comparison
├── SECTION 6: Workflow Management
│   ├── run_portfolio_strategy() - Single strategy workflow
│   ├── run_multiple_strategies() - Batch processing
│   └── run_quick_test() - Quick testing
├── SECTION 7: Utility Functions
│   ├── validate_portfolio_data() - Data quality checks
│   ├── calculate_portfolio_risk_decomposition() - Risk attribution
│   └── generate_performance_report() - Formatted reports
└── SECTION 8: Testing & Execution
    ├── demo_quick_test() - Quick verification
    ├── demo_full_strategy() - Full long-only demo
    ├── demo_long_short_strategy() - Long/short demo
    └── demo_multiple_strategies() - Strategy comparison
```

## Installation

### Requirements

```bash
pip install numpy pandas scipy matplotlib pandas-market-calendars openpyxl python-dotenv
```

### Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your data paths:
```bash
# Base path to SharedFolioBeyond data directory
FOLIOBEYOND_BASE_PATH=/path/to/your/SharedFolioBeyond
```

## Quick Start

### Basic Usage

```python
from folio_beyond_fixed_income import *

# Run quick verification test (1-2 minutes)
demo_quick_test()

# Run full long-only strategy
demo_full_strategy()

# Run long/short strategy with leverage
demo_long_short_strategy()
```

### API Usage

```python
# Single strategy - Long Only
result = run_portfolio_strategy(
    strategy_name="Moderate Int.20121101.current",
    max_windows=100,
    debug_mode=True,
    enable_long_short=False
)

# Single strategy - Long/Short with Leverage
result = run_portfolio_strategy(
    strategy_name="Moderate Int.20121101.current",
    max_windows=100,
    debug_mode=True,
    enable_long_short=True
)

# Multiple strategies comparison
results = run_multiple_strategies(
    strategy_names=[
        "Moderate Int.20121101.current",
        "High Int.20121101.current",
        "Low Int.20121101.current"
    ],
    max_windows=50,
    debug_mode=True
)
```

## Portfolio Strategies

### Predefined Strategies

| Strategy | Volatility Target | Description |
|----------|-------------------|-------------|
| Low Long/Int/Short | 2.5% | Conservative, low volatility |
| Moderate Long/Int/Short | 3.5% | Balanced risk/return |
| High Long/Int/Short | 4.5% | Higher volatility, higher return potential |
| Sixty Forty | 2.5% | Classic 60/40 style allocation |
| VariableVol | 3.5% | Dynamic volatility scaling |
| RRS | 15% | Risk Reduction Strategy |

### Date Range Formats

Strategies follow the naming convention: `{StrategyName}.{StartDate}.{EndDate}`

Examples:
- `Moderate Int.20121101.current` - From Nov 2012 to present
- `High Long.20070501.20100831` - Specific date range

## Optimization Constraints

### Long-Only Mode

| Constraint | Description |
|------------|-------------|
| Budget | Weights sum to 1.0 |
| Non-negativity | All weights >= 0 |
| Max Weight | No single asset > 30% |
| Volatility | Portfolio volatility <= target |
| Stress Test | Portfolio stress loss <= 15% |

### Long/Short Mode

| Constraint | Description |
|------------|-------------|
| Net Exposure | Weights sum to 1.0 (±10% tolerance) |
| Gross Exposure | Sum of absolute weights <= 180% |
| Long Limits | Individual long positions <= 50% |
| Short Limits | Individual short positions >= -30% |
| Volatility | Portfolio volatility <= target |
| Stress Test | Portfolio stress loss <= 15% |

## Debug Mode

All operations run in debug mode by default for safety:

```python
# Debug mode (default) - outputs to local temp folder
portfolio = Portfolio(debug_mode=True)

# Production mode - writes to production paths
portfolio = Portfolio(debug_mode=False)
```

Debug mode features:
- All outputs written to `folio_beyond_debug_{YYYYMMDD}/` folder
- Production data is never overwritten
- Easy cleanup with `portfolio.cleanup_debug_files()`
- List outputs with `portfolio.list_debug_outputs()`

## Output Files

### Generated Files

| File | Description |
|------|-------------|
| `{strategy}_weights.csv` | Daily portfolio weights |
| `{strategy}_returns.csv` | Daily portfolio returns |
| `{strategy}_summary.csv` | Performance metrics summary |

### Performance Metrics

- **Return Metrics**: Annualized return, total return, cumulative return
- **Risk Metrics**: Annualized volatility, VaR (95%), CVaR, maximum drawdown
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio
- **Concentration**: Effective number of assets, HHI, max concentration

### Long/Short Specific Metrics

- Average gross exposure
- Average net exposure
- Average long/short exposure
- Average leverage ratio
- Exposure statistics (min, max, std)

## Technical Details

### Point-in-Time Stress Testing

The system calculates stress test values dynamically using only historical data available at each optimization date, eliminating look-ahead bias:

```python
# Stress values calculated from rolling maximum drawdown
# over 2-year lookback period, with -5% minimum floor
stress_values = _calculate_pit_stress_values(
    portfolio,
    current_date,
    lookback_days=730,
    min_stress_floor=-5.0
)
```

### Transaction Cost Awareness

The optimizer includes a transaction cost penalty to reduce unnecessary turnover:

```
Maximize: w'μ - TC × |w_t - w_{t-1}| - (λ/2) × w'Σw
```

Configuration:
```python
Config.TC_BPS_PER_TRADE = 10  # 10 basis points per trade
Config.TC_INCLUDE_IN_OPTIMIZER = True  # Enable/disable TC penalty
```

### Covariance Matrix Regularization

Automatic Ledoit-Wolf shrinkage is applied when the covariance matrix is nearly singular:

```python
# Shrinkage intensity adapts based on minimum eigenvalue
shrinkage_intensity = min(0.3, max(0.05, -log10(min_eigenval) / 20))
cov_shrunk = (1 - intensity) * cov + intensity * mu * I
```

## Configuration Reference

### Config Class (Long-Only)

```python
TRADING_DAYS_PER_YEAR = 253
LOOKBACK_WINDOW_DAYS = 730  # ~2 years
MAX_WEIGHT_PER_ASSET = 0.30
STRESS_TEST_MAX_LOSS = 15.0  # Maximum stress loss %
TC_BPS_PER_TRADE = 10  # Transaction cost in basis points
```

### LongShortConfig Class

```python
MAX_LONG_WEIGHT = 0.50
MAX_SHORT_WEIGHT = -0.30
GROSS_EXPOSURE_LIMIT = 1.80
NET_EXPOSURE_TARGET = 1.00
NET_EXPOSURE_TOLERANCE = 0.10
LEVERAGE_COST_BPS = 200  # Annual
BORROW_COST_BPS = 50  # Annual
```

## Data Requirements

### Required Input Files

| File | Description |
|------|-------------|
| `{strategy}_Daily_Returns.csv` | Daily ETF returns (scaled by 100) |
| `{strategy}_Daily_Adjusted_YTM.csv` | Projected returns (yield-to-maturity) |
| `ETF Stress Test Final (values)...xlsx` | ETF order and stress test data |
| `Daily_Scaling.csv` | Volatility scaling ratios |
| `AGG_ETF_1Yr_Vol_Monthly.csv` | Variable volatility data |
| `AGG.csv` | Benchmark data |

### Data Format

- Returns and projected returns are scaled by 100 (e.g., 0.05% = 5.0)
- Date index in YYYY-MM-DD format
- Column names must match ETF order

## Error Handling

### Constraint Failure Protocol

When optimization fails to satisfy all constraints:

1. **Primary**: Return previous period's weights (no forced trades)
2. **Secondary**: Log specific constraint violations for analysis
3. **Never**: Drop constraints silently to force a solution

### Diagnostic Output

```python
# Constraint violation diagnostics
_diagnose_constraint_violations(
    weights, covariance_matrix, vol_target,
    stress_values, stress_limit, enable_long_short
)
```

## License

Proprietary - FolioBeyond

## Version History

- **v2.0** - Long/short with leverage, PIT stress testing, TC awareness
- **v1.0** - Initial release, long-only optimization
