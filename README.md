# FolioBeyond Fixed Income Portfolio Optimization

A comprehensive portfolio optimization system for fixed income ETF portfolios using mean-variance optimization with stress test constraints.

## Features

- **Mean-Variance Optimization**: Scipy-based portfolio optimization with multiple constraints
- **Stress Test Constraints**: Maximum 15% portfolio loss constraint
- **Multiple Volatility Targets**: Low (2.5%), Moderate (3.5%), High (4.5%), RRS (15%)
- **Rolling Window Analysis**: 730-day lookback period for optimization
- **Comprehensive Performance Analytics**: Returns, volatility, Sharpe ratio, drawdown, VaR, CVaR
- **Debug Mode**: Safe testing mode that protects production data

## Architecture

The system is organized into 8 main sections:

1. **Configuration & Setup** - Centralized parameters and constants
2. **Portfolio Class** - Data management and path handling
3. **Data Loading** - Standardized data loading and validation
4. **Optimization Engine** - Complete optimization with all constraints
5. **Performance Analysis** - Risk and return calculations
6. **Workflow Management** - High-level orchestration
7. **Utilities** - Supporting functions
8. **Testing & Execution** - Demo and test functions

## Quick Start

```python
from folio_beyond_fixed_income import run_quick_test, run_portfolio_strategy

# Quick test with limited data
result = run_quick_test(strategy_name="Moderate Int.20121101.current", num_windows=10)

# Full strategy run
result = run_portfolio_strategy(
    strategy_name="Moderate Int.20121101.current",
    debug_mode=True  # Writes to local debug folder
)
```

## Main Entry Points

- `run_quick_test()` - Quick testing with limited windows
- `run_portfolio_strategy()` - Complete single strategy workflow
- `run_multiple_strategies()` - Batch processing multiple strategies

## Demo Functions

- `demo_quick_test()` - 1-2 minute verification
- `demo_full_strategy()` - 5-15 minute full demo
- `demo_multiple_strategies()` - Compare multiple strategies

## Configuration

Key parameters in `Config` class:
- Trading days per year: 253
- Lookback window: 730 days (~2 years)
- Max weight per asset: 30%
- Stress test max loss: 15%
- Rebalance threshold: 15%

## Portfolio Strategies

The system supports 12 portfolio strategies:
- Low/Moderate/High × Long/Int/Short (9 strategies)
- Sixty Forty
- VariableVol
- RRS (with stress test constraints)

## Requirements

```
numpy
pandas
scipy
matplotlib
pandas_market_calendars
openpyxl
```

## Output

All outputs are written to debug folders by default for safe testing:
- CSV exports: weights, returns, performance summary
- Excel exports with multiple sheets
- Formatted performance reports

## License

Proprietary - FolioBeyond
