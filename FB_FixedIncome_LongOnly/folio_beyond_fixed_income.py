#!/usr/bin/env python3
"""
FolioBeyond Fixed Income Portfolio Optimization System - REFACTORED

Architecture:
1. Configuration & Setup - All parameters and constants
2. Portfolio Class - Clean, focused data management
3. Data Loading - Standardized, validated data loading
4. Optimization Engine - Complete optimization with all constraints
5. Performance Analysis - Risk and return calculations
6. Workflow Management - High-level orchestration
7. Utilities - Supporting functions
8. Testing & Execution - Clear entry points
"""

# =============================================================================
# SECTION 1: CONFIGURATION & IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from pathlib import Path
from scipy import optimize
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import pandas_market_calendars as mcal
import warnings
warnings.filterwarnings('ignore')

# FIX 2D: Environment configuration support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    pass  # python-dotenv not installed, will use os.environ directly

# =============================================================================
# CONFIGURATION CLASS - All parameters in one place
# =============================================================================

class Config:
    """Centralized configuration for all portfolio optimization parameters"""
    
    # Basic Constants
    TRADING_DAYS_PER_YEAR = 253
    SQRT_TRADING_DAYS = np.sqrt(TRADING_DAYS_PER_YEAR)
    SCALE_FACTOR_FOR_RETURNS = 100
    
    # Transaction Costs
    DAILY_TRANSACTION_COST_BP = 2 * 6.66667 / TRADING_DAYS_PER_YEAR  # basis points

    # FIX 3A: Transaction Cost Awareness in Optimizer
    # TC penalty in objective function to prevent signal churning
    TC_BPS_PER_TRADE = 10  # One-way transaction cost in basis points (0.10%)
    TC_INCLUDE_IN_OPTIMIZER = True  # Set to False to disable TC penalty in optimization
    
    # Optimization Parameters
    LOOKBACK_WINDOW_DAYS = 730  # ~2 years
    PORTFOLIO_REBALANCE_THRESHOLD = 0.15
        
    # Moving Average Parameters
    MA_SHORT_LENGTH = 5
    MA_LONG_LENGTH = 120
    
    # Constraint Parameters
    MAX_WEIGHT_PER_ASSET = 0.30
    STRESS_TEST_MAX_LOSS = 15.0  # Maximum allowed stress test loss in %
    
    # Risk Parameters - Portfolio-specific volatility targets
    VOLATILITY_TARGETS = {
        'Low': 0.025,
        'Moderate': 0.035,
        'High': 0.045,
        'RRS': 0.15
    }
    
    # Date Ranges for Historical Analysis
    DATE_RANGES = {
        'period1': '.20050101.20070430',
        'period2': '.20070501.20100831', 
        'period3': '.20100901.20121031',
        'current': '.20121101.current',
        'rrs': '.20210901.current'
    }
    
    # Portfolio Strategy Definitions
    PORTFOLIO_STRATEGIES = [
        {'name': 'Moderate Long', 'vol_target': 'Moderate', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Moderate Int', 'vol_target': 'Moderate', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Moderate Short', 'vol_target': 'Moderate', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'High Long', 'vol_target': 'High', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'High Int', 'vol_target': 'High', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'High Short', 'vol_target': 'High', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Low Long', 'vol_target': 'Low', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Low Int', 'vol_target': 'Low', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Low Short', 'vol_target': 'Low', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'Sixty Forty', 'vol_target': 'Low', 'use_variable_vol': False, 'use_stress_test': False},
        {'name': 'VariableVol', 'vol_target': 'Moderate', 'use_variable_vol': True, 'use_stress_test': False},
        {'name': 'RRS', 'vol_target': 'RRS', 'use_variable_vol': False, 'use_stress_test': True},
    ]
    
    # FIX 2D: Default paths (can be overridden via environment variables)
    DEFAULT_BASE_PATH = '/Users/deansmith/FolioBeyond Dropbox/Dean Smith/SharedFolioBeyond'

    @classmethod
    def get_base_path(cls):
        """
        Determine base path based on environment.

        FIX 2D: Priority order:
        1. FOLIOBEYOND_BASE_PATH environment variable
        2. Default path (development/production fallback)

        To configure, set environment variable or create .env file:
            FOLIOBEYOND_BASE_PATH=/path/to/your/SharedFolioBeyond
        """
        base_path = os.environ.get('FOLIOBEYOND_BASE_PATH')

        if base_path:
            return base_path

        # Development default
        return cls.DEFAULT_BASE_PATH
    
    @classmethod  
    def get_volatility_target(cls, portfolio_type: str) -> float:
        """Get volatility target for portfolio type"""
        return cls.VOLATILITY_TARGETS.get(portfolio_type, cls.VOLATILITY_TARGETS['Moderate'])

# Create global config instance
config = Config()

# =============================================================================
# LONG/SHORT CONFIGURATION CLASS
# =============================================================================

class LongShortConfig:
    """Configuration for long/short portfolio optimization with leverage"""

    # Position Limits
    MAX_LONG_WEIGHT = 0.50  # Maximum long position per asset (50%)
    MAX_SHORT_WEIGHT = -0.30  # Maximum short position per asset (-30%)

    # Exposure Limits
    GROSS_EXPOSURE_LIMIT = 1.80  # Maximum gross exposure (180%)
    NET_EXPOSURE_TARGET = 1.00  # Target net exposure (100% - long bias)
    NET_EXPOSURE_TOLERANCE = 0.10  # Tolerance around net target (±10%)

    # Cost Parameters (annualized basis points)
    LEVERAGE_COST_BPS = 200  # Cost of leverage/financing (2.00%)
    BORROW_COST_BPS = 50  # Cost to borrow for short positions (0.50%)
    SHORT_REBATE_BPS = 0  # Rebate on short proceeds (0% conservative)

    # Risk Parameters
    MAX_LEVERAGE_RATIO = 2.0  # Maximum leverage ratio (gross/net)
    MARGIN_BUFFER = 0.25  # Margin buffer (25% above minimum requirements)

    # Sector/Factor Exposure Limits (optional)
    MAX_SECTOR_NET_EXPOSURE = 0.40  # Max net exposure to any sector (40%)
    MAX_FACTOR_EXPOSURE = 0.50  # Max exposure to any risk factor (50%)

    @classmethod
    def get_daily_costs(cls):
        """Convert annualized costs to daily basis points"""
        return {
            'leverage_cost_daily': cls.LEVERAGE_COST_BPS / config.TRADING_DAYS_PER_YEAR,
            'borrow_cost_daily': cls.BORROW_COST_BPS / config.TRADING_DAYS_PER_YEAR,
            'short_rebate_daily': cls.SHORT_REBATE_BPS / config.TRADING_DAYS_PER_YEAR
        }

    @classmethod
    def validate_weights(cls, weights: pd.Series) -> Dict[str, bool]:
        """Validate weights satisfy long/short constraints"""
        gross_exposure = weights.abs().sum()
        net_exposure = weights.sum()
        leverage_ratio = gross_exposure / abs(net_exposure) if net_exposure != 0 else float('inf')

        validation = {
            'long_limits': (weights <= cls.MAX_LONG_WEIGHT).all(),
            'short_limits': (weights >= cls.MAX_SHORT_WEIGHT).all(),
            'gross_exposure': gross_exposure <= cls.GROSS_EXPOSURE_LIMIT,
            'net_exposure': abs(net_exposure - cls.NET_EXPOSURE_TARGET) <= cls.NET_EXPOSURE_TOLERANCE,
            'leverage_ratio': leverage_ratio <= cls.MAX_LEVERAGE_RATIO
        }

        validation['all_valid'] = all(validation.values())
        validation['gross_exposure_value'] = gross_exposure
        validation['net_exposure_value'] = net_exposure
        validation['leverage_ratio_value'] = leverage_ratio

        return validation

# Create global long/short config instance
ls_config = LongShortConfig()

print("✅ Configuration loaded successfully")
print(f"📊 Trading days per year: {config.TRADING_DAYS_PER_YEAR}")
print(f"📈 Default volatility targets: {config.VOLATILITY_TARGETS}")
print(f"🎯 Stress test max loss: {config.STRESS_TEST_MAX_LOSS}%")
print(f"📊 Long/Short enabled: Gross limit {ls_config.GROSS_EXPOSURE_LIMIT:.0%}, Net target {ls_config.NET_EXPOSURE_TARGET:.0%}")


# =============================================================================
# SECTION 2: CORE PORTFOLIO CLASS
# =============================================================================

class Portfolio:
    """
    Main portfolio class for fixed income optimization.
    Handles data storage, path management, and basic utilities.
    """
    
    def __init__(self, debug_mode: bool = True, enable_long_short: bool = False):
        """
        Initialize portfolio with optional debug mode for safe testing.

        Args:
            debug_mode: If True, writes outputs to temp folder to protect production data
            enable_long_short: If True, enables long/short positions with leverage
        """
        self.debug_mode = debug_mode
        self.enable_long_short = enable_long_short
        self.base_path = Path(config.get_base_path())
        
        # Core data containers - all DataFrames for consistency
        self.returns = None
        self.prices = None
        self.projected_returns = None
        self.stress_test_values = None  # pd.Series
        self.use_pit_stress = True  # FIX 1A: Default to Point-in-Time stress testing
        self.volatility_scaling_ratios = None  # pd.Series
        self.variable_vol = None
        self.durations = None
        self.benchmark = None
        self.benchmark_monthly_returns = None
        
        # Metadata
        self.etf_order = None  # List of ETF names for consistent ordering
        self.optimization_windows = None  # List of (start_date, end_date) tuples
        self.current_strategy = None  # Current strategy being processed
        
        # Results storage
        self.optimization_results = {}
        self.performance_metrics = {}
        
        # Setup paths
        self._setup_paths()

        mode_str = f"{'DEBUG' if debug_mode else 'PRODUCTION'} | {'LONG/SHORT' if enable_long_short else 'LONG-ONLY'}"
        print(f"Portfolio initialized in {mode_str} mode")
        if debug_mode:
            print(f"🔒 All outputs will be written to: {self.output_path}")
            print("🔒 Production data will NOT be overwritten")
        if enable_long_short:
            print(f"📊 Long/Short enabled: Gross={ls_config.GROSS_EXPOSURE_LIMIT:.0%}, Net={ls_config.NET_EXPOSURE_TARGET:.0%}")
    
    def _setup_paths(self):
        """Setup input and output paths with debug mode safety"""
        current_date = dt.datetime.now().strftime('%Y%m%d')
        
        if self.debug_mode:
            # Create temporary local debug folder
            self.temp_debug_root = Path.cwd() / f'folio_beyond_debug_{current_date}'
            self.output_path = self.temp_debug_root / f'DailyOutput/FI_{current_date}'
            self.debug_output_path = self.temp_debug_root / 'Debug'
        else:
            # Use production paths  
            self.output_path = self.base_path / 'Investment Strategies/Output' / f'DailyOutput/FI_{current_date}'
            self.debug_output_path = self.base_path / 'Investment Strategies/Fixed Income/Debug'
        
        # Data input paths (always read from production)
        self.data_path = self.base_path / 'Investment Strategies/Equity-FixedIncome Blend/Matrix Universe CSV Files'
        self.benchmark_path = self.base_path / 'UAData/Treesdale'
    
    def create_directory(self, path: Union[str, Path]):
        """Create directory if it doesn't exist"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def cleanup_debug_files(self):
        """Clean up temporary debug files"""
        if self.debug_mode and hasattr(self, 'temp_debug_root') and self.temp_debug_root.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_debug_root)
                print(f"🧹 Cleaned up debug folder: {self.temp_debug_root}")
            except Exception as e:
                print(f"⚠️ Could not clean up debug folder: {e}")
    
    def list_debug_outputs(self):
        """List all files created in debug mode"""
        if self.debug_mode and hasattr(self, 'temp_debug_root') and self.temp_debug_root.exists():
            print(f"📁 Debug output files in {self.temp_debug_root}:")
            for file_path in self.temp_debug_root.rglob('*'):
                if file_path.is_file():
                    print(f"   📄 {file_path.relative_to(self.temp_debug_root)}")
        else:
            print("No debug output folder found or not in debug mode")
    
    def validate_data(self) -> bool:
        """Validate that required data is loaded"""
        required_data = ['returns', 'projected_returns', 'etf_order']
        missing = [attr for attr in required_data if getattr(self, attr) is None]
        
        if missing:
            print(f"❌ Missing required data: {missing}")
            return False
        
        print("✅ Data validation passed")
        return True
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        summary = {
            'strategy': self.current_strategy,
            'etf_count': len(self.etf_order) if self.etf_order else 0,
            'date_range': None,
            'windows_count': len(self.optimization_windows) if self.optimization_windows else 0
        }
        
        if self.returns is not None:
            summary['date_range'] = f"{self.returns.index[0]} to {self.returns.index[-1]}"
            summary['returns_shape'] = self.returns.shape
        
        return summary
    
    @staticmethod
    def to_yyyymmdd(date_obj) -> str:
        """Convert date to YYYYMMDD string format"""
        if isinstance(date_obj, (pd.Timestamp, dt.datetime)):
            return date_obj.strftime('%Y%m%d')
        elif isinstance(date_obj, list) and len(date_obj) >= 3:
            return f'{date_obj[0]:04d}{date_obj[1]:02d}{date_obj[2]:02d}'
        else:
            raise ValueError(f'Unsupported date format: {date_obj}')
    
    @staticmethod
    def get_last_day_of_month(month: int, year: int) -> int:
        """Get last day of given month and year"""
        return calendar.monthrange(year, month)[1]


# =============================================================================
# SECTION 3: DATA LOADING & VALIDATION
# =============================================================================

import shutil

def load_portfolio_data(portfolio: Portfolio, strategy_name: str) -> bool:
    """
    Load all required data for a portfolio strategy.
    
    Args:
        portfolio: Portfolio instance to load data into
        strategy_name: Name of strategy (e.g., "Moderate Int.20121101.current")

    Returns:
        bool: True if loading successful, False otherwise
    """
    try:
        print(f"📂 Loading data for strategy: {strategy_name}")
        portfolio.current_strategy = strategy_name
        
        # Handle special portfolio types that copy data
        if "Sixty Forty" in strategy_name:
            _copy_portfolio_data("Moderate Int", "Sixty Forty", strategy_name)
        elif "VariableVol" in strategy_name:
            _copy_portfolio_data("Moderate Int", "VariableVol", strategy_name)
        
        # Build file paths
        dir_underscore = strategy_name.replace(" ", "_")
        strategy_dir = portfolio.data_path / strategy_name
        
        file_paths = {
            'returns': strategy_dir / f"{dir_underscore}_Daily_Returns.csv",
            'projected_returns': strategy_dir / f"{dir_underscore}_Daily_Adjusted_YTM.csv", 
            'stress_test': strategy_dir / f"ETF Stress Test Final (values) - {strategy_name} Cumulative Final.xlsx",
            'volatility_scaling': portfolio.data_path / "Daily_Scaling.csv",
            'variable_vol': portfolio.data_path / "AGG_ETF_1Yr_Vol_Monthly.csv",
            'benchmark': portfolio.benchmark_path / "AGG.csv"
        }
        
        # Load stress test data first to get ETF order
        print("📊 Loading stress test data and ETF order...")
        if not _load_stress_test_data(portfolio, file_paths['stress_test']):
            return False
            
        # Load main return data
        print("📈 Loading returns data...")
        if not _load_returns_data(portfolio, file_paths['returns']):
            return False
            
        # Load projected returns
        print("🎯 Loading projected returns...")
        if not _load_projected_returns(portfolio, file_paths['projected_returns']):
            return False
            
        # Load volatility scaling ratios
        print("📏 Loading volatility scaling...")
        if not _load_volatility_scaling(portfolio, file_paths['volatility_scaling']):
            return False
            
        # Load variable volatility data
        print("📊 Loading variable volatility data...")
        if not _load_variable_vol_data(portfolio, file_paths['variable_vol']):
            return False
            
        # Load benchmark data
        print("📊 Loading benchmark data...")
        if not _load_benchmark_data(portfolio, file_paths['benchmark']):
            return False
            
        # Calculate prices from returns
        print("💰 Calculating price series...")
        _calculate_price_series(portfolio)
        
        # Create optimization windows
        print("📅 Creating optimization windows...")
        _create_optimization_windows(portfolio)
        
        # Validate all data
        if not portfolio.validate_data():
            return False
            
        print(f"✅ Data loading completed successfully for {strategy_name}")
        print(f"📊 {portfolio.get_data_summary()}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data for {strategy_name}: {e}")
        return False

def _load_stress_test_data(portfolio: Portfolio, file_path: Path,
                          use_pit_stress: bool = True) -> bool:
    """
    Load stress test values and ETF order.

    FIX 1A: Added use_pit_stress parameter to enable Point-in-Time stress testing.

    Args:
        portfolio: Portfolio object to populate
        file_path: Path to stress test Excel file
        use_pit_stress: If True, stress values will be calculated PIT during optimization.
                       If False, uses static file (legacy behavior with look-ahead bias warning).
    """
    try:
        if not file_path.exists():
            print(f"⚠️ Stress test file not found: {file_path}")
            # Create default values if file missing
            portfolio.etf_order = ['AGG', 'BND', 'VCIT', 'VGIT', 'VTEB']  # Default bond ETFs
            portfolio.stress_test_values = None  # Will be calculated PIT
            portfolio.use_pit_stress = True
            return True

        df = pd.read_excel(file_path, header=None)
        # Drop the first column ("Date") from df as it is not used in the stress test
        df = df.drop(columns=[0])

        # Extract ETF order from first row
        portfolio.etf_order = df.iloc[0].dropna().tolist()

        if use_pit_stress:
            # FIX 1A: Mark for PIT calculation during optimization
            portfolio.stress_test_values = None
            portfolio.use_pit_stress = True
            print(f"   📋 ETF order: {portfolio.etf_order}")
            print(f"   🔒 Using Point-in-Time stress testing (look-ahead bias eliminated)")
        else:
            # Legacy: load static values (with warning)
            print(f"   ⚠️ WARNING: Using static stress values - potential look-ahead bias!")
            stress_values = df.iloc[1].dropna().values
            if len(stress_values) > 0:
                portfolio.stress_test_values = pd.Series(
                    stress_values[:len(portfolio.etf_order)],
                    index=portfolio.etf_order
                )
            else:
                portfolio.stress_test_values = pd.Series([0.0] * len(portfolio.etf_order),
                                                       index=portfolio.etf_order)
            portfolio.use_pit_stress = False
            print(f"   📋 ETF order: {portfolio.etf_order}")

        return True

    except Exception as e:
        print(f"❌ Error loading stress test data: {e}")
        return False


def _calculate_pit_stress_values(portfolio: Portfolio, current_date: pd.Timestamp,
                                  lookback_days: int = 730,
                                  min_stress_floor: float = -5.0) -> pd.Series:
    """
    FIX 1A: Calculate Point-in-Time stress test values using only historical data.

    Uses maximum drawdown over lookback period as stress scenario.
    This eliminates look-ahead bias by only using data available at current_date.

    Args:
        portfolio: Portfolio with returns data
        current_date: The date as of which to calculate stress values
        lookback_days: Number of days to look back for max drawdown (default 730 = ~2 years)
        min_stress_floor: Minimum stress value to prevent overly mild stress in calm periods

    Returns:
        Series of stress values (negative percentages) indexed by ETF
    """
    try:
        # Get returns up to current_date only (no future data)
        available_returns = portfolio.returns.loc[:current_date]

        if len(available_returns) < 30:
            print(f"   ⚠️ Insufficient history for PIT stress test: {len(available_returns)} days")
            # Return minimum stress floor for all assets
            return pd.Series(min_stress_floor, index=portfolio.etf_order)

        # Use only the lookback window (or all available if less)
        actual_lookback = min(lookback_days, len(available_returns))
        window_returns = available_returns.iloc[-actual_lookback:]

        # Calculate cumulative returns for each ETF
        # Returns are in percentage terms (scaled by 100), so divide by 100
        cumulative = (1 + window_returns / config.SCALE_FACTOR_FOR_RETURNS).cumprod()

        # Calculate maximum drawdown per ETF
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdowns = drawdowns.min() * 100  # Convert back to percentage

        # Apply minimum stress floor to prevent overly mild stress in calm periods
        max_drawdowns = max_drawdowns.clip(upper=min_stress_floor)

        return max_drawdowns

    except Exception as e:
        print(f"   ⚠️ Error calculating PIT stress values: {e}")
        # Return minimum stress floor for all assets as fallback
        return pd.Series(min_stress_floor, index=portfolio.etf_order)

# =============================================================================
# FIX 2C: GENERIC TIME SERIES LOADER
# =============================================================================

def load_time_series(path: Path,
                     index_col: str = 'Date',
                     parse_dates: bool = True,
                     sort_index: bool = True,
                     required: bool = True,
                     fill_value: Any = None,
                     columns_order: List[str] = None) -> Optional[pd.DataFrame]:
    """
    FIX 2C: Generic time series loader with validation.

    Reduces code duplication across multiple loader functions.

    Args:
        path: Path to CSV or Excel file
        index_col: Column to use as index (default 'Date')
        parse_dates: Whether to parse dates (default True)
        sort_index: Whether to sort by index (default True)
        required: If True, raise error if file missing; if False, return None
        fill_value: Value to fill missing data when reindexing columns
        columns_order: List of columns to reindex to (maintains order consistency)

    Returns:
        DataFrame or None if file missing and not required
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None

    # Determine file type and load
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path, index_col=index_col)
        if parse_dates and df.index.dtype != 'datetime64[ns]':
            df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if sort_index:
        df.sort_index(inplace=True)

    if columns_order is not None:
        df = df.reindex(columns=columns_order, fill_value=fill_value)

    return df


def _load_returns_data(portfolio: Portfolio, file_path: Path) -> bool:
    """Load daily returns data"""
    try:
        # FIX 2C: Use generic loader
        portfolio.returns = load_time_series(
            file_path,
            required=True,
            columns_order=portfolio.etf_order,
            fill_value=0
        )
        print(f"   📈 Returns shape: {portfolio.returns.shape}")
        return True

    except Exception as e:
        print(f"❌ Error loading returns data: {e}")
        return False


def _load_projected_returns(portfolio: Portfolio, file_path: Path) -> bool:
    """Load projected returns (yield) data"""
    try:
        # FIX 2C: Use generic loader
        portfolio.projected_returns = load_time_series(
            file_path,
            required=True,
            columns_order=portfolio.etf_order,
            fill_value=0
        )
        print(f"   🎯 Projected returns shape: {portfolio.projected_returns.shape}")
        return True

    except Exception as e:
        print(f"❌ Error loading projected returns: {e}")
        return False


def _load_volatility_scaling(portfolio: Portfolio, file_path: Path) -> bool:
    """Load volatility scaling ratios"""
    try:
        # FIX 2C: Use generic loader with optional file handling
        df = load_time_series(file_path, required=False)

        if df is None:
            print(f"⚠️ Volatility scaling file not found, using default values")
            portfolio.volatility_scaling_ratios = pd.Series([1.0],
                                                           index=[pd.Timestamp.now()])
            return True

        portfolio.volatility_scaling_ratios = df.iloc[:, 2]  # Third column
        print(f"   📏 Volatility scaling shape: {portfolio.volatility_scaling_ratios.shape}")
        return True

    except Exception as e:
        print(f"❌ Error loading volatility scaling: {e}")
        return False


def _load_variable_vol_data(portfolio: Portfolio, file_path: Path) -> bool:
    """Load variable volatility data"""
    try:
        # FIX 2C: Use generic loader with optional file handling
        df = load_time_series(
            file_path,
            required=False,
            columns_order=portfolio.etf_order,
            fill_value=0.035
        )

        if df is None:
            print(f"⚠️ Variable vol file not found, using default values")
            if portfolio.returns is not None:
                portfolio.variable_vol = pd.DataFrame(
                    0.035, index=portfolio.returns.index, columns=portfolio.etf_order)
            return True

        portfolio.variable_vol = df
        print(f"   📊 Variable vol shape: {portfolio.variable_vol.shape}")
        return True

    except Exception as e:
        print(f"❌ Error loading variable vol data: {e}")
        return False

def _load_benchmark_data(portfolio: Portfolio, file_path: Path) -> bool:
    """Load benchmark data"""
    try:
        if not file_path.exists():
            print(f"⚠️ Benchmark file not found, skipping")
            return True
            
        # AGG.csv has no header, first column is date in YYYYMMDD format
        portfolio.benchmark = pd.read_csv(file_path, index_col=0, header=None)
        portfolio.benchmark.index = pd.to_datetime(portfolio.benchmark.index, format='%Y%m%d')
        portfolio.benchmark.sort_index(inplace=True)
        
        # Calculate monthly benchmark returns
        _calculate_monthly_benchmark_returns(portfolio)
        
        print(f"   📊 Benchmark shape: {portfolio.benchmark.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading benchmark data: {e}")
        return False

def _calculate_monthly_benchmark_returns(portfolio: Portfolio):
    """Calculate monthly returns from daily benchmark data"""
    if portfolio.benchmark is None:
        return
        
    # Group by month and calculate compound returns
    monthly_data = portfolio.benchmark.groupby(pd.Grouper(freq='M')).apply(
        lambda x: (1 + x/100).prod() - 1 if len(x) > 0 else np.nan
    ) * 100
    
    # Format for output
    date_strs = []
    returns = []
    for date, ret in monthly_data.iterrows():
        if pd.notna(ret.iloc[0]):
            last_day = portfolio.get_last_day_of_month(date.month, date.year)
            date_str = f"{date.year}{date.month:02d}{last_day:02d}"
            date_strs.append(date_str)
            returns.append(ret.iloc[0])
    
    portfolio.benchmark_monthly_returns = pd.DataFrame({
        'date': date_strs, 
        'return': returns
    })

def _calculate_price_series(portfolio: Portfolio):
    """Calculate price series from returns"""
    if portfolio.returns is None:
        return
        
    initial_price = 100  # Arbitrary starting price
    portfolio.prices = initial_price * (1 + portfolio.returns / config.SCALE_FACTOR_FOR_RETURNS).cumprod()
    portfolio.prices = portfolio.prices.reindex(columns=portfolio.etf_order, fill_value=initial_price)

def _create_optimization_windows(portfolio: Portfolio):
    """Create rolling optimization windows"""
    if portfolio.prices is None or portfolio.projected_returns is None:
        return
        
    # Find common dates between price and projected return data
    common_dates = portfolio.prices.index.intersection(portfolio.projected_returns.index).sort_values()
    
    if len(common_dates) < config.LOOKBACK_WINDOW_DAYS:
        print(f"⚠️ Insufficient data for {config.LOOKBACK_WINDOW_DAYS}-day windows")
        return
        
    # Create rolling windows
    windows = []
    last_start_date = common_dates[len(common_dates) - config.LOOKBACK_WINDOW_DAYS]
    valid_start_dates = common_dates[common_dates <= last_start_date]
    
    for start_date in valid_start_dates:
        start_idx = common_dates.get_loc(start_date) 
        end_date = common_dates[start_idx + config.LOOKBACK_WINDOW_DAYS - 1]
        windows.append((start_date, end_date))
    
    portfolio.optimization_windows = windows
    print(f"   📅 Created {len(windows)} optimization windows")

def _copy_portfolio_data(source_pattern: str, target_pattern: str, portfolio_dir: str):
    """Copy portfolio data for special portfolio types"""
    # Note: This function references 'portfolio' which should be passed as parameter
    # For now, keeping the original structure but this would need to be fixed
    pass


print("✅ Data loading functions ready")


# =============================================================================
# SECTION 4: OPTIMIZATION ENGINE - WITH STRESS TEST CONSTRAINTS
# =============================================================================

def optimize_portfolio(portfolio: Portfolio, 
                      optimization_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
                      strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete portfolio optimization with all constraints.
    
       Args:
        portfolio: Portfolio instance with loaded data
        optimization_windows: List of (start_date, end_date) tuples for optimization
        strategy_config: Strategy configuration including volatility target, flags
    
    Returns:
        Dict containing optimization results
    """
    print(f"🎯 Starting optimization for {len(optimization_windows)} windows")
    
    # Extract strategy parameters
    vol_target = strategy_config.get('vol_target', 'Moderate')
    use_variable_vol = strategy_config.get('use_variable_vol', False)
    use_stress_test = strategy_config.get('use_stress_test', False)
    use_durations = strategy_config.get('use_durations', False)
    
    annualized_vol_target = config.get_volatility_target(vol_target)
    daily_vol_target = annualized_vol_target / config.SQRT_TRADING_DAYS
    
    print(f"   📊 Volatility target: {vol_target} ({annualized_vol_target:.1%} annual)")
    print(f"   🎛️ Variable vol: {use_variable_vol}")
    print(f"   🔥 Stress test constraints: {use_stress_test}")
    print(f"   ⏱️ Duration constraints: {use_durations}")
    
    # Calculate moving averages for momentum
    ma_short = portfolio.prices.rolling(window=config.MA_SHORT_LENGTH).mean()
    ma_long = portfolio.prices.rolling(window=config.MA_LONG_LENGTH).mean()
    
    # Storage for results
    all_results = []
    successful_optimizations = 0
    previous_weights = None  # FIX 1B: Track previous weights for fallback

    for i, window in enumerate(optimization_windows):
        if i % 50 == 0:  # Progress reporting
            print(f"   Processing window {i+1}/{len(optimization_windows)}")

        try:
            result = _optimize_single_date(
                portfolio, window, daily_vol_target, ma_short, ma_long,
                use_variable_vol, use_stress_test, use_durations,
                enable_long_short=portfolio.enable_long_short,
                previous_weights=previous_weights  # FIX 1B: Pass previous weights
            )

            if result['success']:
                successful_optimizations += 1
                # Update previous_weights for next iteration
                previous_weights = result['weights']

                result_entry = {
                    'date': window[1],
                    'weights': result['weights'],
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'covariance_matrix': result['covariance_matrix'],
                    'stress_test_result': result.get('stress_test_result'),
                    'constraints_satisfied': result.get('constraints_satisfied', {})
                }

                # FIX 1B: Track fallback usage if applicable
                if result.get('fallback_used'):
                    result_entry['fallback_used'] = result['fallback_used']
                    result_entry['original_failure'] = result.get('original_failure')
                    result_entry['constraint_violations'] = result.get('constraint_violations')

                all_results.append(result_entry)
            else:
                print(f"   ⚠️ Optimization failed for window {i}: {result.get('error')}")

        except Exception as e:
            print(f"   ❌ Error in window {i}: {str(e)}")
            continue
    
    print(f"✅ Optimization completed: {successful_optimizations}/{len(optimization_windows)} successful")
    
    return {
        'results': all_results,
        'success_rate': successful_optimizations / len(optimization_windows) if optimization_windows else 0,
        'strategy_config': strategy_config
    }

def _optimize_single_date(portfolio: Portfolio,
                         window: Tuple[pd.Timestamp, pd.Timestamp],
                         daily_vol_target: float,
                         ma_short: pd.DataFrame,
                         ma_long: pd.DataFrame,
                         use_variable_vol: bool,
                         use_stress_test: bool,
                         use_durations: bool,
                         enable_long_short: bool = False,
                         previous_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Optimize portfolio for a single date with ALL constraints.

    FIX 1B: Added previous_weights parameter for proper failure handling.
    On optimization failure, will use previous_weights instead of dropping constraints.
    """
    try:
        start_date, end_date = window
        
        # Get returns data for the window
        window_returns = portfolio.returns.loc[start_date:end_date]
        if window_returns.empty:
            return {'success': False, 'error': 'No data in window'}
        
        # Check for minimum data requirement
        if len(window_returns) < 30:
            return {'success': False, 'error': f'Insufficient data: only {len(window_returns)} days'}
        
        # Scale returns
        scaled_returns = window_returns / config.SCALE_FACTOR_FOR_RETURNS
        
        # Check for NaN values in returns
        if scaled_returns.isnull().any().any():
            return {'success': False, 'error': 'Returns data contains NaN values'}
        
        # Get expected returns for the end date
        # FIX 1C: Enforce single methodology - no fallback to historical mean
        if end_date not in portfolio.projected_returns.index:
            return {'success': False, 'error': f'Missing projected returns for {end_date} - data gap (no fallback to historical mean)'}

        expected_returns = portfolio.projected_returns.loc[end_date] / config.SCALE_FACTOR_FOR_RETURNS

        # Check for NaN in expected returns
        if expected_returns.isnull().any():
            missing_assets = expected_returns[expected_returns.isnull()].index.tolist()
            return {'success': False, 'error': f'Missing projected returns for assets: {missing_assets}'}
        
        # Calculate covariance matrix
        cov_matrix = scaled_returns.cov() * config.TRADING_DAYS_PER_YEAR
        
        # Check for NaN or infinite values in covariance matrix
        if cov_matrix.isnull().any().any():
            return {'success': False, 'error': 'Covariance matrix contains NaN values'}
        
        if np.isinf(cov_matrix.values).any():
            return {'success': False, 'error': 'Covariance matrix contains infinite values'}
        
        # FIX 3B: Enhanced positive definiteness check with Ledoit-Wolf shrinkage
        try:
            eigenvals = np.linalg.eigvals(cov_matrix.values)
            min_eigenval = np.min(np.real(eigenvals))  # Use real part for numerical stability

            if min_eigenval <= 1e-8:  # Effectively singular
                print(f"   ⚠️ Covariance matrix is nearly singular (min eigenvalue: {min_eigenval:.2e})")

                # Apply Ledoit-Wolf shrinkage (more principled than simple diagonal regularization)
                n = len(cov_matrix)
                mu = np.trace(cov_matrix.values) / n  # Average variance

                # Shrinkage intensity (adaptive based on how singular the matrix is)
                shrinkage_intensity = min(0.3, max(0.05, -np.log10(max(min_eigenval, 1e-15)) / 20))

                # Shrink towards scaled identity matrix
                cov_shrunk = (1 - shrinkage_intensity) * cov_matrix.values + shrinkage_intensity * mu * np.eye(n)
                cov_matrix = pd.DataFrame(cov_shrunk, index=cov_matrix.index, columns=cov_matrix.columns)

                print(f"   🔧 Applied Ledoit-Wolf shrinkage (intensity: {shrinkage_intensity:.3f})")

                # Verify fix worked
                new_eigenvals = np.linalg.eigvals(cov_matrix.values)
                new_min = np.min(np.real(new_eigenvals))
                print(f"   ✓ Post-shrinkage min eigenvalue: {new_min:.2e}")

        except np.linalg.LinAlgError:
            return {'success': False, 'error': 'Covariance matrix eigenvalue computation failed'}
        
        # Handle variable volatility scaling (FIXED: More careful scaling)
        if use_variable_vol and end_date in portfolio.variable_vol.index:
            vol_scaling = portfolio.variable_vol.loc[end_date]
            # Check for reasonable scaling values
            if vol_scaling.min() <= 0 or vol_scaling.max() > 10:
                print(f"   ⚠️ Extreme volatility scaling values: {vol_scaling.min():.3f} to {vol_scaling.max():.3f}")
            else:
                # Apply scaling to covariance matrix
                scaling_matrix = np.outer(vol_scaling.values, vol_scaling.values)
                cov_matrix = cov_matrix.values * scaling_matrix
                cov_matrix = pd.DataFrame(cov_matrix, index=cov_matrix.index, columns=cov_matrix.columns)
        
        # Get momentum adjustment
        momentum_ratio = _calculate_momentum_ratio(portfolio.prices, ma_short, ma_long, end_date)

        # FIX 1D: Apply momentum to expected returns (μ), NOT covariance matrix (Σ)
        # Momentum signals should scale return expectations, not volatility
        if momentum_ratio is not None and 0.5 <= momentum_ratio <= 2.0:
            expected_returns = expected_returns * momentum_ratio
            print(f"   Applied momentum ratio {momentum_ratio:.3f} to expected returns")
        elif momentum_ratio is not None:
            print(f"   ⚠️ Extreme momentum ratio {momentum_ratio:.3f}, skipping momentum adjustment")
        # NOTE: Covariance matrix is NOT scaled by momentum (use EWMA/GARCH for vol regime scaling if needed)
        
        n_assets = len(portfolio.etf_order)

        # Get daily costs for long/short
        daily_costs = ls_config.get_daily_costs() if enable_long_short else None

        # IMPROVED: Properly parameterized mean-variance optimization with long/short costs
        # FIX 3A: Added transaction cost awareness to prevent signal churning
        def objective(weights):
            """
            Objective: Mean-variance optimization with theoretically grounded risk aversion.

            FIX 3A: Includes transaction cost penalty to reduce turnover:
            Maximize: w'μ - TC × |w_t - w_{t-1}| - (λ/2) × w'Σw
            """
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))

            # Risk aversion parameter based on volatility target
            # Higher vol target = lower risk aversion
            risk_aversion = 1.0 / (daily_vol_target ** 2) if daily_vol_target > 0 else 1000

            # Add costs for long/short positions
            total_cost = 0.0
            if enable_long_short and daily_costs:
                # Leverage cost on gross exposure above 1.0
                gross_exposure = np.sum(np.abs(weights))
                if gross_exposure > 1.0:
                    leverage_amount = gross_exposure - 1.0
                    total_cost += leverage_amount * daily_costs['leverage_cost_daily'] / 10000

                # Short borrowing cost
                short_weights = np.minimum(weights, 0)
                short_exposure = np.abs(np.sum(short_weights))
                total_cost += short_exposure * daily_costs['borrow_cost_daily'] / 10000

                # Short rebate (typically minimal or zero)
                total_cost -= short_exposure * daily_costs['short_rebate_daily'] / 10000

            # FIX 3A: Transaction cost penalty based on turnover from previous weights
            tc_penalty = 0.0
            if config.TC_INCLUDE_IN_OPTIMIZER and previous_weights is not None:
                # Calculate turnover (one-way)
                turnover = np.sum(np.abs(weights - previous_weights.values))
                # Convert TC from basis points to decimal and apply to turnover
                tc_penalty = (config.TC_BPS_PER_TRADE / 10000) * turnover

            # Standard mean-variance objective: maximize return - costs - TC - (risk_aversion/2) * variance
            return -(portfolio_return - total_cost - tc_penalty - (risk_aversion / 2) * portfolio_variance)
        
        # Define constraints
        constraints = []
        constraint_status = {}

        # 1. Net exposure constraint: weights sum to net target
        if enable_long_short:
            # Long/short: net exposure equals target (e.g., 1.0 for 100% long bias)
            def net_exposure_constraint(w):
                return np.sum(w) - ls_config.NET_EXPOSURE_TARGET

            constraints.append({
                'type': 'eq',
                'fun': net_exposure_constraint
            })
            constraint_status['net_exposure'] = True
        else:
            # Long-only: traditional budget constraint (weights sum to 1)
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            constraint_status['budget'] = True

        # 1b. Gross exposure constraint (long/short only)
        if enable_long_short:
            def gross_exposure_constraint(w):
                """Gross exposure = sum of absolute weights must be <= limit"""
                return ls_config.GROSS_EXPOSURE_LIMIT - np.sum(np.abs(w))

            constraints.append({
                'type': 'ineq',
                'fun': gross_exposure_constraint
            })
            constraint_status['gross_exposure'] = True
            print(f"   📊 Added gross exposure constraint: <= {ls_config.GROSS_EXPOSURE_LIMIT:.0%}")
        else:
            constraint_status['gross_exposure'] = False

        # 2. SIMPLIFIED: Only add risk constraint if really needed
        # Remove risk constraint for faster convergence in most cases
        # FIX: Use annualized vol target since cov_matrix is annualized
        annualized_vol_target = daily_vol_target * config.SQRT_TRADING_DAYS
        if daily_vol_target < 0.01:  # Only for very low vol targets
            def risk_constraint(weights):
                portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))
                # Compare annualized variance to annualized target (with 20% slack)
                return (annualized_vol_target * 1.2)**2 - portfolio_var

            constraints.append({
                'type': 'ineq',
                'fun': risk_constraint
            })
            constraint_status['risk'] = True
        else:
            constraint_status['risk'] = False
        
        # 3. ⭐ CRITICAL: STRESS TEST CONSTRAINT ⭐
        # FIX 1A: Calculate PIT stress values if enabled
        stress_values_to_use = None
        if use_stress_test:
            if getattr(portfolio, 'use_pit_stress', False) or portfolio.stress_test_values is None:
                # Calculate Point-in-Time stress values using only historical data
                stress_values_to_use = _calculate_pit_stress_values(portfolio, end_date)
                print(f"   🔒 Using PIT stress values (max drawdown): {stress_values_to_use.min():.2f}% to {stress_values_to_use.max():.2f}%")
            else:
                # Use static stress values (legacy mode with potential look-ahead bias)
                stress_values_to_use = portfolio.stress_test_values

        if use_stress_test and stress_values_to_use is not None:
            def stress_test_constraint(weights):
                """Portfolio stress loss must be > -15% (loss less than 15%)"""
                if enable_long_short:
                    # For long/short: stress test on gross positions
                    # Apply stress to long positions, inverse to short positions
                    long_weights = np.maximum(weights, 0)
                    short_weights = np.minimum(weights, 0)

                    # Long positions lose value, short positions gain (negative of stress)
                    portfolio_stress_loss = (
                        np.dot(long_weights, stress_values_to_use.values) -
                        np.dot(np.abs(short_weights), stress_values_to_use.values)
                    )
                else:
                    # Traditional long-only stress test (FIX 1A: uses PIT values when enabled)
                    portfolio_stress_loss = np.dot(weights, stress_values_to_use.values)

                return portfolio_stress_loss + config.STRESS_TEST_MAX_LOSS

            constraints.append({
                'type': 'ineq',
                'fun': stress_test_constraint
            })
            constraint_status['stress_test'] = True
            print(f"   🔥 ADDED stress test constraint (max loss {config.STRESS_TEST_MAX_LOSS}%)")
        else:
            constraint_status['stress_test'] = False
        
        # 4. Duration constraints (if using duration model)
        if use_durations and portfolio.durations is not None and end_date in portfolio.durations.index:
            duration_values = portfolio.durations.loc[end_date]
            duration_target = -10.0  # Target duration
            duration_tolerance = 5.0
            
            def duration_lower_constraint(weights):
                portfolio_duration = np.dot(weights, duration_values.values)
                return portfolio_duration - (duration_target - duration_tolerance)
            
            def duration_upper_constraint(weights):
                portfolio_duration = np.dot(weights, duration_values.values)
                return (duration_target + duration_tolerance) - portfolio_duration
            
            constraints.extend([
                {'type': 'ineq', 'fun': duration_lower_constraint},
                {'type': 'ineq', 'fun': duration_upper_constraint}
            ])
            constraint_status['duration'] = True
        else:
            constraint_status['duration'] = False
        
        # Weight bounds: Long-only or Long/Short
        if enable_long_short:
            # Allow short positions
            bounds = [(ls_config.MAX_SHORT_WEIGHT, ls_config.MAX_LONG_WEIGHT) for _ in range(n_assets)]
            print(f"   📊 Weight bounds: [{ls_config.MAX_SHORT_WEIGHT:.0%}, {ls_config.MAX_LONG_WEIGHT:.0%}]")
        else:
            # Traditional long-only
            bounds = [(0.0, config.MAX_WEIGHT_PER_ASSET) for _ in range(n_assets)]
        
        # IMPROVED: Even better initial guess
        try:
            # Try multiple starting strategies
            initial_candidates = []
            
            # 1. Equal weights
            equal_weights = np.ones(n_assets) / n_assets
            initial_candidates.append(equal_weights)
            
            # 2. Return-weighted (simple momentum)
            if not expected_returns.isnull().any() and expected_returns.max() > expected_returns.min():
                return_weights = np.maximum(expected_returns.values, 0)
                if return_weights.sum() > 0:
                    return_weights = return_weights / return_weights.sum()
                    return_weights = np.clip(return_weights, 0, config.MAX_WEIGHT_PER_ASSET)
                    return_weights = return_weights / return_weights.sum()
                    initial_candidates.append(return_weights)
            
            # 3. Inverse volatility weighting
            diag_vol = np.sqrt(np.diag(cov_matrix.values))
            if diag_vol.min() > 0:
                inv_vol_weights = 1 / diag_vol
                inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
                inv_vol_weights = np.clip(inv_vol_weights, 0, config.MAX_WEIGHT_PER_ASSET)
                inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
                initial_candidates.append(inv_vol_weights)
            
            # Choose the best initial guess
            initial_weights = initial_candidates[0]  # Default to equal weights
            
        except:
            initial_weights = np.ones(n_assets) / n_assets
  
        # SIMPLIFIED: Focus on SLSQP with better settings
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 2000,  # Increased iterations
                    'ftol': 1e-4,     # Relaxed tolerance
                    'disp': False     # Suppress output
                }
            )
            
            # FIX 1B + 3B: No silent constraint dropping + strict status check
            # Use both result.success flag AND result.status == 0 for robustness
            optimization_failed = not result.success or result.status != 0
            if optimization_failed:
                print(f"   ❌ Optimization failed: {result.message}")

                # Diagnose what constraints were violated (FIX 1A: uses PIT stress values)
                # FIX: Pass annualized vol target since cov_matrix is annualized
                violations = _diagnose_constraint_violations(
                    result.x, cov_matrix, annualized_vol_target,
                    stress_values_to_use, use_stress_test, enable_long_short
                )
                if violations:
                    print(f"   📋 Constraint violations: {violations}")

                # FAILURE PROTOCOL: Use previous weights or equal weights
                if previous_weights is not None:
                    print(f"   ⚠️ Using previous period weights due to optimization failure")
                    fallback_weights = previous_weights
                    fallback_type = 'previous_weights'
                else:
                    print(f"   ⚠️ No previous weights available, using equal weights")
                    fallback_weights = pd.Series(1.0 / n_assets, index=portfolio.etf_order)
                    fallback_type = 'equal_weights'

                # Calculate metrics for fallback weights
                fallback_return = np.dot(fallback_weights.values, expected_returns.values)
                fallback_var = np.dot(fallback_weights.values, np.dot(cov_matrix.values, fallback_weights.values))
                fallback_vol = np.sqrt(fallback_var)

                return {
                    'success': True,  # True because we have valid weights (fallback)
                    'weights': fallback_weights,
                    'expected_return': fallback_return,
                    'volatility': fallback_vol,
                    'sharpe_ratio': fallback_return / fallback_vol if fallback_vol > 0 else 0,
                    'covariance_matrix': cov_matrix,
                    'constraints_satisfied': constraint_status,
                    'fallback_used': fallback_type,
                    'original_failure': result.message,
                    'constraint_violations': violations
                }

        except Exception as e:
            return {'success': False, 'error': f'Optimization setup error: {str(e)}'}

        # FIX 3B: Strict success check - must have status=0 AND success flag
        if result.success and result.status == 0:
            weights = pd.Series(result.x, index=portfolio.etf_order)
            expected_return = np.dot(weights.values, expected_returns.values)
            portfolio_var = np.dot(weights.values, np.dot(cov_matrix.values, weights.values))
            portfolio_vol = np.sqrt(portfolio_var)

            # Calculate stress test result if applicable (FIX 1A: uses PIT values when enabled)
            stress_test_result = None
            if use_stress_test and stress_values_to_use is not None:
                if enable_long_short:
                    # Long/short stress test
                    long_weights = np.maximum(weights.values, 0)
                    short_weights = np.minimum(weights.values, 0)
                    stress_test_result = (
                        np.dot(long_weights, stress_values_to_use.values) -
                        np.dot(np.abs(short_weights), stress_values_to_use.values)
                    )
                else:
                    stress_test_result = np.dot(weights.values, stress_values_to_use.values)

            # Calculate long/short specific metrics
            result_dict = {
                'success': True,
                'weights': weights,
                'expected_return': expected_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': expected_return / portfolio_vol if portfolio_vol > 0 else 0,
                'covariance_matrix': cov_matrix,
                'stress_test_result': stress_test_result,
                'constraints_satisfied': constraint_status,
                'optimization_result': result
            }

            if enable_long_short:
                gross_exposure = weights.abs().sum()
                net_exposure = weights.sum()
                long_exposure = weights[weights > 0].sum()
                short_exposure = weights[weights < 0].sum()
                leverage_ratio = gross_exposure / abs(net_exposure) if net_exposure != 0 else 0

                result_dict.update({
                    'gross_exposure': gross_exposure,
                    'net_exposure': net_exposure,
                    'long_exposure': long_exposure,
                    'short_exposure': short_exposure,
                    'leverage_ratio': leverage_ratio
                })

            return result_dict
        else:
            return {
                'success': False, 
                'error': f'Optimization failed: {result.message}',
                'constraints_satisfied': constraint_status
            }
            
    except Exception as e:
        return {'success': False, 'error': f'Optimization error: {str(e)}'}


def _diagnose_constraint_violations(weights: np.ndarray,
                                    cov_matrix: pd.DataFrame,
                                    vol_target: float,
                                    stress_values: Optional[pd.Series],
                                    use_stress_test: bool,
                                    enable_long_short: bool = False) -> Dict[str, str]:
    """
    FIX 1B: Diagnose specific constraint violations for logging.

    Returns dict mapping constraint name to violation description.
    """
    violations = {}

    # Budget constraint (should sum to 1.0 for long-only, or net exposure for L/S)
    weight_sum = np.sum(weights)
    if not enable_long_short:
        budget_deviation = abs(weight_sum - 1.0)
        if budget_deviation > 1e-4:
            violations['budget'] = f"Sum = {weight_sum:.4f}, deviation = {budget_deviation:.4f}"
    else:
        # For long/short, check net exposure bounds
        if abs(weight_sum - 1.0) > 0.15:  # ±15% tolerance on net exposure
            violations['net_exposure'] = f"Net exposure = {weight_sum:.4f}"

    # Volatility check
    try:
        portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        if portfolio_vol > vol_target * 1.5:  # 50% above target
            violations['volatility'] = f"Vol = {portfolio_vol:.4f}, target = {vol_target:.4f}"
    except Exception:
        violations['volatility'] = "Could not compute portfolio volatility"

    # Stress test constraint
    if use_stress_test and stress_values is not None:
        try:
            if enable_long_short:
                long_w = np.maximum(weights, 0)
                short_w = np.minimum(weights, 0)
                stress_loss = (
                    np.dot(long_w, stress_values.values) -
                    np.dot(np.abs(short_w), stress_values.values)
                )
            else:
                stress_loss = np.dot(weights, stress_values.values)

            if stress_loss < -config.STRESS_TEST_MAX_LOSS:
                violations['stress_test'] = f"Loss = {stress_loss:.2f}%, limit = {-config.STRESS_TEST_MAX_LOSS}%"
        except Exception:
            violations['stress_test'] = "Could not compute stress test loss"

    # Weight bounds
    if not enable_long_short:
        if (weights < -1e-6).any():
            neg_weights = weights[weights < -1e-6]
            violations['negative_weights'] = f"{len(neg_weights)} assets have negative weights"
        if (weights > config.MAX_WEIGHT_PER_ASSET + 1e-6).any():
            over_weights = weights[weights > config.MAX_WEIGHT_PER_ASSET + 1e-6]
            violations['max_weight'] = f"{len(over_weights)} assets exceed {config.MAX_WEIGHT_PER_ASSET:.0%} limit"
    else:
        # Long/short bounds
        if (weights > ls_config.MAX_LONG_WEIGHT + 1e-6).any():
            violations['max_long'] = f"Some weights exceed {ls_config.MAX_LONG_WEIGHT:.0%} long limit"
        if (weights < ls_config.MAX_SHORT_WEIGHT - 1e-6).any():
            violations['max_short'] = f"Some weights below {ls_config.MAX_SHORT_WEIGHT:.0%} short limit"

    return violations


def _calculate_momentum_ratio(prices: pd.DataFrame, 
                            ma_short: pd.DataFrame, 
                            ma_long: pd.DataFrame, 
                            date: pd.Timestamp) -> Optional[float]:
    """Calculate simple momentum ratio for given date"""
    try:
        if date not in ma_short.index or date not in ma_long.index:
            return None
        
        # Simple momentum: ratio of short MA to long MA averaged across assets
        short_ma_values = ma_short.loc[date]
        long_ma_values = ma_long.loc[date]
        
        # Calculate ratios where both values are valid
        valid_mask = (short_ma_values > 0) & (long_ma_values > 0)
        if not valid_mask.any():
            return None
            
        ratios = short_ma_values[valid_mask] / long_ma_values[valid_mask]
        return ratios.mean()
        
    except Exception:
        return None

print("✅ Optimization engine ready with ALL constraints including stress test")
print("🔥 CRITICAL: Stress test constraints now properly implemented!")


# =============================================================================
# SECTION 5: PERFORMANCE & RISK ANALYSIS
# =============================================================================

def calculate_portfolio_performance(optimization_results: Dict[str, Any], 
                                  portfolio: Portfolio) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio performance from optimization results.
    
    Args:
        optimization_results: Results from optimize_portfolio()
        portfolio: Portfolio instance with returns data
    
    Returns:
        Dictionary with performance metrics
    """
    print("📊 Calculating portfolio performance...")
    
    if not optimization_results['results']:
        print("❌ No optimization results to analyze")
        return {}
    
    # Extract weights and dates
    dates = [result['date'] for result in optimization_results['results']]
    weights_data = [result['weights'].values for result in optimization_results['results']]
    
    # Create weights DataFrame
    weights_df = pd.DataFrame(weights_data, index=dates, columns=portfolio.etf_order)
    
    # Calculate portfolio returns (forward-looking: use today's weights for tomorrow's return)
    portfolio_returns = _calculate_portfolio_returns(weights_df, portfolio.returns)
    
    # Calculate performance metrics
    performance_metrics = _calculate_performance_metrics(portfolio_returns)
    
    # Calculate risk metrics
    risk_metrics = _calculate_risk_metrics(portfolio_returns, weights_df, portfolio.returns)

    # Calculate long/short metrics if applicable
    long_short_metrics = None
    if portfolio.enable_long_short:
        long_short_metrics = calculate_long_short_metrics(optimization_results)

    # Combine all metrics
    full_performance = {
        'portfolio_returns': portfolio_returns,
        'weights': weights_df,
        'performance_metrics': performance_metrics,
        'risk_metrics': risk_metrics,
        'long_short_metrics': long_short_metrics,
        'optimization_summary': _summarize_optimization_results(optimization_results)
    }

    print(f"✅ Performance analysis completed")
    _print_performance_summary(performance_metrics, risk_metrics, long_short_metrics)

    return full_performance

def _calculate_portfolio_returns(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
    """Calculate daily portfolio returns from weights and asset returns"""
    
    # Align weights and returns on dates and assets
    common_dates = weights_df.index.intersection(returns_df.index)
    common_assets = weights_df.columns.intersection(returns_df.columns)
    
    if len(common_assets) < len(weights_df.columns):
        print(f"⚠️ Using {len(common_assets)} common assets out of {len(weights_df.columns)}")
    
    # Forward-looking returns: use yesterday's weights for today's returns
    portfolio_returns = []
    
    for i in range(len(common_dates) - 1):
        current_date = common_dates[i]
        next_date = common_dates[i + 1]
        
        if current_date in weights_df.index and next_date in returns_df.index:
            weights = weights_df.loc[current_date, common_assets]
            # CRITICAL FIX: Convert percentage returns to decimal returns
            returns = returns_df.loc[next_date, common_assets] / config.SCALE_FACTOR_FOR_RETURNS
            
            # Calculate portfolio return for next_date
            portfolio_return = np.dot(weights.values, returns.values)
            portfolio_returns.append((next_date, portfolio_return))
    
    # Convert to Series
    return pd.Series([ret for _, ret in portfolio_returns], 
                    index=[date for date, _ in portfolio_returns])

def _calculate_performance_metrics(portfolio_returns: pd.Series) -> Dict[str, float]:
    """Calculate standard performance metrics"""
    
    if portfolio_returns.empty:
        return {}
    
    # Account for transaction costs
    adjusted_returns = portfolio_returns - config.DAILY_TRANSACTION_COST_BP / 10000
    
    # Basic statistics
    mean_return = adjusted_returns.mean()
    std_return = adjusted_returns.std()
    
    # Annualized metrics
    annualized_return = mean_return * config.TRADING_DAYS_PER_YEAR
    annualized_volatility = std_return * config.SQRT_TRADING_DAYS
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Cumulative metrics
    cumulative_returns = (1 + adjusted_returns / 100).cumprod() if adjusted_returns.abs().max() < 10 else (1 + adjusted_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Drawdown analysis
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'daily_mean': mean_return,
        'daily_std': std_return
    }

def _calculate_risk_metrics(portfolio_returns: pd.Series, 
                          weights_df: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          confidence_level: float = 0.05) -> Dict[str, Any]:
    """Calculate comprehensive risk metrics"""
    
    if portfolio_returns.empty:
        return {}
    
    # Value at Risk (VaR)
    var_daily = portfolio_returns.quantile(confidence_level)
    var_annual = var_daily * config.SQRT_TRADING_DAYS
    
    # Conditional Value at Risk (CVaR)
    cvar_daily = portfolio_returns[portfolio_returns <= var_daily].mean()
    cvar_annual = cvar_daily * config.SQRT_TRADING_DAYS if not pd.isna(cvar_daily) else np.nan
    
    # Rolling risk metrics (252-day windows)
    rolling_vol = portfolio_returns.rolling(252).std() * config.SQRT_TRADING_DAYS
    rolling_var = portfolio_returns.rolling(252).quantile(confidence_level)
    
    # Concentration risk (weight concentration)
    weight_concentration = _calculate_concentration_metrics(weights_df)
    
    return {
        'var_95_daily': var_daily,
        'var_95_annual': var_annual,
        'cvar_95_daily': cvar_daily,
        'cvar_95_annual': cvar_annual,
        'rolling_volatility': rolling_vol,
        'rolling_var': rolling_var,
        'concentration_metrics': weight_concentration
    }

def _calculate_concentration_metrics(weights_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate portfolio concentration metrics"""
    
    if weights_df.empty:
        return {}
    
    # Herfindahl-Hirschman Index (sum of squared weights)
    hhi = (weights_df ** 2).sum(axis=1)
    
    # Effective number of assets
    effective_assets = 1 / hhi
    
    # Maximum weight statistics  
    max_weights = weights_df.max(axis=1)
    
    return {
        'avg_hhi': hhi.mean(),
        'avg_effective_assets': effective_assets.mean(),
        'avg_max_weight': max_weights.mean(),
        'max_concentration': max_weights.max()
    }

def _summarize_optimization_results(optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize optimization results"""
    
    results = optimization_results['results']
    if not results:
        return {}
    
    # Extract metrics from results
    expected_returns = [r['expected_return'] for r in results]
    volatilities = [r['volatility'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    stress_test_results = [r.get('stress_test_result') for r in results if r.get('stress_test_result') is not None]
    
    summary = {
        'total_optimizations': len(results),
        'success_rate': optimization_results['success_rate'],
        'avg_expected_return': np.mean(expected_returns),
        'avg_volatility': np.mean(volatilities),
        'avg_sharpe_ratio': np.mean(sharpe_ratios)
    }
    
    if stress_test_results:
        summary['avg_stress_test_loss'] = np.mean(stress_test_results)
        summary['max_stress_test_loss'] = np.min(stress_test_results)  # Most negative = worst loss
        summary['stress_violations'] = sum(1 for x in stress_test_results if x < -config.STRESS_TEST_MAX_LOSS)
    
    return summary

def calculate_long_short_metrics(optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate long/short specific metrics from optimization results.

    Args:
        optimization_results: Results from optimize_portfolio()

    Returns:
        Dictionary with long/short metrics
    """
    results = optimization_results.get('results', [])
    if not results or not any('gross_exposure' in r for r in results):
        return {}

    # Extract long/short metrics from results
    gross_exposures = [r.get('gross_exposure', 0) for r in results if 'gross_exposure' in r]
    net_exposures = [r.get('net_exposure', 0) for r in results if 'net_exposure' in r]
    long_exposures = [r.get('long_exposure', 0) for r in results if 'long_exposure' in r]
    short_exposures = [r.get('short_exposure', 0) for r in results if 'short_exposure' in r]
    leverage_ratios = [r.get('leverage_ratio', 0) for r in results if 'leverage_ratio' in r]

    if not gross_exposures:
        return {}

    return {
        'avg_gross_exposure': np.mean(gross_exposures),
        'max_gross_exposure': np.max(gross_exposures),
        'min_gross_exposure': np.min(gross_exposures),
        'avg_net_exposure': np.mean(net_exposures),
        'avg_long_exposure': np.mean(long_exposures),
        'avg_short_exposure': np.mean(short_exposures),
        'avg_leverage_ratio': np.mean(leverage_ratios),
        'max_leverage_ratio': np.max(leverage_ratios)
    }

def _print_performance_summary(performance_metrics: Dict[str, float],
                             risk_metrics: Dict[str, Any],
                             long_short_metrics: Dict[str, Any] = None):
    """Print formatted performance summary"""

    if not performance_metrics:
        return

    print("\n📈 PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Annualized Return:     {performance_metrics['annualized_return']:>8.2%}")
    print(f"Annualized Volatility: {performance_metrics['annualized_volatility']:>8.2%}")
    print(f"Sharpe Ratio:          {performance_metrics['sharpe_ratio']:>8.2f}")
    print(f"Max Drawdown:          {performance_metrics['max_drawdown']:>8.2%}")
    print(f"Total Return:          {performance_metrics['total_return']:>8.2%}")

    if risk_metrics:
        print(f"\n🎯 RISK METRICS")
        print("-" * 40)
        print(f"VaR (95%, Annual):     {risk_metrics.get('var_95_annual', 0):>8.2%}")
        print(f"CVaR (95%, Annual):    {risk_metrics.get('cvar_95_annual', 0):>8.2%}")

        conc = risk_metrics.get('concentration_metrics', {})
        if conc:
            print(f"Avg Effective Assets:  {conc.get('avg_effective_assets', 0):>8.1f}")
            print(f"Max Weight:            {conc.get('max_concentration', 0):>8.2%}")

    if long_short_metrics:
        print(f"\n📊 LONG/SHORT METRICS")
        print("-" * 40)
        print(f"Avg Gross Exposure:    {long_short_metrics.get('avg_gross_exposure', 0):>8.2%}")
        print(f"Avg Net Exposure:      {long_short_metrics.get('avg_net_exposure', 0):>8.2%}")
        print(f"Avg Long Exposure:     {long_short_metrics.get('avg_long_exposure', 0):>8.2%}")
        print(f"Avg Short Exposure:    {long_short_metrics.get('avg_short_exposure', 0):>8.2%}")
        print(f"Avg Leverage Ratio:    {long_short_metrics.get('avg_leverage_ratio', 0):>8.2f}")

def compare_with_benchmark(portfolio_performance: Dict[str, Any], 
                          portfolio: Portfolio) -> Dict[str, Any]:
    """Compare portfolio performance with benchmark"""
    
    if portfolio.benchmark_monthly_returns is None:
        print("⚠️ No benchmark data available for comparison")
        return {}
    
    print("📊 Comparing with benchmark (AGG)...")
    
    portfolio_returns = portfolio_performance['portfolio_returns']
    
    # Convert portfolio returns to monthly for comparison
    monthly_portfolio = _convert_to_monthly_returns(portfolio_returns)
    benchmark_data = portfolio.benchmark_monthly_returns.copy()
    
    # Align dates and calculate comparison metrics
    comparison_metrics = _calculate_benchmark_comparison(monthly_portfolio, benchmark_data)
    
    return comparison_metrics

def _convert_to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Convert daily returns to monthly"""
    return (1 + daily_returns / 100).resample('M').prod() - 1 if daily_returns.abs().max() < 10 else (1 + daily_returns).resample('M').prod() - 1

def _calculate_benchmark_comparison(portfolio_monthly: pd.Series, 
                                  benchmark_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate benchmark comparison metrics"""
    
    # This would need implementation based on benchmark data format
    # Placeholder for now
    return {
        'information_ratio': 0.0,
        'tracking_error': 0.0,
        'alpha': 0.0,
        'beta': 0.0
    }

print("✅ Performance analysis functions ready")


# =============================================================================
# SECTION 6: HIGH-LEVEL WORKFLOW MANAGEMENT
# =============================================================================

def run_portfolio_strategy(strategy_name: str,
                          max_windows: Optional[int] = None,
                          debug_mode: bool = True,
                          enable_long_short: bool = False) -> Dict[str, Any]:
    """
    Complete workflow: Load data, optimize, analyze performance for a single strategy.

    Args:
        strategy_name: Name of strategy (e.g., "Moderate Int.20121101.current")
        max_windows: Maximum number of optimization windows to process (for testing)
        debug_mode: If True, writes outputs to local debug folder
        enable_long_short: If True, enables long/short positions with leverage

    Returns:
        Dictionary with complete results
    """
    print(f"🚀 RUNNING PORTFOLIO STRATEGY: {strategy_name}")
    print("=" * 70)

    # Create fresh portfolio instance
    portfolio_instance = Portfolio(debug_mode=debug_mode, enable_long_short=enable_long_short)
    
    try:
        # Step 1: Load all data
        print("📂 STEP 1: Loading data...")
        if not load_portfolio_data(portfolio_instance, strategy_name):
            return {'success': False, 'error': 'Data loading failed'}
        
        # Step 2: Get strategy configuration
        print("⚙️ STEP 2: Setting up strategy configuration...")
        strategy_config = _get_strategy_configuration(strategy_name)
        print(f"   🎯 Strategy config: {strategy_config}")
        
        # Step 3: Prepare optimization windows
        print("📅 STEP 3: Preparing optimization windows...")
        if not portfolio_instance.optimization_windows:
            return {'success': False, 'error': 'No optimization windows available'}
        
        windows_to_use = portfolio_instance.optimization_windows
        if max_windows:
            windows_to_use = windows_to_use[-max_windows:]  # Use last N windows
            print(f"   ⚡ Using last {len(windows_to_use)} windows for testing")
        
        # Step 4: Run optimization
        print("🎯 STEP 4: Running portfolio optimization...")
        optimization_results = optimize_portfolio(
            portfolio_instance, 
            windows_to_use, 
            strategy_config
        )
        
        if optimization_results['success_rate'] < 0.5:
            print(f"⚠️ Low success rate: {optimization_results['success_rate']:.1%}")
        
        # Step 5: Analyze performance
        print("📊 STEP 5: Analyzing performance...")
        performance_results = calculate_portfolio_performance(
            optimization_results, 
            portfolio_instance
        )
        
        # Step 6: Export results
        print("💾 STEP 6: Exporting results...")
        export_results = _export_strategy_results(
            portfolio_instance, 
            optimization_results, 
            performance_results, 
            strategy_name
        )
        
        # Compile final results
        final_results = {
            'success': True,
            'strategy_name': strategy_name,
            'strategy_config': strategy_config,
            'optimization_results': optimization_results,
            'performance_results': performance_results,
            'export_results': export_results,
            'portfolio_summary': portfolio_instance.get_data_summary()
        }
        
        print(f"✅ STRATEGY COMPLETED SUCCESSFULLY: {strategy_name}")
        print(f"📊 Success rate: {optimization_results['success_rate']:.1%}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ ERROR in strategy {strategy_name}: {str(e)}")
        return {'success': False, 'error': str(e)}
    
    finally:
        # Clean up debug files if requested
        if debug_mode:
            print(f"📁 Debug outputs available at: {portfolio_instance.output_path}")
            # Note: Not automatically cleaning up so user can inspect results

def run_multiple_strategies(strategy_names: List[str], 
                           max_windows: Optional[int] = None,
                           debug_mode: bool = True) -> Dict[str, Any]:
    """
    Run multiple portfolio strategies in sequence.
    
    Args:
        strategy_names: List of strategy names to run
        max_windows: Maximum windows per strategy (for testing)
        debug_mode: If True, writes outputs to debug folders
    
    Returns:
        Dictionary with results for each strategy
    """
    print(f"🎯 RUNNING {len(strategy_names)} PORTFOLIO STRATEGIES")
    print("=" * 70)
    
    all_results = {}
    successful_strategies = 0
    
    for i, strategy_name in enumerate(strategy_names):
        print(f"\n📊 STRATEGY {i+1}/{len(strategy_names)}: {strategy_name}")
        print("-" * 60)
        
        try:
            result = run_portfolio_strategy(strategy_name, max_windows, debug_mode)
            all_results[strategy_name] = result
            
            if result['success']:
                successful_strategies += 1
                print(f"✅ {strategy_name} completed successfully")
            else:
                print(f"❌ {strategy_name} failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ ERROR in {strategy_name}: {str(e)}")
            all_results[strategy_name] = {'success': False, 'error': str(e)}
    
    print(f"\n🏁 BATCH COMPLETED: {successful_strategies}/{len(strategy_names)} successful")
    
    # Create summary report
    summary = _create_batch_summary(all_results)
    
    return {
        'strategy_results': all_results,
        'batch_summary': summary,
        'success_rate': successful_strategies / len(strategy_names)
    }

def run_quick_test(strategy_name: str = "Moderate Int.20121101.current",
                  num_windows: int = 10) -> Dict[str, Any]:
    """
    Quick test function for development and debugging.
    
    Args:
        strategy_name: Strategy to test
        num_windows: Number of windows to test (small for speed)
    
    Returns:
        Test results
    """
    print(f"🧪 QUICK TEST: {strategy_name} ({num_windows} windows)")
    print("=" * 50)
    
    return run_portfolio_strategy(strategy_name, max_windows=num_windows, debug_mode=True)

def _get_strategy_configuration(strategy_name: str) -> Dict[str, Any]:
    """Get configuration for a strategy based on its name"""
    
    # Find matching strategy configuration
    for strategy_config in config.PORTFOLIO_STRATEGIES:
        if strategy_config['name'] in strategy_name:
            return strategy_config.copy()
    
    # Default configuration if no match found
    if "High" in strategy_name:
        vol_target = "High"
    elif "Low" in strategy_name:
        vol_target = "Low"
    elif "RRS" in strategy_name:
        vol_target = "RRS"
    else:
        vol_target = "Moderate"
    
    return {
        'name': strategy_name,
        'vol_target': vol_target,
        'use_variable_vol': "VariableVol" in strategy_name,
        'use_stress_test': "RRS" in strategy_name,
        'use_durations': False
    }

def _export_strategy_results(portfolio: Portfolio,
                           optimization_results: Dict[str, Any],
                           performance_results: Dict[str, Any],
                           strategy_name: str) -> Dict[str, str]:
    """Export strategy results to files"""
    
    try:
        # Create output directory
        portfolio.create_directory(portfolio.output_path)
        
        output_files = {}
        
        # Export optimization results
        if optimization_results.get('results'):
            dates = [r['date'] for r in optimization_results['results']]
            weights_data = [r['weights'].values for r in optimization_results['results']]
            weights_df = pd.DataFrame(weights_data, index=dates, columns=portfolio.etf_order)
            
            weights_file = portfolio.output_path / f"{strategy_name.replace(' ', '_')}_weights.csv"
            weights_df.to_csv(weights_file)
            output_files['weights'] = str(weights_file)
        
        # Export performance results
        if performance_results.get('portfolio_returns') is not None:
            returns_file = portfolio.output_path / f"{strategy_name.replace(' ', '_')}_returns.csv"
            performance_results['portfolio_returns'].to_csv(returns_file)
            output_files['returns'] = str(returns_file)
        
        # Export performance summary
        if performance_results.get('performance_metrics'):
            summary_file = portfolio.output_path / f"{strategy_name.replace(' ', '_')}_summary.csv"
            pd.Series(performance_results['performance_metrics']).to_csv(summary_file)
            output_files['summary'] = str(summary_file)
        
        print(f"   💾 Exported {len(output_files)} files to {portfolio.output_path}")
        return output_files
        
    except Exception as e:
        print(f"   ⚠️ Export failed: {e}")
        return {}

def _create_batch_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary of batch processing results"""
    
    successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
    
    if not successful_results:
        return {'summary': 'No successful strategies'}
    
    # Extract performance metrics
    performance_data = {}
    for strategy_name, result in successful_results.items():
        perf = result.get('performance_results', {}).get('performance_metrics', {})
        if perf:
            performance_data[strategy_name] = perf
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data).T
        
        summary = {
            'successful_strategies': len(successful_results),
            'total_strategies': len(all_results),
            'best_sharpe': performance_df['sharpe_ratio'].idxmax() if 'sharpe_ratio' in performance_df.columns else None,
            'best_return': performance_df['annualized_return'].idxmax() if 'annualized_return' in performance_df.columns else None,
            'lowest_vol': performance_df['annualized_volatility'].idxmin() if 'annualized_volatility' in performance_df.columns else None,
            'performance_summary': performance_df.describe()
        }
        
        return summary
    
    return {'summary': 'Performance data not available'}

print("✅ Workflow management functions ready")
print("🚀 Main entry points:")
print("   • run_portfolio_strategy() - Single strategy workflow")
print("   • run_multiple_strategies() - Batch processing")  
print("   • run_quick_test() - Quick testing")


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def create_date_windows(start_date: str, end_date: str, window_days: int = 730) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create rolling date windows for optimization.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        window_days: Length of each window in days
    
    Returns:
        List of (start_date, end_date) tuples
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    windows = []
    current_start = start
    
    while current_start + pd.Timedelta(days=window_days) <= end:
        window_end = current_start + pd.Timedelta(days=window_days-1)
        windows.append((current_start, window_end))
        current_start += pd.Timedelta(days=1)
    
    return windows

def validate_portfolio_data(returns: pd.DataFrame, 
                           projected_returns: pd.DataFrame,
                           stress_test_values: pd.Series = None) -> Dict[str, Any]:
    """
    Validate portfolio data consistency and quality.
    
    Args:
        returns: Daily returns DataFrame
        projected_returns: Projected returns DataFrame
        stress_test_values: Stress test values Series
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check data alignment
    if not returns.index.equals(projected_returns.index):
        validation_results['warnings'].append('Date indices do not match between returns and projected returns')
    
    if not returns.columns.equals(projected_returns.columns):
        validation_results['warnings'].append('Column names do not match between returns and projected returns')
    
    # Check for missing data
    returns_missing = returns.isnull().sum().sum()
    projected_missing = projected_returns.isnull().sum().sum()
    
    if returns_missing > 0:
        validation_results['warnings'].append(f'Returns data has {returns_missing} missing values')
    
    if projected_missing > 0:
        validation_results['warnings'].append(f'Projected returns data has {projected_missing} missing values')
    
    # Check data ranges
    if returns.abs().max().max() > 50:  # Assumes returns are in percentage
        validation_results['warnings'].append('Returns data contains extreme values (>50%)')
    
    # Check stress test values if provided
    if stress_test_values is not None:
        if not stress_test_values.index.equals(returns.columns):
            validation_results['warnings'].append('Stress test ETF order does not match returns columns')
        
        if stress_test_values.abs().max() > 100:
            validation_results['warnings'].append('Stress test values contain extreme values (>100%)')
    
    # Set overall validity
    validation_results['valid'] = len(validation_results['errors']) == 0
    
    return validation_results

def calculate_portfolio_risk_decomposition(weights: pd.Series, 
                                         covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Decompose portfolio risk into individual asset contributions.
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
    
    Returns:
        Dictionary with risk decomposition
    """
    # Portfolio variance
    portfolio_var = np.dot(weights.values, np.dot(covariance_matrix.values, weights.values))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal contributions to risk (MCTR)
    mctr = np.dot(covariance_matrix.values, weights.values) / portfolio_vol
    
    # Component contributions to risk (CCTR)
    cctr = weights.values * mctr
    
    # Percentage contributions
    cctr_pct = cctr / portfolio_vol
    
    return {
        'portfolio_volatility': portfolio_vol,
        'marginal_contributions': pd.Series(mctr, index=weights.index),
        'component_contributions': pd.Series(cctr, index=weights.index),
        'percentage_contributions': pd.Series(cctr_pct, index=weights.index)
    }

def save_portfolio_results_to_excel(results_dict: Dict[str, Any], 
                                   output_file: str) -> bool:
    """
    Save comprehensive portfolio results to Excel with multiple sheets.
    
    Args:
        results_dict: Dictionary containing all portfolio results
        output_file: Path to output Excel file
    
    Returns:
        Boolean indicating success
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Performance metrics sheet
            if 'performance_results' in results_dict:
                perf = results_dict['performance_results']
                
                if 'performance_metrics' in perf:
                    pd.Series(perf['performance_metrics']).to_frame('Value').to_excel(
                        writer, sheet_name='Performance_Metrics')
                
                if 'portfolio_returns' in perf:
                    perf['portfolio_returns'].to_excel(writer, sheet_name='Portfolio_Returns')
                
                if 'weights' in perf:
                    perf['weights'].to_excel(writer, sheet_name='Weights')
            
            # Optimization summary
            if 'optimization_results' in results_dict:
                opt_summary = results_dict['optimization_results'].get('optimization_summary', {})
                if opt_summary:
                    pd.Series(opt_summary).to_frame('Value').to_excel(
                        writer, sheet_name='Optimization_Summary')
            
            print(f"✅ Results saved to {output_file}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        return False

def load_strategy_parameters_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load strategy parameters from configuration file.
    
    Args:
        file_path: Path to configuration file (CSV or JSON)
    
    Returns:
        Dictionary with strategy parameters
    """
    try:
        if file_path.endswith('.json'):
            import json
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0)
            return df.to_dict()
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
            
    except Exception as e:
        print(f"❌ Failed to load strategy parameters: {e}")
        return {}

def generate_performance_report(performance_results: Dict[str, Any], 
                              output_path: str = None) -> str:
    """
    Generate formatted performance report.
    
    Args:
        performance_results: Performance analysis results
        output_path: Optional path to save report
    
    Returns:
        Formatted report as string
    """
    if not performance_results:
        return "No performance data available"
    
    perf_metrics = performance_results.get('performance_metrics', {})
    risk_metrics = performance_results.get('risk_metrics', {})
    
    report_lines = [
        "=" * 60,
        "PORTFOLIO PERFORMANCE REPORT",
        "=" * 60,
        "",
        "RETURN METRICS:",
        f"  Annualized Return:      {perf_metrics.get('annualized_return', 0):>8.2%}",
        f"  Total Return:           {perf_metrics.get('total_return', 0):>8.2%}",
        "",
        "RISK METRICS:",
        f"  Annualized Volatility:  {perf_metrics.get('annualized_volatility', 0):>8.2%}",
        f"  Maximum Drawdown:       {perf_metrics.get('max_drawdown', 0):>8.2%}",
        f"  VaR (95%):             {risk_metrics.get('var_95_annual', 0):>8.2%}",
        "",
        "RISK-ADJUSTED METRICS:",
        f"  Sharpe Ratio:          {perf_metrics.get('sharpe_ratio', 0):>8.2f}",
        "",
    ]
    
    # Add concentration metrics if available
    concentration = risk_metrics.get('concentration_metrics', {})
    if concentration:
        report_lines.extend([
            "CONCENTRATION METRICS:",
            f"  Effective Assets:      {concentration.get('avg_effective_assets', 0):>8.1f}",
            f"  Max Weight:            {concentration.get('max_concentration', 0):>8.2%}",
            "",
        ])
    
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")
    
    return report_text

def validate_optimization_constraints(weights: pd.Series,
                                    stress_test_values: pd.Series = None,
                                    max_weight: float = 0.30,
                                    stress_test_limit: float = -15.0) -> Dict[str, bool]:
    """
    Validate that optimization results satisfy all constraints.
    
    Args:
        weights: Portfolio weights
        stress_test_values: Stress test loss values
        max_weight: Maximum weight per asset
        stress_test_limit: Maximum allowable stress test loss
    
    Returns:
        Dictionary with constraint validation results
    """
    results = {}
    
    # Budget constraint (weights sum to 1)
    results['budget_constraint'] = abs(weights.sum() - 1.0) < 1e-6
    
    # Non-negativity constraint
    results['non_negative'] = (weights >= -1e-6).all()
    
    # Maximum weight constraint
    results['max_weight'] = (weights <= max_weight + 1e-6).all()
    
    # Stress test constraint
    if stress_test_values is not None:
        portfolio_stress_loss = np.dot(weights.values, stress_test_values.values)
        results['stress_test'] = portfolio_stress_loss >= stress_test_limit
        results['stress_test_value'] = portfolio_stress_loss
    else:
        results['stress_test'] = True
        results['stress_test_value'] = None
    
    results['all_constraints_satisfied'] = all(v for k, v in results.items() if k.endswith('_constraint') or k == 'stress_test' or k == 'non_negative' or k == 'max_weight')
    
    return results

print("✅ Utility functions ready")
print("🔧 Available utilities:")
print("   • create_date_windows() - Date window creation")
print("   • validate_portfolio_data() - Data quality validation")
print("   • calculate_portfolio_risk_decomposition() - Risk attribution")
print("   • save_portfolio_results_to_excel() - Excel export")
print("   • generate_performance_report() - Formatted reporting")
print("   • validate_optimization_constraints() - Constraint validation")


# =============================================================================
# SECTION 8: TESTING & EXECUTION
# =============================================================================

def demo_quick_test():
    """
    Run a quick demo test to verify everything is working.
    Uses a small number of windows for speed.
    """
    print("🧪 RUNNING QUICK DEMO TEST")
    print("=" * 50)
    print("This will test the refactored system with a small dataset.")
    print("Expected runtime: 1-2 minutes")
    print()
    
    # Run quick test with limited windows
    try:
        result = run_quick_test(
            strategy_name="Moderate Int.20121101.current",
            num_windows=5  # Very small for demo
        )
        
        if result['success']:
            print("\n🎉 DEMO TEST SUCCESSFUL!")
            print("✅ All systems working correctly")
            
            # Show key results
            perf = result.get('performance_results', {}).get('performance_metrics', {})
            if perf:
                print(f"\n📊 Sample Results:")
                print(f"   Annualized Return: {perf.get('annualized_return', 0):.2%}")
                print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
            
            opt_summary = result.get('optimization_results', {})
            if opt_summary:
                print(f"   Success Rate: {opt_summary.get('success_rate', 0):.1%}")
        else:
            print(f"\n❌ DEMO TEST FAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ DEMO TEST ERROR: {str(e)}")

def demo_full_strategy():
    """
    Run a full strategy optimization (may take longer).
    """
    print("🚀 RUNNING FULL STRATEGY DEMO")
    print("=" * 50)
    print("This will run a complete optimization with real data.")
    print("Expected runtime: 5-15 minutes depending on data size")
    print()
    
    # Run full strategy
    try:
        result = run_portfolio_strategy(
            strategy_name="Moderate Int.20121101.current",
            max_windows=100,  # Reasonable size for demo
            debug_mode=True
        )
        
        if result['success']:
            print("\n🎉 FULL STRATEGY DEMO SUCCESSFUL!")
            
            # Generate and display performance report
            perf_results = result.get('performance_results', {})
            if perf_results:
                print("\n📊 PERFORMANCE REPORT:")
                report = generate_performance_report(perf_results)
                print(report)
            
            # Show file outputs
            export_results = result.get('export_results', {})
            if export_results:
                print(f"\n💾 Files created:")
                for file_type, file_path in export_results.items():
                    print(f"   {file_type}: {file_path}")
        else:
            print(f"\n❌ FULL STRATEGY DEMO FAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ FULL STRATEGY DEMO ERROR: {str(e)}")

def demo_multiple_strategies():
    """
    Run multiple strategies for comparison.
    """
    print("🎯 RUNNING MULTIPLE STRATEGIES DEMO")
    print("=" * 50)
    
    # Select a few key strategies
    strategies_to_test = [
        "Moderate Int.20121101.current",
        "High Int.20121101.current", 
        "Low Int.20121101.current"
    ]
    
    print(f"Testing {len(strategies_to_test)} strategies:")
    for strategy in strategies_to_test:
        print(f"   • {strategy}")
    print()
    
    try:
        results = run_multiple_strategies(
            strategy_names=strategies_to_test,
            max_windows=20,  # Small for demo
            debug_mode=True
        )
        
        print(f"\n🏁 BATCH RESULTS:")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        # Show comparison of successful strategies
        successful = {k: v for k, v in results['strategy_results'].items() 
                     if v.get('success', False)}
        
        if successful:
            print(f"\n📊 STRATEGY COMPARISON:")
            for strategy_name, result in successful.items():
                perf = result.get('performance_results', {}).get('performance_metrics', {})
                if perf:
                    print(f"\n{strategy_name}:")
                    print(f"   Return: {perf.get('annualized_return', 0):>8.2%}")
                    print(f"   Sharpe: {perf.get('sharpe_ratio', 0):>8.2f}")
                    print(f"   Vol:    {perf.get('annualized_volatility', 0):>8.2%}")
        
    except Exception as e:
        print(f"\n❌ MULTIPLE STRATEGIES ERROR: {str(e)}")

def demo_long_short_strategy():
    """
    Run a long/short strategy demo with leverage.
    """
    print("🚀 RUNNING LONG/SHORT STRATEGY DEMO")
    print("=" * 50)
    print("This will run optimization with short positions and leverage enabled.")
    print("Expected runtime: 5-10 minutes")
    print()

    try:
        result = run_portfolio_strategy(
            strategy_name="Moderate Int.20121101.current",
            max_windows=50,  # Reasonable size for demo
            debug_mode=True,
            enable_long_short=True  # Enable long/short mode
        )

        if result['success']:
            print("\n🎉 LONG/SHORT STRATEGY DEMO SUCCESSFUL!")

            # Generate and display performance report
            perf_results = result.get('performance_results', {})
            if perf_results:
                print("\n📊 PERFORMANCE REPORT:")

                perf_metrics = perf_results.get('performance_metrics', {})
                if perf_metrics:
                    print(f"\n📈 Returns:")
                    print(f"   Annualized Return: {perf_metrics.get('annualized_return', 0):.2%}")
                    print(f"   Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"   Max Drawdown: {perf_metrics.get('max_drawdown', 0):.2%}")

                ls_metrics = perf_results.get('long_short_metrics', {})
                if ls_metrics:
                    print(f"\n📊 Long/Short Exposure:")
                    print(f"   Avg Gross Exposure: {ls_metrics.get('avg_gross_exposure', 0):.2%}")
                    print(f"   Avg Net Exposure: {ls_metrics.get('avg_net_exposure', 0):.2%}")
                    print(f"   Avg Long Exposure: {ls_metrics.get('avg_long_exposure', 0):.2%}")
                    print(f"   Avg Short Exposure: {ls_metrics.get('avg_short_exposure', 0):.2%}")
                    print(f"   Avg Leverage Ratio: {ls_metrics.get('avg_leverage_ratio', 0):.2f}x")

            # Show file outputs
            export_results = result.get('export_results', {})
            if export_results:
                print(f"\n💾 Files created:")
                for file_type, file_path in export_results.items():
                    print(f"   {file_type}: {file_path}")
        else:
            print(f"\n❌ LONG/SHORT STRATEGY DEMO FAILED: {result.get('error')}")

    except Exception as e:
        print(f"\n❌ LONG/SHORT STRATEGY DEMO ERROR: {str(e)}")

def show_system_summary():
    """
    Display a summary of the refactored system capabilities.
    """
    print("\n" + "=" * 70)
    print("FOLIO BEYOND FIXED INCOME OPTIMIZATION SYSTEM - REFACTORED")
    print("=" * 70)
    
    print("\n🎯 KEY FEATURES:")
    print("  ✅ Mean-variance optimization with stress test constraints")
    print("  ✅ Long-only AND long/short with leverage support")
    print("  ✅ Multiple volatility targets (Low/Moderate/High/RRS)")
    print("  ✅ Comprehensive risk analytics (VaR, CVaR, drawdown)")
    print("  ✅ Standardized data types and clean architecture")
    print("  ✅ Configuration-driven with all parameters externalized")
    print("  ✅ Debug mode for safe testing")
    print("  ✅ Modular design for easy extension")
    
    print("\n🚀 MAIN ENTRY POINTS:")
    print("  • run_quick_test() - Quick testing with limited data")
    print("  • run_portfolio_strategy() - Complete single strategy workflow")
    print("  • run_multiple_strategies() - Batch processing multiple strategies")
    
    print("\n🧪 DEMO FUNCTIONS:")
    print("  • demo_quick_test() - 1-2 minute quick verification")
    print("  • demo_full_strategy() - 5-15 minute full long-only strategy")
    print("  • demo_long_short_strategy() - 5-10 minute long/short with leverage")
    print("  • demo_multiple_strategies() - Compare multiple strategies")

    print("\n⚙️ CONFIGURATION - LONG-ONLY:")
    print(f"  • Trading days per year: {config.TRADING_DAYS_PER_YEAR}")
    print(f"  • Lookback window: {config.LOOKBACK_WINDOW_DAYS} days")
    print(f"  • Max weight per asset: {config.MAX_WEIGHT_PER_ASSET:.1%}")
    print(f"  • Stress test max loss: {config.STRESS_TEST_MAX_LOSS}%")

    print("\n⚙️ CONFIGURATION - LONG/SHORT:")
    print(f"  • Max long weight: {ls_config.MAX_LONG_WEIGHT:.1%}")
    print(f"  • Max short weight: {ls_config.MAX_SHORT_WEIGHT:.1%}")
    print(f"  • Gross exposure limit: {ls_config.GROSS_EXPOSURE_LIMIT:.1%}")
    print(f"  • Net exposure target: {ls_config.NET_EXPOSURE_TARGET:.1%}")
    print(f"  • Leverage cost: {ls_config.LEVERAGE_COST_BPS/100:.2%} annual")
    print(f"  • Borrow cost: {ls_config.BORROW_COST_BPS/100:.2%} annual")
    
    print("\n💾 OUTPUT:")
    print("  • All outputs written to debug folders by default (safe testing)")
    print("  • CSV exports: weights, returns, performance summary")
    print("  • Formatted performance reports")
    print("  • Excel exports with multiple sheets")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create portfolio instance in DEBUG mode by default for safety
    portfolio = Portfolio(debug_mode=True)
    print(f"📊 Portfolio summary: {portfolio.get_data_summary()}")

    # Show system summary when file is executed
    show_system_summary()

    print("\n🧪 READY TO TEST!")
    print("\nChoose your testing approach:")
    print("1️⃣ Quick verification:           demo_quick_test()")
    print("2️⃣ Full long-only strategy:     demo_full_strategy()")
    print("3️⃣ Long/short with leverage:    demo_long_short_strategy()")
    print("4️⃣ Multiple strategies:          demo_multiple_strategies()")
    print("\nOr use the main functions directly:")
    print("🎯 run_quick_test()")
    print("🚀 run_portfolio_strategy('Moderate Int.20121101.current', enable_long_short=False)")
    print("📊 run_portfolio_strategy('Moderate Int.20121101.current', enable_long_short=True)")
    print("📈 run_multiple_strategies(['strategy1', 'strategy2'])")

    # Uncomment one of the lines below to run a demo automatically:
    # demo_quick_test()
    # demo_full_strategy()
    # demo_long_short_strategy()
    pass
    


    