"""
Evaluation and backtesting module for portfolio models.

Calculates performance metrics and compares against benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from ..config import Config
from ..interface.model_output import ModelOutputTimeSeries
from ..data.loader import PortfolioData


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    
    Includes portfolio returns, risk metrics, and comparison to benchmark.
    """
    
    returns: pd.Series
    weights: pd.DataFrame
    model_name: str
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Turnover metrics
    avg_turnover: float = 0.0
    total_turnover: float = 0.0
    
    # Benchmark comparison
    benchmark_return: Optional[float] = None
    benchmark_volatility: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"\n{'='*50}",
            f"Backtest Results: {self.model_name}",
            f"{'='*50}",
            f"Period: {self.returns.index[0].strftime('%Y-%m-%d')} to {self.returns.index[-1].strftime('%Y-%m-%d')}",
            f"",
            f"Returns:",
            f"  Total Return:       {self.total_return:>10.2%}",
            f"  Annualized Return:  {self.annualized_return:>10.2%}",
            f"",
            f"Risk:",
            f"  Annualized Vol:     {self.annualized_volatility:>10.2%}",
            f"  Max Drawdown:       {self.max_drawdown:>10.2%}",
            f"  VaR (95%):          {self.var_95:>10.2%}",
            f"  CVaR (95%):         {self.cvar_95:>10.2%}",
            f"",
            f"Risk-Adjusted:",
            f"  Sharpe Ratio:       {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:      {self.sortino_ratio:>10.2f}",
            f"",
            f"Trading:",
            f"  Avg Turnover:       {self.avg_turnover:>10.2%}",
            f"  Total Turnover:     {self.total_turnover:>10.2%}",
        ]
        
        if self.benchmark_return is not None:
            lines.extend([
                f"",
                f"Benchmark Comparison:",
                f"  Benchmark Return:   {self.benchmark_return:>10.2%}",
                f"  Alpha:              {self.alpha:>10.2%}" if self.alpha else "",
                f"  Beta:               {self.beta:>10.2f}" if self.beta else "",
                f"  Information Ratio:  {self.information_ratio:>10.2f}" if self.information_ratio else "",
                f"  Tracking Error:     {self.tracking_error:>10.2%}" if self.tracking_error else "",
            ])
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'avg_turnover': self.avg_turnover,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio
        }


class Evaluator:
    """
    Evaluator for portfolio model performance.
    
    Calculates returns from predicted weights and actual returns,
    then computes comprehensive performance metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
    
    def evaluate(
        self,
        outputs: ModelOutputTimeSeries,
        data: PortfolioData,
        benchmark: Optional[pd.Series] = None,
        transaction_cost_bps: float = 10.0
    ) -> BacktestResult:
        """
        Evaluate model predictions against actual returns.
        
        Args:
            outputs: ModelOutputTimeSeries with predicted weights
            data: PortfolioData with actual returns
            benchmark: Optional benchmark returns for comparison
            transaction_cost_bps: Transaction cost in basis points
            
        Returns:
            BacktestResult with all metrics
        """
        if len(outputs) == 0:
            raise ValueError("No outputs to evaluate")
        
        # Get weights DataFrame
        weights_df = outputs.weights_df
        
        # Get actual returns (scaled)
        returns = data.get_returns_scaled()
        
        # Align dates
        common_dates = weights_df.index.intersection(returns.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between predictions and returns")
        
        weights_df = weights_df.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Calculate portfolio returns
        # Use previous day's weights for current day's return (realistic lag)
        portfolio_returns = (weights_df.shift(1) * returns).sum(axis=1).dropna()
        
        # Apply transaction costs
        turnover = self._calculate_turnover(weights_df)
        tc_cost = turnover * (transaction_cost_bps / 10000)
        portfolio_returns_net = portfolio_returns - tc_cost.reindex(portfolio_returns.index).fillna(0)
        
        # Calculate metrics
        result = self._calculate_metrics(
            portfolio_returns_net,
            weights_df,
            outputs.model_name
        )
        
        # Benchmark comparison
        if benchmark is not None or data.benchmark_returns is not None:
            bench = benchmark if benchmark is not None else data.benchmark_returns
            result = self._add_benchmark_metrics(result, portfolio_returns_net, bench)
        
        return result
    
    def _calculate_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover."""
        weight_changes = weights.diff().abs().sum(axis=1)
        return weight_changes / 2  # One-way turnover
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        model_name: str
    ) -> BacktestResult:
        """Calculate performance metrics from returns."""
        trading_days = self.config.TRADING_DAYS_PER_YEAR
        
        # Total and annualized return
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_years = len(returns) / trading_days
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        annualized_vol = returns.std() * np.sqrt(trading_days)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        
        # Turnover
        turnover = self._calculate_turnover(weights)
        avg_turnover = turnover.mean()
        total_turnover = turnover.sum()
        
        return BacktestResult(
            returns=returns,
            weights=weights,
            model_name=model_name,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            avg_turnover=avg_turnover,
            total_turnover=total_turnover
        )
    
    def _add_benchmark_metrics(
        self,
        result: BacktestResult,
        portfolio_returns: pd.Series,
        benchmark: pd.Series
    ) -> BacktestResult:
        """Add benchmark comparison metrics."""
        # Align
        common_idx = portfolio_returns.index.intersection(benchmark.index)
        if len(common_idx) == 0:
            return result
        
        port_ret = portfolio_returns.loc[common_idx]
        bench_ret = benchmark.loc[common_idx]
        
        # Scale benchmark if needed
        if bench_ret.abs().mean() > 1:  # Likely scaled by 100
            bench_ret = bench_ret / self.config.SCALE_FACTOR_FOR_RETURNS
        
        trading_days = self.config.TRADING_DAYS_PER_YEAR
        
        # Benchmark return and vol
        bench_cumulative = (1 + bench_ret).cumprod()
        bench_total = bench_cumulative.iloc[-1] - 1
        n_years = len(bench_ret) / trading_days
        bench_ann_return = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_ann_vol = bench_ret.std() * np.sqrt(trading_days)
        
        # Active returns
        active_returns = port_ret - bench_ret
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(trading_days)
        
        # Information ratio
        active_ann_return = active_returns.mean() * trading_days
        information_ratio = active_ann_return / tracking_error if tracking_error > 0 else 0
        
        # Alpha and Beta (CAPM)
        if len(port_ret) > 30:
            cov_matrix = np.cov(port_ret, bench_ret)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0
            alpha = result.annualized_return - beta * bench_ann_return
        else:
            beta = 1.0
            alpha = result.annualized_return - bench_ann_return
        
        result.benchmark_return = bench_ann_return
        result.benchmark_volatility = bench_ann_vol
        result.alpha = alpha
        result.beta = beta
        result.information_ratio = information_ratio
        result.tracking_error = tracking_error
        
        return result
    
    def compare_models(
        self,
        results: List[BacktestResult]
    ) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison = []
        
        for result in results:
            comparison.append(result.to_dict())
        
        df = pd.DataFrame(comparison)
        df = df.set_index('model_name')
        
        # Format percentages
        pct_columns = ['total_return', 'annualized_return', 'annualized_volatility',
                      'max_drawdown', 'var_95', 'cvar_95', 'avg_turnover',
                      'benchmark_return', 'alpha', 'tracking_error']
        
        return df
    
    def plot_results(
        self,
        result: BacktestResult,
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Plot backtest results.
        
        Args:
            result: BacktestResult to plot
            benchmark: Optional benchmark for comparison
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        cumulative = (1 + result.returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label=result.model_name)
        
        if benchmark is not None:
            bench_aligned = benchmark.reindex(result.returns.index)
            if bench_aligned.abs().mean() > 1:
                bench_aligned = bench_aligned / self.config.SCALE_FACTOR_FOR_RETURNS
            bench_cum = (1 + bench_aligned).cumprod()
            ax1.plot(bench_cum.index, bench_cum.values, label='Benchmark', alpha=0.7)
        
        ax1.set_title('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5)
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe
        ax3 = axes[1, 0]
        rolling_sharpe = (
            result.returns.rolling(63).mean() * np.sqrt(self.config.TRADING_DAYS_PER_YEAR) /
            result.returns.rolling(63).std()
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling Sharpe (63-day)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Weight allocation over time
        ax4 = axes[1, 1]
        result.weights.plot.area(ax=ax4, alpha=0.7)
        ax4.set_title('Weight Allocation')
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
