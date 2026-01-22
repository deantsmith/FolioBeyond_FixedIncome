"""Training pipeline and evaluation modules."""

from .trainer import Trainer
from .evaluation import Evaluator, BacktestResult

__all__ = ['Trainer', 'Evaluator', 'BacktestResult']
