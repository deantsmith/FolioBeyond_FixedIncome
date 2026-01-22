"""
Neural network models for portfolio optimization.

Implements feedforward and LSTM architectures using PyTorch.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import Config
from ..data.features import FeatureSet
from .base import BasePortfolioModel


class FeedforwardPortfolioModel(BasePortfolioModel):
    """
    Feedforward neural network for portfolio optimization.
    
    Directly predicts portfolio weights from features using
    dense layers with dropout regularization.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "feedforward"):
        super().__init__(config, name)
        self.model = None
        self.scaler = None
        self.params = self.config.NEURAL_PARAMS.copy()
        self.device = self.config.get_torch_device()
        self.training_history: Dict[str, List[float]] = {}
    
    def _build_model(self, input_dim: int, output_dim: int):
        """Build the neural network architecture."""
        import torch
        import torch.nn as nn
        
        hidden_dims = self.params['hidden_dims']
        dropout = self.params['dropout']
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw predictions)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        return self.model
    
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        validation_fraction: float = 0.2,
        **kwargs
    ) -> 'FeedforwardPortfolioModel':
        """
        Train the feedforward network.
        
        Args:
            features: FeatureSet with training data
            asset_names: List of asset names
            validation_fraction: Fraction for validation
            **kwargs: Override default parameters
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        params = {**self.params, **kwargs}
        
        self.asset_names = asset_names
        self.feature_names = features.features.columns.tolist()
        
        # Prepare data
        X = features.features.values
        
        if features.target is None:
            raise ValueError("FeatureSet must include target for training")
        
        # Get target for all assets
        y = features.target[asset_names].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(y).any(axis=1) & ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n🧠 Training Feedforward NN ({len(X)} samples, {X.shape[1]} features)...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/val split
        split_idx = int(len(X) * (1 - validation_fraction))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # Build model
        self._build_model(X.shape[1], len(asset_names))
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(params['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self._best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= params['early_stopping_patience']:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Load best model
        if hasattr(self, '_best_model_state'):
            self.model.load_state_dict(self._best_model_state)
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_assets': len(asset_names),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'epochs_trained': len(self.training_history['train_loss'])
        }
        
        print(f"   ✅ Training complete (best val_loss: {best_val_loss:.6f})")
        return self
    
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """Generate predictions from the neural network."""
        import torch
        
        X = features.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions[0]
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'params': self.params,
            'device': self.device,
            'training_history': self.training_history
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        import torch
        
        self.scaler = state['scaler']
        self.params = state['params']
        self.device = state.get('device', 'cpu')
        self.training_history = state.get('training_history', {})
        
        if state['model_state_dict'] and self.feature_names and self.asset_names:
            self._build_model(len(self.feature_names), len(self.asset_names))
            self.model.load_state_dict(state['model_state_dict'])


class LSTMPortfolioModel(BasePortfolioModel):
    """
    LSTM neural network for portfolio optimization.
    
    Uses sequence of historical features to capture temporal
    dependencies in market data.
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = "lstm"):
        super().__init__(config, name)
        self.model = None
        self.scaler = None
        self.params = self.config.LSTM_PARAMS.copy()
        self.device = self.config.get_torch_device()
        self.training_history: Dict[str, List[float]] = {}
    
    def _build_model(self, input_dim: int, output_dim: int):
        """Build the LSTM architecture."""
        import torch
        import torch.nn as nn
        
        hidden_dim = self.params['hidden_dim']
        num_layers = self.params['num_layers']
        dropout = self.params['dropout']
        
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, 
                    hidden_dim, 
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last timestep
                last_output = lstm_out[:, -1, :]
                out = self.dropout(last_output)
                out = self.fc(out)
                return out
        
        self.model = LSTMModel(
            input_dim, hidden_dim, output_dim, num_layers, dropout
        ).to(self.device)
        
        return self.model
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        sequences = []
        targets = []
        
        for i in range(seq_length, len(X)):
            sequences.append(X[i-seq_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def fit(
        self, 
        features: FeatureSet,
        asset_names: List[str],
        validation_fraction: float = 0.2,
        **kwargs
    ) -> 'LSTMPortfolioModel':
        """
        Train the LSTM network.
        
        Args:
            features: FeatureSet with training data
            asset_names: List of asset names
            validation_fraction: Fraction for validation
            **kwargs: Override default parameters
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        params = {**self.params, **kwargs}
        seq_length = params['sequence_length']
        
        self.asset_names = asset_names
        self.feature_names = features.features.columns.tolist()
        
        # Prepare data
        X = features.features.values
        
        if features.target is None:
            raise ValueError("FeatureSet must include target for training")
        
        y = features.target[asset_names].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(y).any(axis=1) & ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n🔄 Training LSTM ({len(X)} samples, seq_length={seq_length})...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y, seq_length)
        
        print(f"   Created {len(X_seq)} sequences")
        
        # Train/val split
        split_idx = int(len(X_seq) * (1 - validation_fraction))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # Build model
        self._build_model(X.shape[1], len(asset_names))
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(params['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Load best model
        if hasattr(self, '_best_model_state'):
            self.model.load_state_dict(self._best_model_state)
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_sequences': len(X_seq),
            'n_features': X.shape[1],
            'n_assets': len(asset_names),
            'sequence_length': seq_length,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1]
        }
        
        print(f"   ✅ Training complete (best val_loss: {best_val_loss:.6f})")
        return self
    
    def predict_raw(
        self, 
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> np.ndarray:
        """
        Generate predictions from LSTM.
        
        Note: For LSTM, features should contain the sequence
        (seq_length rows of features).
        """
        import torch
        
        X = features.values
        seq_length = self.params['sequence_length']
        
        # Check if we have enough data for a sequence
        if len(X) < seq_length:
            # Pad with zeros or repeat
            padding = np.zeros((seq_length - len(X), X.shape[1]))
            X = np.vstack([padding, X])
        elif len(X) > seq_length:
            X = X[-seq_length:]
        
        # Scale and reshape
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions[0]
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'params': self.params,
            'device': self.device,
            'training_history': self.training_history
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        import torch
        
        self.scaler = state['scaler']
        self.params = state['params']
        self.device = state.get('device', 'cpu')
        self.training_history = state.get('training_history', {})
        
        if state['model_state_dict'] and self.feature_names and self.asset_names:
            self._build_model(len(self.feature_names), len(self.asset_names))
            self.model.load_state_dict(state['model_state_dict'])
