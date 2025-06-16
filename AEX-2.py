# ==============================================================================
# AEX Index Forecasting with Attention-Enhanced LSTM
# Professional Implementation for Academic Research
# ==============================================================================

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
import json
import pickle

# Deep Learning & ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Financial Data
import yfinance as yf
import ta

# Optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Visualization & Reporting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Statistical Testing
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS

# Configuration Management
from omegaconf import OmegaConf

# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Comprehensive configuration class for the AEX forecasting system."""
    
    # Data Configuration
    data_config: Dict = None
    
    # Model Configuration  
    model_config: Dict = None
    
    # Training Configuration
    training_config: Dict = None
    
    # Evaluation Configuration
    evaluation_config: Dict = None
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = {
                'symbol': '^AEX',
                'start_date': '2010-01-01',
                'end_date': '2024-12-31',
                'sequence_length': 60,
                'prediction_horizon': 1,
                'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'technical_indicators': True,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'scaling_method': 'minmax'
            }
            
        if self.model_config is None:
            self.model_config = {
                'lstm_hidden_size': 64,
                'lstm_num_layers': 2,
                'attention_hidden_size': 32,
                'dropout_rate': 0.2,
                'output_size': 1,
                'bidirectional': False
            }
            
        if self.training_config is None:
            self.training_config = {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 10,
                'min_delta': 1e-6,
                'gradient_clip_val': 1.0,
                'scheduler_patience': 5,
                'scheduler_factor': 0.5
            }
            
        if self.evaluation_config is None:
            self.evaluation_config = {
                'metrics': ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy'],
                'statistical_tests': ['diebold_mariano', 'dm_test'],
                'confidence_level': 0.95,
                'bootstrap_samples': 1000
            }

# Initialize configuration
config = Config()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging configuration."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"aex_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
    
    return logger

logger = setup_logging()

# ==============================================================================
# DATA COLLECTION AND PREPROCESSING
# ==============================================================================

class AEXDataProcessor:
    """Professional data processing pipeline for AEX index forecasting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.feature_columns = None
        self.raw_data = None
        self.processed_data = None
        
    def download_data(self) -> pd.DataFrame:
        """Download AEX index data with comprehensive error handling."""
        try:
            logger.info(f"Downloading AEX data from {self.config.data_config['start_date']} to {self.config.data_config['end_date']}")
            
            ticker = yf.Ticker(self.config.data_config['symbol'])
            data = ticker.history(
                start=self.config.data_config['start_date'],
                end=self.config.data_config['end_date'],
                interval='1d'
            )
            
            if data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
                
            logger.info(f"Successfully downloaded {len(data)} trading days of data")
            self.raw_data = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the dataset."""
        try:
            logger.info("Computing technical indicators...")
            
            # Trend Indicators
            data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            
            data['EMA_5'] = ta.trend.ema_indicator(data['Close'], window=5)
            data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
            data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()
            
            # Momentum Indicators
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            data['Stochastic_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stochastic_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Volatility Indicators
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bollinger.bollinger_hband()
            data['BB_middle'] = bollinger.bollinger_mavg()
            data['BB_lower'] = bollinger.bollinger_lband()
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Volume Indicators
            data['Volume_SMA'] = ta.trend.sma_indicator(data['Volume'], window=20)
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Price-based features
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            
            # Returns
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Volatility (rolling standard deviation of returns)
            data['Volatility_5'] = data['Returns'].rolling(window=5).std()
            data['Volatility_20'] = data['Returns'].rolling(window=20).std()
            
            logger.info(f"Added {len(data.columns) - 5} technical indicators")
            
            return data
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning and quality assurance."""
        try:
            logger.info("Starting data cleaning process...")
            
            initial_rows = len(data)
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill missing values for financial data
            data = data.fillna(method='ffill')
            
            # Backward fill any remaining NaN values at the beginning
            data = data.fillna(method='bfill')
            
            # Remove any remaining rows with NaN values
            data = data.dropna()
            
            # Remove outliers using IQR method for returns
            if 'Returns' in data.columns:
                Q1 = data['Returns'].quantile(0.01)
                Q3 = data['Returns'].quantile(0.99)
                data = data[(data['Returns'] >= Q1) & (data['Returns'] <= Q3)]
            
            # Ensure positive values for volume
            if 'Volume' in data.columns:
                data = data[data['Volume'] > 0]
            
            final_rows = len(data)
            logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows ({final_rows/initial_rows:.2%} retained)")
            
            return data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """Scale features using specified scaling method."""
        try:
            logger.info(f"Scaling features using {self.config.data_config['scaling_method']} method...")
            
            # Select numeric columns only
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if self.config.data_config['scaling_method'] == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif self.config.data_config['scaling_method'] == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {self.config.data_config['scaling_method']}")
            
            # Fit scaler on training data only (first 70% of data)
            train_size = int(len(data) * self.config.data_config['train_ratio'])
            train_data = data.iloc[:train_size]
            
            scaler.fit(train_data[numeric_columns])
            
            # Transform all data
            scaled_values = scaler.transform(data[numeric_columns])
            scaled_data = pd.DataFrame(scaled_values, columns=numeric_columns, index=data.index)
            
            # Add back non-numeric columns if any
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in non_numeric_columns:
                scaled_data[col] = data[col].values
            
            self.scaler = scaler
            self.feature_columns = numeric_columns
            
            logger.info(f"Successfully scaled {len(numeric_columns)} features")
            
            return scaled_data, scaler
            
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            raise
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with comprehensive validation."""
        try:
            logger.info("Creating sequences for LSTM training...")
            
            sequence_length = self.config.data_config['sequence_length']
            
            # Select features for modeling
            if self.config.data_config['technical_indicators']:
                # Use all available features
                feature_cols = [col for col in data.columns if col != 'Close']
                target_col = 'Close'
            else:
                # Use only OHLCV data
                feature_cols = ['Open', 'High', 'Low', 'Volume']
                target_col = 'Close'
            
            # Ensure all feature columns exist
            available_cols = [col for col in feature_cols if col in data.columns]
            if len(available_cols) != len(feature_cols):
                missing_cols = set(feature_cols) - set(available_cols)
                logger.warning(f"Missing columns: {missing_cols}. Using available columns: {available_cols}")
                feature_cols = available_cols
            
            features = data[feature_cols].values
            targets = data[target_col].values
            
            X, y = [], []
            
            for i in range(sequence_length, len(data)):
                X.append(features[i-sequence_length:i])
                y.append(targets[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            logger.info(f"Features used: {feature_cols}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets with temporal ordering."""
        try:
            logger.info("Splitting data into train/validation/test sets...")
            
            total_samples = len(X)
            train_size = int(total_samples * self.config.data_config['train_ratio'])
            val_size = int(total_samples * self.config.data_config['val_ratio'])
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def process_all(self) -> Dict[str, np.ndarray]:
        """Execute complete data processing pipeline."""
        try:
            logger.info("Starting complete data processing pipeline...")
            
            # Download data
            data = self.download_data()
            
            # Add technical indicators if requested
            if self.config.data_config['technical_indicators']:
                data = self.add_technical_indicators(data)
            
            # Clean data
            data = self.clean_data(data)
            
            # Scale features
            scaled_data, scaler = self.scale_features(data)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Store processed data
            self.processed_data = {
                'X_train': X_train,
                'X_val': X_val, 
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler,
                'feature_columns': self.feature_columns,
                'raw_data': data,
                'scaled_data': scaled_data
            }
            
            logger.info("Data processing pipeline completed successfully")
            
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

# ==============================================================================
# ATTENTION-ENHANCED LSTM MODEL
# ==============================================================================

class AttentionMechanism(nn.Module):
    """Advanced attention mechanism for LSTM networks."""
    
    def __init__(self, hidden_size: int, attention_size: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            attended_output: Weighted sum of LSTM outputs
            attention_weights: Attention weights for interpretability
        """
        # Calculate attention scores
        attention_scores = torch.tanh(self.attention_linear(lstm_outputs))
        attention_scores = self.context_vector(attention_scores).squeeze(2)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights to LSTM outputs
        attended_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(2), dim=1)
        
        return attended_output, attention_weights

class AttentionLSTM(nn.Module):
    """Professional Attention-Enhanced LSTM model for financial forecasting."""
    
    def __init__(self, config: Config):
        super(AttentionLSTM, self).__init__()
        self.config = config
        
        # Model parameters
        self.input_size = None  # Will be set based on data
        self.hidden_size = config.model_config['lstm_hidden_size']
        self.num_layers = config.model_config['lstm_num_layers']
        self.attention_size = config.model_config['attention_hidden_size']
        self.dropout_rate = config.model_config['dropout_rate']
        self.output_size = config.model_config['output_size']
        self.bidirectional = config.model_config['bidirectional']
        
        # Initialize layers (will be properly initialized in set_input_size)
        self.lstm = None
        self.attention = None
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = None
        
    def set_input_size(self, input_size: int):
        """Set input size and initialize layers."""
        self.input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # Adjust hidden size for bidirectional LSTM
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Attention mechanism
        self.attention = AttentionMechanism(lstm_output_size, self.attention_size)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(lstm_output_size // 2, self.output_size)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention-enhanced LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            output: Model predictions
            attention_weights: Attention weights for analysis
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(lstm_out)
        
        # Final prediction
        output = self.output_layer(attended_output)
        
        return output, attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'attention_size': self.attention_size,
            'bidirectional': self.bidirectional,
            'dropout_rate': self.dropout_rate
        }

class BaselineLSTM(nn.Module):
    """Baseline LSTM model for comparison (replicating Bhandari et al. 2022)."""
    
    def __init__(self, input_size: int, hidden_size: int = 150, dropout_rate: float = 0.2):
        super(BaselineLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Single LSTM layer as per Bhandari et al.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through baseline LSTM."""
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]
        
        # Apply dropout and generate output
        dropped = self.dropout(last_hidden)
        output = self.output_layer(dropped)
        
        return output

# ==============================================================================
# TRAINING AND OPTIMIZATION
# ==============================================================================

class EarlyStopping:
    """Early stopping implementation with comprehensive monitoring."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_weights_to_model(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class ModelTrainer:
    """Professional model training pipeline with comprehensive monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def create_data_loaders(self, data: Dict[str, np.ndarray]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders."""
        
        # Convert to tensors
        X_train = torch.FloatTensor(data['X_train']).to(self.device)
        y_train = torch.FloatTensor(data['y_train']).to(self.device)
        X_val = torch.FloatTensor(data['X_val']).to(self.device)
        y_val = torch.FloatTensor(data['y_val']).to(self.device)
        X_test = torch.FloatTensor(data['X_test']).to(self.device)
        y_test = torch.FloatTensor(data['y_test']).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        batch_size = self.config.training_config['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train model with comprehensive monitoring and early stopping."""
        
        # Optimizers and schedulers
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.training_config['learning_rate'],
            weight_decay=self.config.training_config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.training_config['scheduler_factor'],
            patience=self.config.training_config['scheduler_patience']
        )
        
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(
            patience=self.config.training_config['patience'],
            min_delta=self.config.training_config['min_delta']
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info("Starting model training...")
        
        for epoch in range(self.config.training_config['epochs']):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if isinstance(model, AttentionLSTM):
                    outputs, _ = model(batch_X)
                else:
                    outputs = model(batch_X)
                
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.training_config['gradient_clip_val']
                )
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if isinstance(model, AttentionLSTM):
                        outputs, _ = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_losses.append(loss.item())
            
            # Calculate epoch averages
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rate'].append(current_lr)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.training_config['epochs']}: "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                          f"LR: {current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(avg_val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                early_stopping.restore_best_weights_to_model(model)
                break
        
        logger.info("Training completed successfully")
        return history

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION
# ==============================================================================

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION (CONTINUED)
# ==============================================================================

class HyperparameterOptimizer:
    """Professional hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Config, data: Dict[str, np.ndarray]):
        self.config = config
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        
        # Suggest hyperparameters
        suggested_config = Config()
        suggested_config.data_config = self.config.data_config.copy()
        suggested_config.evaluation_config = self.config.evaluation_config.copy()
        
        # Model hyperparameters
        suggested_config.model_config = {
            'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [32, 64, 128, 256]),
            'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
            'attention_hidden_size': trial.suggest_categorical('attention_hidden_size', [16, 32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'output_size': 1,
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False])
        }
        
        # Training hyperparameters
        suggested_config.training_config = {
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50,  # Reduced for optimization
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'patience': 10,
            'min_delta': 1e-6,
            'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.5, 2.0),
            'scheduler_patience': 5,
            'scheduler_factor': 0.5
        }
        
        try:
            # Create model
            model = AttentionLSTM(suggested_config)
            input_size = self.data['X_train'].shape[2]
            model.set_input_size(input_size)
            model.to(self.device)
            
            # Create trainer
            trainer = ModelTrainer(suggested_config)
            train_loader, val_loader, _ = trainer.create_data_loaders(self.data)
            
            # Train model
            history = trainer.train_model(model, train_loader, val_loader)
            
            # Return best validation loss
            best_val_loss = min(history['val_loss'])
            
            return best_val_loss
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return float('inf')
    
    def optimize(self, n_trials: int = 100, timeout: int = 3600) -> optuna.Study:
        """Run hyperparameter optimization."""
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name='aex_lstm_optimization'
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Optimization completed. Best value: {study.best_value:.6f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study

# ==============================================================================
# EVALUATION AND METRICS
# ==============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with statistical testing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {}
        
        # Basic regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Directional accuracy
        y_true_direction = np.diff(y_true) > 0
        y_pred_direction = np.diff(y_pred) > 0
        metrics['directional_accuracy'] = np.mean(y_true_direction == y_pred_direction) * 100
        
        # Additional financial metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        # Correlation metrics
        metrics['pearson_correlation'] = stats.pearsonr(y_true, y_pred)[0]
        metrics['spearman_correlation'] = stats.spearmanr(y_true, y_pred)[0]
        
        return metrics
    
    def diebold_mariano_test(self, y_true: np.ndarray, pred1: np.ndarray, 
                           pred2: np.ndarray) -> Dict[str, float]:
        """Diebold-Mariano test for forecast comparison."""
        
        # Calculate loss differences
        loss1 = (y_true - pred1) ** 2
        loss2 = (y_true - pred2) ** 2
        loss_diff = loss1 - loss2
        
        # Calculate test statistic
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)
        n = len(loss_diff)
        
        if var_diff == 0:
            dm_stat = 0
            p_value = 1.0
        else:
            dm_stat = mean_diff / np.sqrt(var_diff / n)
            p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), n - 1))
        
        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                      data_type: str = "test") -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        model.eval()
        predictions = []
        actuals = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                if isinstance(model, AttentionLSTM):
                    batch_pred, attention_weights = model(batch_X)
                    attention_weights_list.append(attention_weights.cpu().numpy())
                else:
                    batch_pred = model(batch_X)
                
                predictions.extend(batch_pred.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(actuals, predictions)
        
        # Prepare results
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'data_type': data_type
        }
        
        # Add attention weights if available
        if attention_weights_list:
            results['attention_weights'] = np.concatenate(attention_weights_list, axis=0)
        
        logger.info(f"{data_type.capitalize()} Evaluation Results:")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return results

# ==============================================================================
# VISUALIZATION AND REPORTING
# ==============================================================================

class VisualizationEngine:
    """Professional visualization and reporting system."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: str = None) -> go.Figure:
        """Plot comprehensive training history."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Learning Rate Schedule',
                          'Loss Difference', 'Training Progress'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Training and validation loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Training Loss',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Learning rate schedule
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rate'], name='Learning Rate',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # Loss difference
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        fig.add_trace(
            go.Scatter(x=epochs, y=loss_diff, name='Val - Train Loss',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Training progress (smoothed validation loss)
        if len(history['val_loss']) > 10:
            smoothed_val = pd.Series(history['val_loss']).rolling(window=10).mean()
            fig.add_trace(
                go.Scatter(x=epochs, y=smoothed_val, name='Smoothed Val Loss',
                          line=dict(color='orange')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Model Training History",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_predictions(self, results: Dict[str, Any], 
                        save_path: str = None) -> go.Figure:
        """Plot model predictions vs actual values."""
        
        actuals = results['actuals']
        predictions = results['predictions']
        data_type = results['data_type']
        
        # Create time index
        time_index = pd.date_range(start='2023-01-01', periods=len(actuals), freq='D')
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{data_type.capitalize()} Set: Predictions vs Actuals',
                'Prediction Errors',
                'Error Distribution'
            ),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Predictions vs actuals
        fig.add_trace(
            go.Scatter(x=time_index, y=actuals, name='Actual',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_index, y=predictions, name='Predicted',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Prediction errors
        errors = actuals - predictions
        fig.add_trace(
            go.Scatter(x=time_index, y=errors, name='Prediction Error',
                      line=dict(color='green')),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=errors, name='Error Distribution', nbinsx=50,
                        marker_color='lightblue'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"Model Performance Analysis - {data_type.capitalize()} Set",
            height=1000,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_attention_weights(self, attention_weights: np.ndarray, 
                             sequence_length: int, save_path: str = None) -> go.Figure:
        """Visualize attention weights heatmap."""
        
        # Take sample of attention weights for visualization
        sample_size = min(100, attention_weights.shape[0])
        sample_weights = attention_weights[:sample_size]
        
        fig = go.Figure(data=go.Heatmap(
            z=sample_weights,
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title="Attention Weights Heatmap",
            xaxis_title="Time Steps",
            yaxis_title="Samples",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_comprehensive_report(self, attention_results: Dict[str, Any],
                                  baseline_results: Dict[str, Any],
                                  dm_test: Dict[str, float],
                                  save_dir: str = "reports") -> str:
        """Create comprehensive HTML report."""
        
        # Create reports directory
        report_dir = Path(save_dir)
        report_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"aex_forecasting_report_{timestamp}.html"
        
        # Extract metrics
        att_metrics = attention_results['metrics']
        base_metrics = baseline_results['metrics']
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AEX Index Forecasting - Comprehensive Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .improvement {{ color: green; font-weight: bold; }}
                .decline {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AEX Index Forecasting Report</h1>
                <h2>Attention-Enhanced LSTM vs Baseline LSTM</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive comparison between attention-enhanced LSTM and baseline LSTM models for AEX index forecasting.</p>
                
                <h3>Key Findings:</h3>
                <ul>
                    <li>Attention-Enhanced LSTM RMSE: {att_metrics['rmse']:.6f}</li>
                    <li>Baseline LSTM RMSE: {base_metrics['rmse']:.6f}</li>
                    <li>RMSE Improvement: {((base_metrics['rmse'] - att_metrics['rmse']) / base_metrics['rmse'] * 100):+.2f}%</li>
                    <li>Diebold-Mariano Test p-value: {dm_test['p_value']:.4f}</li>
                    <li>Statistical Significance: {"Yes" if dm_test['significant'] else "No"}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Detailed Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Attention-Enhanced LSTM</th>
                        <th>Baseline LSTM</th>
                        <th>Improvement</th>
                    </tr>
                    <tr>
                        <td>RMSE</td>
                        <td>{att_metrics['rmse']:.6f}</td>
                        <td>{base_metrics['rmse']:.6f}</td>
                        <td class="{'improvement' if att_metrics['rmse'] < base_metrics['rmse'] else 'decline'}">
                            {((base_metrics['rmse'] - att_metrics['rmse']) / base_metrics['rmse'] * 100):+.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>MAE</td>
                        <td>{att_metrics['mae']:.6f}</td>
                        <td>{base_metrics['mae']:.6f}</td>
                        <td class="{'improvement' if att_metrics['mae'] < base_metrics['mae'] else 'decline'}">
                            {((base_metrics['mae'] - att_metrics['mae']) / base_metrics['mae'] * 100):+.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>MAPE (%)</td>
                        <td>{att_metrics['mape']:.2f}</td>
                        <td>{base_metrics['mape']:.2f}</td>
                        <td class="{'improvement' if att_metrics['mape'] < base_metrics['mape'] else 'decline'}">
                            {((base_metrics['mape'] - att_metrics['mape']) / base_metrics['mape'] * 100):+.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>R²</td>
                        <td>{att_metrics['r2']:.4f}</td>
                        <td>{base_metrics['r2']:.4f}</td>
                        <td class="{'improvement' if att_metrics['r2'] > base_metrics['r2'] else 'decline'}">
                            {((att_metrics['r2'] - base_metrics['r2']) / abs(base_metrics['r2']) * 100):+.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>Directional Accuracy (%)</td>
                        <td>{att_metrics['directional_accuracy']:.2f}</td>
                        <td>{base_metrics['directional_accuracy']:.2f}</td>
                        <td class="{'improvement' if att_metrics['directional_accuracy'] > base_metrics['directional_accuracy'] else 'decline'}">
                            {(att_metrics['directional_accuracy'] - base_metrics['directional_accuracy']):+.2f}pp
                        </td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Significance Testing</h2>
                <h3>Diebold-Mariano Test Results</h3>
                <ul>
                    <li>Test Statistic: {dm_test['dm_statistic']:.4f}</li>
                    <li>P-value: {dm_test['p_value']:.4f}</li>
                    <li>Significant at 5% level: {"Yes" if dm_test['significant'] else "No"}</li>
                </ul>
                
                <p><strong>Interpretation:</strong> 
                {"The attention-enhanced LSTM shows statistically significant improvement over the baseline model." if dm_test['significant'] 
                 else "The improvement is not statistically significant at the 5% level."}
                </p>
            </div>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <h3>Attention-Enhanced LSTM:</h3>
                <ul>
                    <li>Hidden Size: {self.config.model_config['lstm_hidden_size']}</li>
                    <li>Number of Layers: {self.config.model_config['lstm_num_layers']}</li>
                    <li>Attention Size: {self.config.model_config['attention_hidden_size']}</li>
                    <li>Dropout Rate: {self.config.model_config['dropout_rate']}</li>
                </ul>
                
                <h3>Training Configuration:</h3>
                <ul>
                    <li>Batch Size: {self.config.training_config['batch_size']}</li>
                    <li>Learning Rate: {self.config.training_config['learning_rate']}</li>
                    <li>Sequence Length: {self.config.data_config['sequence_length']}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>
                This comprehensive analysis demonstrates {'significant' if dm_test['significant'] else 'modest'} 
                improvements in AEX index forecasting accuracy when using attention-enhanced LSTM networks 
                compared to traditional LSTM approaches. The attention mechanism provides 
                {'statistically significant' if dm_test['significant'] else 'measurable but not statistically significant'} 
                benefits in terms of prediction accuracy and model interpretability.
                </p>
                
                <h3>Key Takeaways:</h3>
                <ul>
                    <li>RMSE improvement of {((base_metrics['rmse'] - att_metrics['rmse']) / base_metrics['rmse'] * 100):.2f}%</li>
                    <li>Enhanced directional accuracy for trading applications</li>
                    <li>Improved model interpretability through attention weights</li>
                    <li>{'Statistically significant results support deployment for practical applications' if dm_test['significant'] 
                        else 'Results suggest potential but require further validation'}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        return str(report_path)

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

class AEXForecastingPipeline:
    """Main pipeline orchestrating the complete forecasting system."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_processor = None
        self.processed_data = None
        self.attention_model = None
        self.baseline_model = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = None
        
        # Create output directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary output directories."""
        directories = ['models', 'reports', 'figures', 'data', 'logs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
    def run_data_pipeline(self) -> Dict[str, np.ndarray]:
        """Execute data processing pipeline."""
        logger.info("=== STARTING DATA PROCESSING PIPELINE ===")
        
        self.data_processor = AEXDataProcessor(self.config)
        self.processed_data = self.data_processor.process_all()
        
        # Save processed data
        with open('data/processed_data.pkl', 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        logger.info("Data processing pipeline completed successfully")
        return self.processed_data
    
    def run_hyperparameter_optimization(self, n_trials: int = 50) -> optuna.Study:
        """Run hyperparameter optimization."""
        logger.info("=== STARTING HYPERPARAMETER OPTIMIZATION ===")
        
        optimizer = HyperparameterOptimizer(self.config, self.processed_data)
        study = optimizer.optimize(n_trials=n_trials)
        
        # Update config with best parameters
        self.config.model_config.update(study.best_params)
        self.config.training_config.update({k: v for k, v in study.best_params.items() 
                                          if k in self.config.training_config})
        
        # Save optimization results
        with open('reports/optimization_study.pkl', 'wb') as f:
            pickle.dump(study, f)
        
        logger.info("Hyperparameter optimization completed successfully")
        return study
    
    def train_models(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Train both attention-enhanced and baseline models."""
        logger.info("=== STARTING MODEL TRAINING ===")
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.config)
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(self.processed_data)
        
        # Initialize models
        input_size = self.processed_data['X_train'].shape[2]
        
        # Attention-enhanced model
        self.attention_model = AttentionLSTM(self.config)
        self.attention_model.set_input_size(input_size)
        self.attention_model.to(self.trainer.device)
        
        # Baseline model
        self.baseline_model = BaselineLSTM(input_size)
        self.baseline_model.to(self.trainer.device)
        
        logger.info(f"Attention Model Info: {self.attention_model.get_model_info()}")
        
        # Train attention model
        logger.info("Training attention-enhanced LSTM...")
        attention_history = self.trainer.train_model(self.attention_model, train_loader, val_loader)
        
        # Train baseline model
        logger.info("Training baseline LSTM...")
        baseline_history = self.trainer.train_model(self.baseline_model, train_loader, val_loader)
        
        # Save models
        torch.save({
            'model_state_dict': self.attention_model.state_dict(),
            'config': self.config,
            'history': attention_history
        }, 'models/attention_lstm_model.pth')
        
        torch.save({
            'model_state_dict': self.baseline_model.state_dict(),
            'history': baseline_history
        }, 'models/baseline_lstm_model.pth')
        
        logger.info("Model training completed successfully")
        return attention_history, baseline_history
    
    def evaluate_models(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """Comprehensive model evaluation."""
        logger.info("=== STARTING MODEL EVALUATION ===")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.config)
        
        # Create data loaders
        _, _, test_loader = self.trainer.create_data_loaders(self.processed_data)
        
        # Evaluate models
        attention_results = self.evaluator.evaluate_model(self.attention_model, test_loader, "test")
        baseline_results = self.evaluator.evaluate_model(self.baseline_model, test_loader, "test")
        
        # Statistical testing
        dm_test = self.evaluator.diebold_mariano_test(
            attention_results['actuals'],
            attention_results['predictions'],
            baseline_results['predictions']
        )
        
        logger.info("=== STATISTICAL TESTING RESULTS ===")
        logger.info(f"Diebold-Mariano Test - Statistic: {dm_test['dm_statistic']:.4f}, "
                   f"P-value: {dm_test['p_value']:.4f}, "
                   f"Significant: {dm_test['significant']}")
        
        # Save evaluation results
        evaluation_results = {
            'attention_results': attention_results,
            'baseline_results': baseline_results,
            'dm_test': dm_test
        }
        
        with open('reports/evaluation_results.pkl', 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        logger.info("Model evaluation completed successfully")
        return attention_results, baseline_results, dm_test
    
    def generate_visualizations(self, attention_history: Dict[str, List[float]],
                              baseline_history: Dict[str, List[float]],
                              attention_results: Dict[str, Any],
                              baseline_results: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        logger.info("=== GENERATING VISUALIZATIONS ===")
        
        self.visualizer = VisualizationEngine(self.config)
        
        # Training history plots
        attention_fig = self.visualizer.plot_training_history(
            attention_history, 'figures/attention_training_history.html'
        )
        
        baseline_fig = self.visualizer.plot_training_history(
            baseline_history, 'figures/baseline_training_history.html'
        )
        
        # Prediction plots
        attention_pred_fig = self.visualizer.plot_predictions(
            attention_results, 'figures/attention_predictions.html'
        )
        
        baseline_pred_fig = self.visualizer.plot_predictions(
            baseline_results, 'figures/baseline_predictions.html'
        )
        
        # Attention weights visualization
        if 'attention_weights' in attention_results:
            attention_weights_fig = self.visualizer.plot_attention_weights(
                attention_results['attention_weights'],
                self.config.data_config['sequence_length'],
                'figures/attention_weights.html'
            )
        
        logger.info("Visualizations generated successfully")
    
    def run_complete_pipeline(self, optimize_hyperparameters: bool = True, 
                            n_optimization_trials: int = 50) -> Dict[str, Any]:
        """Execute the complete forecasting pipeline."""
        logger.info("=== STARTING COMPLETE AEX FORECASTING PIPELINE ===")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Processing
            processed_data = self.run_data_pipeline()
            
            # Step 2: Hyperparameter Optimization (optional)
            if optimize_hyperparameters:
                study = self.run_hyperparameter_optimization(n_optimization_trials)
            else:
                study = None
            
            # Step 3: Model Training
            attention_history, baseline_history = self.train_models()
            
            # Step 4: Model Evaluation
            attention_results, baseline_results, dm_test = self.evaluate_models()
            
            # Step 5: Generate Visualizations
            self.generate_visualizations(
                attention_history, baseline_history,
                attention_results, baseline_results
            )
            
            # Step 6: Generate Comprehensive Report
            report_path = self.visualizer.create_comprehensive_report(
                attention_results, baseline_results, dm_test
            )
            
            # Calculate execution time
            execution_time = datetime.now() - start_time
            
            # Compile final results
            final_results = {
                'execution_time': str(execution_time),
                'attention_model_metrics': attention_results['metrics'],
                'baseline_model_metrics': baseline_results['metrics'],
                'statistical_test': dm_test,
                'optimization_study': study,
                'report_path': report_path,
                'model_paths': {
                    'attention_model': 'models/attention_lstm_model.pth',
                    'baseline_model': 'models/baseline_lstm_model.pth'
                }
            }
            
            # Helper function to convert numpy.float32 to Python float for JSON serialization
            def convert_numpy_floats(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_floats(i) for i in obj]
                elif isinstance(obj, np.float32):
                    return float(obj)
                return obj

            data_to_serialize = {}
            for k, v in final_results.items():
                if k == 'optimization_study': # Exclude optuna.Study object
                    continue
                data_to_serialize[k] = convert_numpy_floats(v)

            # Save final results summary
            with open('reports/final_results_summary.json', 'w') as f:
                json.dump(data_to_serialize, f, indent=2)
            
            logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total execution time: {execution_time}")
            logger.info(f"Comprehensive report available at: {report_path}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

# ==============================================================================
# DEPLOYMENT AND INFERENCE UTILITIES
# ==============================================================================

class ModelInference:
    """Production-ready model inference class."""
    
    def __init__(self, model_path: str, scaler_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.config = None
        
        self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract config and create model
            self.config = checkpoint['config']
            self.model = AttentionLSTM(self.config)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """Load fitted scaler."""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def preprocess_input(self, data: np.ndarray) -> torch.Tensor:
        """Preprocess input data for inference."""
        if self.scaler:
            data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(self.device)
        return tensor_data
    
    def predict(self, input_sequence: np.ndarray, return_attention: bool = False) -> Dict[str, Any]:
        """Make prediction on input sequence."""
        try:
            # Preprocess input
            processed_input = self.preprocess_input(input_sequence)
            
            # Make prediction
            with torch.no_grad():
                if isinstance(self.model, AttentionLSTM):
                    prediction, attention_weights = self.model(processed_input)
                else:
                    prediction = self.model(processed_input)
                    attention_weights = None
            
            # Process output
            prediction_value = prediction.cpu().numpy().item()
            
            # Inverse transform if scaler available
            if self.scaler:
                prediction_value = self.scaler.inverse_transform([[prediction_value]])[0][0]
            
            results = {
                'prediction': prediction_value,
                'confidence_interval': None  # Could be added with ensemble methods
            }
            
            if return_attention and attention_weights is not None:
                results['attention_weights'] = attention_weights.cpu().numpy()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def batch_predict(self, input_sequences: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Make predictions on multiple sequences."""
        predictions = []
        for sequence in input_sequences:
            pred = self.predict(sequence)
            predictions.append(pred)
        return predictions

# ==============================================================================
# CONFIGURATION MANAGEMENT AND UTILITIES
# ==============================================================================

class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def save_config(config: Config, filepath: str):
        """Save configuration to file."""
        config_dict = {
            'data_config': config.data_config,
            'model_config': config.model_config,
            'training_config': config.training_config,
            'evaluation_config': config.evaluation_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> Config:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = Config()
        config.data_config = config_dict['data_config']
        config.model_config = config_dict['model_config']
        config.training_config = config_dict['training_config']
        config.evaluation_config = config_dict['evaluation_config']
        
        return config
    
    @staticmethod
    def create_experiment_config(experiment_name: str) -> Config:
        """Create configuration for specific experiments."""
        
        if experiment_name == "quick_test":
            config = Config()
            config.data_config['start_date'] = '2020-01-01'
            config.training_config['epochs'] = 20
            config.training_config['batch_size'] = 64
            return config
            
        elif experiment_name == "production":
            config = Config()
            config.training_config['epochs'] = 200
            config.training_config['patience'] = 20
            return config
            
        else:
            return Config()

# ==============================================================================
# TESTING AND VALIDATION UTILITIES
# ==============================================================================

class ModelValidator:
    """Comprehensive model validation utilities."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def validate_model_architecture(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate model architecture and compute statistics."""
        
        validation_results = {}
        
        # Test forward pass
        try:
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            model.to(self.device)
            
            if isinstance(model, AttentionLSTM):
                output, attention = model(dummy_input)
                validation_results['attention_shape'] = attention.shape
            else:
                output = model(dummy_input)
            
            validation_results['output_shape'] = output.shape
            validation_results['forward_pass'] = "SUCCESS"
            
        except Exception as e:
            validation_results['forward_pass'] = f"FAILED: {str(e)}"
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        validation_results.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        })
        
        return validation_results
    
    def validate_data_pipeline(self, processor: AEXDataProcessor) -> Dict[str, Any]:
        """Validate data processing pipeline."""
        
        validation_results = {}
        
        try:
            # Test data download
            data = processor.download_data()
            validation_results['data_download'] = f"SUCCESS: {len(data)} records"
            
            # Test technical indicators
            if processor.config.data_config['technical_indicators']:
                data_with_indicators = processor.add_technical_indicators(data)
                validation_results['technical_indicators'] = f"SUCCESS: {len(data_with_indicators.columns)} features"
            
            # Test data cleaning
            cleaned_data = processor.clean_data(data)
            validation_results['data_cleaning'] = f"SUCCESS: {len(cleaned_data)} records retained"
            
            # Test scaling
            scaled_data, scaler = processor.scale_features(cleaned_data)
            validation_results['feature_scaling'] = "SUCCESS"
            
            # Test sequence creation
            X, y = processor.create_sequences(scaled_data)
            validation_results['sequence_creation'] = f"SUCCESS: {len(X)} sequences created"
            
        except Exception as e:
            validation_results['pipeline_error'] = str(e)
        
        return validation_results

# ==============================================================================
# PERFORMANCE MONITORING AND LOGGING
# ==============================================================================

class PerformanceMonitor:
    """Monitor and log system performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = datetime.now()
        
    def log_memory_usage(self):
        """Log current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.metrics['memory_rss_mb'] = memory_info.rss / (1024 * 1024)
        self.metrics['memory_vms_mb'] = memory_info.vms / (1024 * 1024)
        
    def log_gpu_usage(self):
        """Log GPU usage if available."""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            self.metrics['gpu_memory_allocated_mb'] = gpu_memory_allocated
            self.metrics['gpu_memory_reserved_mb'] = gpu_memory_reserved
        
    def end_monitoring(self):
        """End monitoring and calculate total time."""
        if self.start_time:
            end_time = datetime.now()
            self.metrics['total_execution_time'] = str(end_time - self.start_time)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()

# ==============================================================================
# MAIN EXECUTION FUNCTIONS
# ==============================================================================

def run_quick_test():
    """Run a quick test of the system."""
    logger.info("=== RUNNING QUICK TEST ===")
    
    # Create test configuration
    config = ConfigManager.create_experiment_config("quick_test")
    
    # Initialize pipeline
    pipeline = AEXForecastingPipeline(config)
    
    # Run pipeline without hyperparameter optimization
    results = pipeline.run_complete_pipeline(
        optimize_hyperparameters=False,
        n_optimization_trials=0
    )
    
    logger.info("Quick test completed successfully!")
    return results

def run_full_experiment():
    """Run the complete experimental pipeline."""
    logger.info("=== RUNNING FULL EXPERIMENT ===")
    
    # Create production configuration
    config = Config()
    
    # Initialize pipeline
    pipeline = AEXForecastingPipeline(config)
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # Run complete pipeline with hyperparameter optimization
        results = pipeline.run_complete_pipeline(
            optimize_hyperparameters=True,
            n_optimization_trials=100
        )
        
        # End monitoring
        monitor.end_monitoring()
        monitor.log_memory_usage()
        monitor.log_gpu_usage()
        
        # Save performance metrics
        performance_metrics = monitor.get_metrics()
        with open('reports/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        logger.info("Full experiment completed successfully!")
        logger.info(f"Performance metrics: {performance_metrics}")
        
        return results
        
    except Exception as e:
        logger.error(f"Full experiment failed: {str(e)}")
        raise

def validate_system():
    """Validate the complete system."""
    logger.info("=== RUNNING SYSTEM VALIDATION ===")
    
    validator = ModelValidator()
    
    # Validate data pipeline
    config = Config()
    processor = AEXDataProcessor(config)
    data_validation = validator.validate_data_pipeline(processor)
    
    logger.info("Data Pipeline Validation:")
    for key, value in data_validation.items():
        logger.info(f"  {key}: {value}")
    
    # Validate model architectures
    input_shape = (60, 15)  # sequence_length, num_features
    
    # Test attention model
    attention_model = AttentionLSTM(config)
    attention_model.set_input_size(input_shape[1])
    attention_validation = validator.validate_model_architecture(attention_model, input_shape)
    
    logger.info("Attention Model Validation:")
    for key, value in attention_validation.items():
        logger.info(f"  {key}: {value}")
    
    # Test baseline model
    baseline_model = BaselineLSTM(input_shape[1])
    baseline_validation = validator.validate_model_architecture(baseline_model, input_shape)
    
    logger.info("Baseline Model Validation:")
    for key, value in baseline_validation.items():
        logger.info(f"  {key}: {value}")
    
    return {
        'data_pipeline': data_validation,
        'attention_model': attention_validation,
        'baseline_model': baseline_validation
    }

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AEX Index Forecasting with Attention-Enhanced LSTM')
    parser.add_argument('--mode', choices=['quick', 'full', 'validate', 'inference'], 
                       default='quick', help='Execution mode')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model-path', type=str, help='Path to trained model for inference')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'validate':
            results = validate_system()
            
        elif args.mode == 'quick':
            results = run_quick_test()
            
        elif args.mode == 'full':
            results = run_full_experiment()
            
        elif args.mode == 'inference':
            if not args.model_path:
                logger.error("Model path required for inference mode")
                return
            
            # Initialize inference engine
            inference_engine = ModelInference(args.model_path)
            
            # Example inference (you would replace this with actual data)
            dummy_sequence = np.random.randn(60)  # 60-day sequence
            prediction = inference_engine.predict(dummy_sequence, return_attention=True)
            
            logger.info(f"Prediction: {prediction['prediction']:.6f}")
            if 'attention_weights' in prediction:
                logger.info(f"Attention weights shape: {prediction['attention_weights'].shape}")
            
            results = prediction
        
        logger.info("Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Run main function
    main()

# ==============================================================================
# EXAMPLE USAGE SCRIPTS
# ==============================================================================

def example_quick_start():
    """Example of how to use the system for quick prototyping."""
    
    # Initialize with default configuration
    config = Config()
    
    # Quick modifications for testing
    config.data_config['start_date'] = '2022-01-01'
    config.training_config['epochs'] = 10
    
    # Create and run pipeline
    pipeline = AEXForecastingPipeline(config)
    results = pipeline.run_complete_pipeline(optimize_hyperparameters=False)
    
    print(f"Quick test completed!")
    print(f"Attention Model RMSE: {results['attention_model_metrics']['rmse']:.6f}")
    print(f"Baseline Model RMSE: {results['baseline_model_metrics']['rmse']:.6f}")
    print(f"Report saved to: {results['report_path']}")

def example_custom_configuration():
    """Example of custom configuration usage."""
    
    # Create custom configuration
    custom_config = Config()
    
    # Modify data configuration
    custom_config.data_config.update({
        'sequence_length': 90,  # Longer sequences
        'train_ratio': 0.8,     # More training data
        'technical_indicators': True
    })
    
    # Modify model configuration
    custom_config.model_config.update({
        'lstm_hidden_size': 128,
        'lstm_num_layers': 3,
        'attention_hidden_size': 64,
        'dropout_rate': 0.3
    })
    
    # Save configuration
    ConfigManager.save_config(custom_config, 'config/custom_config.json')
    
    # Use configuration
    pipeline = AEXForecastingPipeline(custom_config)
    results = pipeline.run_complete_pipeline()
    
    return results

def example_model_inference():
    """Example of using trained model for inference."""
    
    # Load trained model
    model_path = 'models/attention_lstm_model.pth'
    inference_engine = ModelInference(model_path)
    
    # Simulate new data (in practice, this would be real market data)
    new_sequence = np.random.randn(60)  # 60-day price sequence
    
    # Make prediction
    prediction = inference_engine.predict(new_sequence, return_attention=True)
    
    print(f"Predicted next-day price: {prediction['prediction']:.2f}")
    if 'attention_weights' in prediction:
        print(f"Attention focused on recent days: {np.argmax(prediction['attention_weights'])}")
    
    return prediction

# ==============================================================================
# DOCUMENTATION AND HELP
# ==============================================================================

__doc__ = """
AEX Index Forecasting with Attention-Enhanced LSTM Networks

This comprehensive system provides a professional implementation for forecasting
the AEX (Amsterdam Exchange Index) using state-of-the-art attention-enhanced 
LSTM networks compared against baseline LSTM models.

Features:
- Professional data processing pipeline with technical indicators
- Attention-enhanced LSTM model with interpretable attention weights
- Comprehensive hyperparameter optimization using Optuna
- Statistical significance testing (Diebold-Mariano test)
- Extensive evaluation metrics and visualizations
- Production-ready inference engine
- Comprehensive reporting and documentation

Usage Examples:

1. Quick Test:
   python aex_forecasting.py --mode quick

2. Full Experiment with Optimization:
   python aex_forecasting.py --mode full --optimize --trials 100

3. System Validation:
   python aex_forecasting.py --mode validate

4. Model Inference:
   python aex_forecasting.py --mode inference --model-path models/attention_lstm_model.pth

Classes:
- Config: Comprehensive configuration management
- AEXDataProcessor: Professional data processing pipeline
- AttentionLSTM: Attention-enhanced LSTM model
- BaselineLSTM: Baseline LSTM for comparison
- ModelTrainer: Training pipeline with early stopping
- ModelEvaluator: Comprehensive evaluation and testing
- VisualizationEngine: Professional visualizations and reporting
- AEXForecastingPipeline: Main orchestration pipeline

For academic research, this implementation follows best practices for:
- Reproducible research (fixed random seeds, comprehensive logging)
- Statistical rigor (proper train/val/test splits, significance testing)
- Professional documentation (comprehensive reporting, code documentation)
- Academic standards (following established methodologies from literature)

Author: Professional Implementation for Academic Research
Version: 1.0.0
License: Academic Use
"""

def print_system_info():
    """Print comprehensive system information."""
    
    print("=" * 80)
    print("AEX INDEX FORECASTING SYSTEM")
    print("Attention-Enhanced LSTM Networks")
    print("=" * 80)
    print()
    print("System Information:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name()}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    print("Features:")
    print("  ✓ Professional data processing pipeline")
    print("  ✓ Attention-enhanced LSTM implementation")
    print("  ✓ Hyperparameter optimization with Optuna")
    print("  ✓ Statistical significance testing")
    print("  ✓ Comprehensive evaluation metrics")
    print("  ✓ Production-ready inference engine")
    print("  ✓ Professional reporting and visualization")
    print()
    print("Usage:")
    print("  Quick Test:     python aex_forecasting.py --mode quick")
    print("  Full Pipeline:  python aex_forecasting.py --mode full --optimize")
    print("  Validation:     python aex_forecasting.py --mode validate")
    print("  Inference:      python aex_forecasting.py --mode inference --model-path <path>")
    print("=" * 80)

# Print system information when module is imported
if __name__ == "__main__":
    print_system_info()
