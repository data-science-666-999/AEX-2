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

from config import Config, ConfigManager

# Initialize configuration
config = Config()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

from logger import setup_logging

logger = setup_logging()

# ==============================================================================
# DATA COLLECTION AND PREPROCESSING
# ==============================================================================

from data_processing import AEXDataProcessor

# ==============================================================================
# ATTENTION-ENHANCED LSTM MODEL
# ==============================================================================

from model import AttentionMechanism, AttentionLSTM, BaselineLSTM

# ==============================================================================
# TRAINING AND OPTIMIZATION
# ==============================================================================

from training import EarlyStopping, ModelTrainer

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION
# ==============================================================================

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION (CONTINUED)
# ==============================================================================

from optimization import HyperparameterOptimizer

# ==============================================================================
# EVALUATION AND METRICS
# ==============================================================================

from evaluation import ModelEvaluator

# ==============================================================================
# VISUALIZATION AND REPORTING
# ==============================================================================

from visualization import VisualizationEngine

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

from pipeline import AEXForecastingPipeline

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
