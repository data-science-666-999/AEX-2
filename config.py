from dataclasses import dataclass
from typing import Dict
import json

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
