import optuna
import torch
import logging
from typing import Dict, Any
import numpy as np

from config import Config
from model import AttentionLSTM
from training import ModelTrainer
from logger import setup_logging
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Initialize logger for this module
logger = setup_logging()

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
