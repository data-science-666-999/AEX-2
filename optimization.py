import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from typing import Dict

from config import Config
from logger_setup import logger
from models import AttentionLSTM
from training import ModelTrainer

class HyperparameterOptimizer:
    """Professional hyperparameter optimization using Optuna."""

    def __init__(self, config: Config, data: Dict[str, torch.Tensor]): # Assuming data is Dict of Tensors
        self.config = config
        self.data = data # This should be the processed_data dictionary containing numpy arrays
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""

        # Suggest hyperparameters
        suggested_config = Config()
        # Deepcopy or careful copy of nested dicts might be needed if original config can be modified
        suggested_config.data_config = self.config.data_config.copy()
        suggested_config.evaluation_config = self.config.evaluation_config.copy()

        # Model hyperparameters
        suggested_config.model_config = {
            'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [32, 64, 128, 256]),
            'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
            'attention_hidden_size': trial.suggest_categorical('attention_hidden_size', [16, 32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'output_size': 1, # Assuming this is fixed
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False])
        }

        # Training hyperparameters
        suggested_config.training_config = {
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50,  # Reduced for optimization, can be part of config
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'patience': 10, # Fixed for HPO, or could be tuned
            'min_delta': 1e-6, # Fixed for HPO
            'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.5, 2.0),
            'scheduler_patience': 5, # Fixed for HPO
            'scheduler_factor': 0.5 # Fixed for HPO
        }

        try:
            # Create model
            model = AttentionLSTM(suggested_config)
            # Data should be Dict[str, np.ndarray] as per AEXDataProcessor output
            # X_train is np.ndarray, shape[2] gives num_features
            input_size = self.data['X_train'].shape[2]
            model.set_input_size(input_size)
            model.to(self.device)

            # Create trainer
            trainer = ModelTrainer(suggested_config) # ModelTrainer uses the suggested_config
            # create_data_loaders expects Dict[str, np.ndarray]
            train_loader, val_loader, _ = trainer.create_data_loaders(self.data)

            # Train model
            history = trainer.train_model(model, train_loader, val_loader)

            # Return best validation loss
            best_val_loss = min(history['val_loss'])

            return best_val_loss

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            # Optuna handles exceptions and may prune or fail the trial.
            # Returning float('inf') is a common way to indicate failure.
            return float('inf')

    def optimize(self, n_trials: int = 100, timeout: int = 3600) -> optuna.Study:
        """Run hyperparameter optimization."""

        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        # Create study
        sampler = TPESampler(seed=42) # Make seed configurable if needed
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10) # Params can be part of config

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name='aex_lstm_optimization' # Can be made dynamic
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout, # In seconds
            show_progress_bar=True # Optuna controls this, might not show in all environments
        )

        logger.info(f"Optimization completed. Best value: {study.best_value:.6f}")
        logger.info(f"Best parameters: {study.best_params}")

        return study
