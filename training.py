import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

from config import Config
from logger_setup import logger
from models import AttentionLSTM # Ensure BaselineLSTM is not explicitly needed here, or add if it is.

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
            patience=self.config.training_config['scheduler_patience'],
            verbose=True
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

                if isinstance(model, AttentionLSTM): # Check if model is AttentionLSTM
                    outputs, _ = model(batch_X)
                else: # For BaselineLSTM or other models not returning attention
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
                    if isinstance(model, AttentionLSTM): # Check if model is AttentionLSTM
                        outputs, _ = model(batch_X)
                    else: # For BaselineLSTM or other models
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
