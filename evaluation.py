import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging
from typing import Dict, Any

from config import Config
from model import AttentionLSTM
from logger import setup_logging

# Initialize logger for this module
logger = setup_logging()

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
