import torch
import torch.nn as nn
from typing import Tuple, Any, Dict
from config import Config

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
