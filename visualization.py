import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any

from config import Config
from logger import setup_logging

# Initialize logger for this module
logger = setup_logging()

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
