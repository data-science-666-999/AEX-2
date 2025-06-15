import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime # Explicitly import datetime from datetime
from pathlib import Path
from typing import Dict, List, Any

from config import Config
from logger_setup import logger

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
                          'Loss Difference (Val - Train)', 'Smoothed Validation Loss'), # Updated titles
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
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log") # Log scale for LR often useful

        # Loss difference
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        fig.add_trace(
            go.Scatter(x=epochs, y=loss_diff, name='Val - Train Loss',
                      line=dict(color='purple')),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Loss Difference", row=2, col=1)

        # Training progress (smoothed validation loss)
        if len(history['val_loss']) >= 5: # Ensure enough points for smoothing
            smoothed_val = pd.Series(history['val_loss']).rolling(window=5, min_periods=1).mean() # Use min_periods
            fig.add_trace(
                go.Scatter(x=epochs, y=smoothed_val, name='Smoothed Val Loss (5-epoch MA)',
                          line=dict(color='orange')),
                row=2, col=2
            )
        fig.update_yaxes(title_text="Smoothed Loss", row=2, col=2)

        fig.update_layout(
            title_text="Model Training History", # Changed from title to title_text
            height=800,
            showlegend=True,
            legend_traceorder="reversed" # Show legend in a consistent order
        )

        if save_path:
            try:
                fig.write_html(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving training history plot: {e}")

        return fig

    def plot_predictions(self, results: Dict[str, Any],
                        save_path: str = None) -> go.Figure:
        """Plot model predictions vs actual values."""

        actuals = results['actuals']
        predictions = results['predictions']
        data_type = results['data_type']

        # Create time index if not available (e.g. use simple range)
        # This part might need adjustment based on how raw_data / time_index is handled in the main pipeline
        if 'time_index' in results and results['time_index'] is not None and len(results['time_index']) == len(actuals):
            time_index = results['time_index']
        else:
            logger.warning("Time index not found or mismatched in results, using default range index for plotting.")
            time_index = list(range(len(actuals)))

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{data_type.capitalize()} Set: Predictions vs Actuals',
                'Prediction Errors (Actual - Predicted)',
                'Error Distribution (Histogram)'
            ),
            vertical_spacing=0.08 # Adjust vertical spacing
        )

        # Predictions vs actuals
        fig.add_trace(
            go.Scatter(x=time_index, y=actuals, name='Actual Values',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_index, y=predictions, name='Predicted Values',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )

        # Prediction errors
        errors = actuals - predictions
        fig.add_trace(
            go.Scatter(x=time_index, y=errors, name='Prediction Error',
                      line=dict(color='green', width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=2, col=1)

        # Error distribution
        fig.add_trace(
            go.Histogram(x=errors, name='Error Distribution', nbinsx=50,
                        marker_color='lightblue', opacity=0.75),
            row=3, col=1
        )

        fig.update_layout(
            title_text=f"Model Performance Analysis - {data_type.capitalize()} Set",
            height=1000, # Adjusted height
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if save_path:
            try:
                fig.write_html(save_path)
                logger.info(f"Predictions plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving predictions plot: {e}")

        return fig

    def plot_attention_weights(self, attention_weights: np.ndarray,
                             sequence_length: int, save_path: str = None) -> go.Figure:
        """Visualize attention weights heatmap."""

        # Take sample of attention weights for visualization (e.g., first 100 or all if less)
        sample_size = min(100, attention_weights.shape[0])
        sample_weights = attention_weights[:sample_size]

        fig = go.Figure(data=go.Heatmap(
            z=sample_weights,
            x=[f'Time Step {i+1}' for i in range(sequence_length)], # Label x-axis
            y=[f'Sample {i+1}' for i in range(sample_size)], # Label y-axis
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight")
        ))

        fig.update_layout(
            title_text="Attention Weights Heatmap (Sampled)",
            xaxis_title="Input Sequence Time Steps (Attention Applied To)",
            yaxis_title="Output Samples (Prediction For)",
            height=max(400, sample_size * 8), # Adjust height based on sample size
            width=max(600, sequence_length * 15) # Adjust width based on sequence length
        )

        if save_path:
            try:
                fig.write_html(save_path)
                logger.info(f"Attention weights plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving attention weights plot: {e}")

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
        report_filename = f"aex_forecasting_report_{timestamp}.html"
        report_path = report_dir / report_filename

        # Extract metrics
        att_metrics = attention_results['metrics']
        base_metrics = baseline_results['metrics']

        # Helper to format metric values
        def format_metric(value, precision=4, is_percent=False):
            if isinstance(value, (int, float)):
                return f"{value:.{precision}f}{'%' if is_percent else ''}"
            return str(value)

        # Helper to calculate improvement and style it
        def improvement_style(att_val, base_val, higher_is_better=True):
            if not isinstance(att_val, (int, float)) or not isinstance(base_val, (int, float)) or base_val == 0:
                return "N/A", "" # Cannot calculate or meaningless

            improvement = ((att_val - base_val) / abs(base_val)) * 100
            if not higher_is_better:
                improvement *= -1 # Lower is better (e.g. RMSE, MAE, MAPE)

            style_class = "improvement" if improvement > 0 else ("decline" if improvement < 0 else "neutral")
            return f"{improvement:+.2f}%", style_class

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>AEX Index Forecasting - Comprehensive Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: #005A9C; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0;}}
                .header h1 {{ margin: 0; }}
                .section {{ margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
                .section:last-child {{ border-bottom: none; }}
                .section h2 {{ color: #005A9C; border-bottom: 2px solid #005A9C; padding-bottom: 10px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                .metrics-table th {{ background-color: #f7f7f7; font-weight: bold; }}
                .improvement {{ color: green; }}
                .decline {{ color: red; }}
                .neutral {{ color: dimgray; }}
                ul {{ list-style-type: disc; margin-left: 20px; }}
                .config-details ul li {{ white-space: nowrap; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AEX Index Forecasting Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="section">
                    <h2>Executive Summary</h2>
                    <p>This report compares the performance of an Attention-Enhanced LSTM model against a Baseline LSTM model for forecasting the AEX index.</p>
                    <p><strong>Key Finding:</strong> The Attention-Enhanced LSTM model showed an RMSE of <strong>{format_metric(att_metrics.get('rmse', 'N/A'))}</strong> compared to the Baseline LSTM's RMSE of <strong>{format_metric(base_metrics.get('rmse', 'N/A'))}</strong>.
                       The Diebold-Mariano test yielded a p-value of <strong>{format_metric(dm_test.get('p_value', 'N/A'), 4)}</strong>, indicating that the difference in predictive accuracy is
                       <strong>{'statistically significant' if dm_test.get('significant', False) else 'not statistically significant'}</strong> at the chosen confidence level.</p>
                </div>

                <div class="section">
                    <h2>Detailed Performance Metrics</h2>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Attention-Enhanced LSTM</th>
                                <th>Baseline LSTM</th>
                                <th>Improvement (Attn vs Base)</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        metrics_to_report = [
            ('rmse', 'RMSE', 4, False), ('mae', 'MAE', 4, False), ('mape', 'MAPE (%)', 2, False),
            ('r2', 'R²', 4, True), ('directional_accuracy', 'Directional Accuracy (%)', 2, True),
            ('pearson_correlation', 'Pearson Correlation', 4, True),
            ('spearman_correlation', 'Spearman Correlation', 4, True)
        ]

        for key, name, prec, higher_is_better in metrics_to_report:
            att_val = att_metrics.get(key)
            base_val = base_metrics.get(key)
            imp_str, imp_class = improvement_style(att_val, base_val, higher_is_better)
            html_content += f"""
                            <tr>
                                <td>{name}</td>
                                <td>{format_metric(att_val, prec, '%' in name)}</td>
                                <td>{format_metric(base_val, prec, '%' in name)}</td>
                                <td class="{imp_class}">{imp_str}</td>
                            </tr>
            """
        html_content += f"""
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>Statistical Significance (Diebold-Mariano Test)</h2>
                    <ul>
                        <li>Test Statistic: {format_metric(dm_test.get('dm_statistic', 'N/A'), 4)}</li>
                        <li>P-value: {format_metric(dm_test.get('p_value', 'N/A'), 4)}</li>
                        <li>Significant at {(1.0 - self.config.evaluation_config.get('confidence_level', 0.95)) * 100}% level:
                            <strong>{'Yes' if dm_test.get('significant', False) else 'No'}</strong></li>
                    </ul>
                </div>

                <div class="section config-details">
                    <h2>Model Configuration Highlights</h2>
                    <p><strong>Attention-Enhanced LSTM:</strong></p>
                    <ul>
                        <li>LSTM Hidden Size: {self.config.model_config.get('lstm_hidden_size', 'N/A')}</li>
                        <li>LSTM Layers: {self.config.model_config.get('lstm_num_layers', 'N/A')}</li>
                        <li>Attention Hidden Size: {self.config.model_config.get('attention_hidden_size', 'N/A')}</li>
                        <li>Dropout: {self.config.model_config.get('dropout_rate', 'N/A')}</li>
                        <li>Bidirectional: {self.config.model_config.get('bidirectional', 'N/A')}</li>
                    </ul>
                    <p><strong>Training:</strong></p>
                    <ul>
                        <li>Batch Size: {self.config.training_config.get('batch_size', 'N/A')}</li>
                        <li>Learning Rate: {format_metric(self.config.training_config.get('learning_rate', 'N/A'), 5)}</li>
                        <li>Epochs: {self.config.training_config.get('epochs', 'N/A')}</li>
                    </ul>
                     <p><strong>Data:</strong></p>
                    <ul>
                        <li>Sequence Length: {self.config.data_config.get('sequence_length', 'N/A')}</li>
                        <li>Prediction Horizon: {self.config.data_config.get('prediction_horizon', 'N/A')}</li>
                        <li>Technical Indicators Used: {self.config.data_config.get('technical_indicators', 'N/A')}</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>Conclusion</h2>
                    <p>This report provides a snapshot of the model performances. Further analysis, including feature importance and error analysis, could yield deeper insights.</p>
                </div>
            </div>
        </body>
        </html>
        """

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Comprehensive report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Error saving comprehensive report: {e}")
            return None # Indicate failure

        return str(report_path)
