from datetime import datetime
from pathlib import Path
import pickle
import json
import torch
import optuna # For optuna.Study type hint if needed, and general optuna usage
from typing import Tuple, Dict, Any, List, Optional # Added Optional

from config import Config
from logger_setup import logger
from data_processing import AEXDataProcessor
from optimization import HyperparameterOptimizer
from training import ModelTrainer
from models import AttentionLSTM, BaselineLSTM
from evaluation import ModelEvaluator
from visualization import VisualizationEngine

class AEXForecastingPipeline:
    """Main pipeline orchestrating the complete forecasting system."""

    def __init__(self, config: Config = None):
        self.config = config or Config() # If no config passed, use default
        self.data_processor: Optional[AEXDataProcessor] = None
        self.processed_data: Optional[Dict[str, Any]] = None # np.ndarray or torch.Tensor based on usage
        self.attention_model: Optional[AttentionLSTM] = None
        self.baseline_model: Optional[BaselineLSTM] = None
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.visualizer: Optional[VisualizationEngine] = None

        # Create output directories
        self.create_directories()

    def create_directories(self):
        """Create necessary output directories."""
        directories = ['models', 'reports', 'figures', 'data', 'logs'] # logs is handled by logger_setup
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured output directories exist: {directories}")

    def run_data_pipeline(self) -> Dict[str, Any]: # Return type more generic
        """Execute data processing pipeline."""
        logger.info("=== STARTING DATA PROCESSING PIPELINE ===")

        self.data_processor = AEXDataProcessor(self.config)
        self.processed_data = self.data_processor.process_all()

        # Save processed data
        data_path = Path('data/processed_data.pkl')
        try:
            with open(data_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
            logger.info(f"Processed data saved to {data_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            # Decide if this is a critical error to raise

        logger.info("Data processing pipeline completed successfully")
        return self.processed_data

    def run_hyperparameter_optimization(self, n_trials: int = 50) -> Optional[optuna.Study]:
        """Run hyperparameter optimization."""
        if not self.processed_data:
            logger.error("Processed data not available for hyperparameter optimization. Run data pipeline first.")
            return None

        logger.info("=== STARTING HYPERPARAMETER OPTIMIZATION ===")

        optimizer = HyperparameterOptimizer(self.config, self.processed_data) # Pass processed_data
        study = optimizer.optimize(n_trials=n_trials)

        # Update config with best parameters
        # Ensure keys from best_params exist in the respective config dicts before updating
        for key, value in study.best_params.items():
            if key in self.config.model_config:
                self.config.model_config[key] = value
            if key in self.config.training_config:
                self.config.training_config[key] = value
        logger.info(f"Configuration updated with best hyperparameters: {study.best_params}")

        # Save optimization results
        study_path = Path('reports/optimization_study.pkl')
        try:
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
            logger.info(f"Optimization study saved to {study_path}")
        except Exception as e:
            logger.error(f"Failed to save optimization study: {e}")

        logger.info("Hyperparameter optimization completed successfully")
        return study

    def train_models(self) -> Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]:
        """Train both attention-enhanced and baseline models."""
        if not self.processed_data:
            logger.error("Processed data not available for model training. Run data pipeline first.")
            return None, None

        logger.info("=== STARTING MODEL TRAINING ===")

        # Initialize trainer
        self.trainer = ModelTrainer(self.config)
        # Ensure processed_data is in the expected format (np.ndarray) for create_data_loaders
        train_loader, val_loader, _ = self.trainer.create_data_loaders(self.processed_data)

        # Initialize models
        input_size = self.processed_data['X_train'].shape[2]

        # Attention-enhanced model
        self.attention_model = AttentionLSTM(self.config)
        self.attention_model.set_input_size(input_size) # Call set_input_size
        self.attention_model.to(self.trainer.device)

        # Baseline model
        self.baseline_model = BaselineLSTM(input_size) # Pass input_size
        self.baseline_model.to(self.trainer.device)

        logger.info(f"Attention Model Info: {self.attention_model.get_model_info()}")
        # Could add similar info for BaselineLSTM if a get_model_info method exists/is added

        # Train attention model
        logger.info("Training attention-enhanced LSTM...")
        attention_history = self.trainer.train_model(self.attention_model, train_loader, val_loader)

        # Train baseline model
        logger.info("Training baseline LSTM...")
        baseline_history = self.trainer.train_model(self.baseline_model, train_loader, val_loader)

        # Save models
        attn_model_path = Path('models/attention_lstm_model.pth')
        base_model_path = Path('models/baseline_lstm_model.pth')
        try:
            torch.save({
                'model_state_dict': self.attention_model.state_dict(),
                'config': self.config.model_config, # Save relevant part of config
                'history': attention_history
            }, attn_model_path)
            logger.info(f"Attention model saved to {attn_model_path}")

            torch.save({
                'model_state_dict': self.baseline_model.state_dict(),
                # Add relevant config for baseline if needed, e.g., input_size, hidden_size
                'history': baseline_history
            }, base_model_path)
            logger.info(f"Baseline model saved to {base_model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

        logger.info("Model training completed successfully")
        return attention_history, baseline_history

    def evaluate_models(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
        """Comprehensive model evaluation."""
        if not self.processed_data or not self.attention_model or not self.baseline_model or not self.trainer:
            logger.error("Processed data or models/trainer not available for evaluation. Run previous steps.")
            return None, None, None

        logger.info("=== STARTING MODEL EVALUATION ===")

        self.evaluator = ModelEvaluator(self.config)
        _, _, test_loader = self.trainer.create_data_loaders(self.processed_data) # Get test_loader

        # Evaluate models
        logger.info("Evaluating Attention LSTM model...")
        attention_results = self.evaluator.evaluate_model(self.attention_model, test_loader, "test_attention")
        logger.info("Evaluating Baseline LSTM model...")
        baseline_results = self.evaluator.evaluate_model(self.baseline_model, test_loader, "test_baseline")

        # Statistical testing
        dm_test: Optional[Dict[str, float]] = None
        if attention_results and baseline_results:
            dm_test = self.evaluator.diebold_mariano_test(
                attention_results['actuals'],
                attention_results['predictions'],
                baseline_results['predictions']
            )
            logger.info("=== STATISTICAL TESTING RESULTS (DIEBOLD-MARIANO) ===")
            logger.info(f"  Statistic: {dm_test.get('dm_statistic', 'N/A'):.4f}")
            logger.info(f"  P-value: {dm_test.get('p_value', 'N/A'):.4f}")
            logger.info(f"  Significant: {dm_test.get('significant', 'N/A')}")
        else:
            logger.warning("Could not perform Diebold-Mariano test due to missing evaluation results.")

        # Save evaluation results
        eval_results_path = Path('reports/evaluation_results.pkl')
        try:
            evaluation_output = {
                'attention_results': attention_results,
                'baseline_results': baseline_results,
                'dm_test': dm_test
            }
            with open(eval_results_path, 'wb') as f:
                pickle.dump(evaluation_output, f)
            logger.info(f"Evaluation results saved to {eval_results_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

        logger.info("Model evaluation completed successfully")
        return attention_results, baseline_results, dm_test

    def generate_visualizations(self, attention_history: Optional[Dict[str, List[float]]],
                              baseline_history: Optional[Dict[str, List[float]]],
                              attention_results: Optional[Dict[str, Any]],
                              baseline_results: Optional[Dict[str, Any]]):
        """Generate comprehensive visualizations."""
        if not all([attention_history, baseline_history, attention_results, baseline_results]):
            logger.error("Missing data for generating visualizations. Ensure training and evaluation ran successfully.")
            return

        logger.info("=== GENERATING VISUALIZATIONS ===")
        self.visualizer = VisualizationEngine(self.config)
        figures_dir = Path('figures') # Ensure figures directory is defined for save_path

        # Training history plots
        self.visualizer.plot_training_history(
            attention_history, str(figures_dir / 'attention_training_history.html')
        )
        self.visualizer.plot_training_history(
            baseline_history, str(figures_dir / 'baseline_training_history.html')
        )

        # Prediction plots (pass time_index if available from processed_data or results)
        # Example: time_idx = self.processed_data['scaled_data'].index[-(len(attention_results['actuals'])):]
        # For now, plot_predictions in VisualizationEngine handles missing time_index
        self.visualizer.plot_predictions(
            attention_results, str(figures_dir / 'attention_predictions.html')
        )
        self.visualizer.plot_predictions(
            baseline_results, str(figures_dir / 'baseline_predictions.html')
        )

        # Attention weights visualization
        if 'attention_weights' in attention_results:
            self.visualizer.plot_attention_weights(
                attention_results['attention_weights'],
                self.config.data_config['sequence_length'],
                str(figures_dir / 'attention_weights.html')
            )

        logger.info("Visualizations generated successfully. Check the 'figures' directory.")

    def run_complete_pipeline(self, optimize_hyperparameters: bool = True,
                            n_optimization_trials: int = 50) -> Dict[str, Any]:
        """Execute the complete forecasting pipeline."""
        logger.info("=== STARTING COMPLETE AEX FORECASTING PIPELINE ===")
        start_time = datetime.now()
        final_results: Dict[str, Any] = {} # Initialize to ensure it's always defined

        try:
            # Step 1: Data Processing
            self.processed_data = self.run_data_pipeline()
            if not self.processed_data: raise Exception("Data processing failed.") # Critical step

            # Step 2: Hyperparameter Optimization (optional)
            study_results: Optional[optuna.Study] = None
            if optimize_hyperparameters:
                study_results = self.run_hyperparameter_optimization(n_optimization_trials)

            # Step 3: Model Training
            training_outputs = self.train_models()
            if not training_outputs or not training_outputs[0] or not training_outputs[1]:
                raise Exception("Model training failed.")
            attention_history, baseline_history = training_outputs

            # Step 4: Model Evaluation
            eval_outputs = self.evaluate_models()
            if not eval_outputs or not eval_outputs[0] or not eval_outputs[1]:
                raise Exception("Model evaluation failed.")
            attention_results, baseline_results, dm_test = eval_outputs

            # Step 5: Generate Visualizations
            self.generate_visualizations(
                attention_history, baseline_history,
                attention_results, baseline_results
            )

            # Step 6: Generate Comprehensive Report
            report_path: Optional[str] = None
            if self.visualizer and attention_results and baseline_results and dm_test:
                 report_path = self.visualizer.create_comprehensive_report(
                    attention_results, baseline_results, dm_test
                )
            else:
                logger.warning("Skipping comprehensive report due to missing data.")

            execution_time = datetime.now() - start_time

            final_results = {
                'execution_time': str(execution_time),
                'attention_model_metrics': attention_results.get('metrics') if attention_results else None,
                'baseline_model_metrics': baseline_results.get('metrics') if baseline_results else None,
                'statistical_test': dm_test,
                'optimization_study_summary': study_results.best_params if study_results else None, # Store best params
                'report_path': report_path,
                'model_paths': {
                    'attention_model': 'models/attention_lstm_model.pth',
                    'baseline_model': 'models/baseline_lstm_model.pth'
                }
            }

            summary_path = Path('reports/final_results_summary.json')
            try:
                # Create a serializable version of final_results
                serializable_results = final_results.copy()
                if study_results: # optuna.Study object is not directly JSON serializable
                     serializable_results['optimization_study_summary'] = {
                        "best_value": study_results.best_value,
                        "best_params": study_results.best_params
                    }
                with open(summary_path, 'w') as f:
                    json.dump(serializable_results, f, indent=4)
                logger.info(f"Final results summary saved to {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save final results summary: {e}")

            logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total execution time: {execution_time}")
            if report_path: logger.info(f"Comprehensive report available at: {report_path}")

            return final_results

        except Exception as e:
            logger.error(f"PIPELINE EXECUTION FAILED: {str(e)}", exc_info=True) # Log traceback
            # Potentially re-raise or handle as per application needs
            final_results['error'] = str(e)
            return final_results # Return partial results or error info
