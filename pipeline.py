from datetime import datetime
from pathlib import Path
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any
from utils import NumpyJSONEncoder
import torch
import optuna
import numpy as np

from config import Config
from data_processing import AEXDataProcessor
from optimization import HyperparameterOptimizer
from model import AttentionLSTM, BaselineLSTM
from training import ModelTrainer
from evaluation import ModelEvaluator
from visualization import VisualizationEngine
from logger import setup_logging

# Initialize logger for this module
logger = setup_logging()

class AEXForecastingPipeline:
    """Main pipeline orchestrating the complete forecasting system."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_processor = None
        self.processed_data = None
        self.attention_model = None
        self.baseline_model = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = None

        # Create output directories
        self.create_directories()

    def create_directories(self):
        """Create necessary output directories."""
        directories = ['models', 'reports', 'figures', 'data', 'logs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def run_data_pipeline(self) -> Dict[str, np.ndarray]:
        """Execute data processing pipeline."""
        logger.info("=== STARTING DATA PROCESSING PIPELINE ===")

        self.data_processor = AEXDataProcessor(self.config)
        self.processed_data = self.data_processor.process_all()

        # Save processed data
        with open('data/processed_data.pkl', 'wb') as f:
            pickle.dump(self.processed_data, f)

        logger.info("Data processing pipeline completed successfully")
        return self.processed_data

    def run_hyperparameter_optimization(self, n_trials: int = 50) -> optuna.Study:
        """Run hyperparameter optimization."""
        logger.info("=== STARTING HYPERPARAMETER OPTIMIZATION ===")

        optimizer = HyperparameterOptimizer(self.config, self.processed_data)
        study = optimizer.optimize(n_trials=n_trials)

        # Update config with best parameters
        self.config.model_config.update(study.best_params)
        # Ensure only relevant training params are updated
        for key, value in study.best_params.items():
            if key in self.config.training_config:
                self.config.training_config[key] = value

        # Save optimization results
        with open('reports/optimization_study.pkl', 'wb') as f:
            pickle.dump(study, f)

        logger.info("Hyperparameter optimization completed successfully")
        return study

    def train_models(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Train both attention-enhanced and baseline models."""
        logger.info("=== STARTING MODEL TRAINING ===")

        # Initialize trainer
        self.trainer = ModelTrainer(self.config)
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(self.processed_data)

        # Initialize models
        input_size = self.processed_data['X_train'].shape[2]

        # Attention-enhanced model
        self.attention_model = AttentionLSTM(self.config)
        self.attention_model.set_input_size(input_size)
        self.attention_model.to(self.trainer.device)

        # Baseline model
        self.baseline_model = BaselineLSTM(input_size) # Assuming BaselineLSTM takes input_size directly
        self.baseline_model.to(self.trainer.device)

        logger.info(f"Attention Model Info: {self.attention_model.get_model_info()}")

        # Train attention model
        logger.info("Training attention-enhanced LSTM...")
        attention_history = self.trainer.train_model(self.attention_model, train_loader, val_loader)

        # Train baseline model
        logger.info("Training baseline LSTM...")
        baseline_history = self.trainer.train_model(self.baseline_model, train_loader, val_loader)

        # Save models
        torch.save({
            'model_state_dict': self.attention_model.state_dict(),
            'config': self.config, # Save config with the model
            'history': attention_history
        }, 'models/attention_lstm_model.pth')

        torch.save({
            'model_state_dict': self.baseline_model.state_dict(),
            # 'config': self.config, # Baseline might not need full config if simple
            'history': baseline_history
        }, 'models/baseline_lstm_model.pth')

        logger.info("Model training completed successfully")
        return attention_history, baseline_history

    def evaluate_models(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """Comprehensive model evaluation."""
        logger.info("=== STARTING MODEL EVALUATION ===")

        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.config)

        # Create data loaders (assuming trainer is already initialized)
        if not self.trainer:
             self.trainer = ModelTrainer(self.config) # Ensure trainer exists
        _, _, test_loader = self.trainer.create_data_loaders(self.processed_data)

        # Evaluate models
        attention_results = self.evaluator.evaluate_model(self.attention_model, test_loader, "test")
        baseline_results = self.evaluator.evaluate_model(self.baseline_model, test_loader, "test")

        # Statistical testing
        dm_test = self.evaluator.diebold_mariano_test(
            attention_results['actuals'],
            attention_results['predictions'],
            baseline_results['predictions']
        )

        logger.info("=== STATISTICAL TESTING RESULTS ===")
        logger.info(f"Diebold-Mariano Test - Statistic: {dm_test['dm_statistic']:.4f}, "
                   f"P-value: {dm_test['p_value']:.4f}, "
                   f"Significant: {dm_test['significant']}")

        # Save evaluation results
        evaluation_results = {
            'attention_results': attention_results,
            'baseline_results': baseline_results,
            'dm_test': dm_test
        }

        with open('reports/evaluation_results.pkl', 'wb') as f:
            pickle.dump(evaluation_results, f)

        logger.info("Model evaluation completed successfully")
        return attention_results, baseline_results, dm_test

    def generate_visualizations(self, attention_history: Dict[str, List[float]],
                              baseline_history: Dict[str, List[float]],
                              attention_results: Dict[str, Any],
                              baseline_results: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        logger.info("=== GENERATING VISUALIZATIONS ===")

        self.visualizer = VisualizationEngine(self.config)

        # Training history plots
        self.visualizer.plot_training_history(
            attention_history, 'figures/attention_training_history.html'
        )

        self.visualizer.plot_training_history(
            baseline_history, 'figures/baseline_training_history.html'
        )

        # Prediction plots
        self.visualizer.plot_predictions(
            attention_results, 'figures/attention_predictions.html'
        )

        self.visualizer.plot_predictions(
            baseline_results, 'figures/baseline_predictions.html'
        )

        # Attention weights visualization
        if 'attention_weights' in attention_results:
            self.visualizer.plot_attention_weights(
                attention_results['attention_weights'],
                self.config.data_config['sequence_length'],
                'figures/attention_weights.html'
            )

        logger.info("Visualizations generated successfully")

    def run_complete_pipeline(self, optimize_hyperparameters: bool = True,
                            n_optimization_trials: int = 50) -> Dict[str, Any]:
        """Execute the complete forecasting pipeline."""
        logger.info("=== STARTING COMPLETE AEX FORECASTING PIPELINE ===")

        start_time = datetime.now()

        try:
            # Step 1: Data Processing
            self.processed_data = self.run_data_pipeline()

            # Step 2: Hyperparameter Optimization (optional)
            study = None
            if optimize_hyperparameters:
                study = self.run_hyperparameter_optimization(n_optimization_trials)

            # Step 3: Model Training
            attention_history, baseline_history = self.train_models()

            # Step 4: Model Evaluation
            attention_results, baseline_results, dm_test = self.evaluate_models()

            # Step 5: Generate Visualizations
            self.generate_visualizations(
                attention_history, baseline_history,
                attention_results, baseline_results
            )

            # Step 6: Generate Comprehensive Report
            report_path = self.visualizer.create_comprehensive_report(
                attention_results, baseline_results, dm_test
            )

            # Calculate execution time
            execution_time = datetime.now() - start_time

            # Compile final results
            final_results = {
                'execution_time': str(execution_time),
                'attention_model_metrics': attention_results['metrics'],
                'baseline_model_metrics': baseline_results['metrics'],
                'statistical_test': dm_test,
                'optimization_study': study.best_params if study else None, # Store only best params
                'report_path': report_path,
                'model_paths': {
                    'attention_model': 'models/attention_lstm_model.pth',
                    'baseline_model': 'models/baseline_lstm_model.pth'
                }
            }

            # Save final results summary
            with open('reports/final_results_summary.json', 'w') as f:
                # Custom serializer for optuna objects if needed, or just store relevant parts
                json.dump({k: (v if not isinstance(v, optuna.Study) else str(v))
                           for k, v in final_results.items()}, f, indent=2, cls=NumpyJSONEncoder)

            logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total execution time: {execution_time}")
            logger.info(f"Comprehensive report available at: {report_path}")

            return final_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
