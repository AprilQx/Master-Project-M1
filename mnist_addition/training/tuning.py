import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from itertools import product
import json
import pandas as pd
from .train import ModelTrainer
from ..models.Mnist_Addition_nn import create_model
from ..data.dataset import get_dataloaders

class HyperparameterTuner:
    """Handles hyperparameter tuning for MNIST Addition model"""
    def __init__(self, config: Dict[str, Any], base_save_dir: str = "experiments"):
        self.config = config
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        # Extract hyperparameter lists
        self.param_grid = {
            'hidden_size': config['model']['hidden_sizes'],
            'num_layers': config['model']['num_layers'],
            'dropout_rate': config['model']['dropout_rates'],
            'learning_rate': config['model']['learning_rates'],
            'batch_size': config['model']['batch_sizes']
        }
        # Setup results tracking
        self.results = []
        
    def run_tuning(self):
        """Run hyperparameter tuning
         Returns:
            DataFrame containing results of all experiments
        """
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        
        logging.info(f"Starting hyperparameter tuning with {len(param_combinations)} combinations")

        for i, params in enumerate(param_combinations, 1):
            logging.info(f"\nExperiment {i}/{len(param_combinations)}")
            logging.info(f"Parameters: {params}")

            # Create experiment directory
            exp_dir = self.base_save_dir / f"experiment_{i}"
            exp_dir.mkdir(exist_ok=True)

            #Run experiments
            try:
                metrics = self._run_experiment(params, exp_dir)
                self._save_experiment_results(params, metrics, exp_dir)
            except Exception as e:
                logging.error(f"Experiment {i} failed: {str(e)}")
                continue

        # Compile and save all results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.base_save_dir / "tuning_results.csv", index=False)

        return results_df
    
    def _run_experiments(self, params: Dict[str, Any], exp_dir: Path) -> Dict[str, float]:
        """Run a single experiment with given hyperparameters"""
        # Update config with current parameters
        exp_config = self.config.copy()
        exp_config['model'].update(params)
        exp_config['training']['save_dir'] = str(exp_dir)

        # Create dataloaders with current batch size
        dataloaders = get_dataloaders(exp_config)
        
        # Create and train model
        model = create_model(exp_config)
        trainer = ModelTrainer(model, dataloaders, exp_config)
        _, history = trainer.train()

        # Get best validation metrics
        best_epoch = min(range(len(history['val_loss'])), 
                        key=lambda i: history['val_loss'][i])
        
        return {
            'best_val_loss': history['val_loss'][best_epoch],
            'best_val_accuracy': history['val_accuracy'][best_epoch],
            'best_val_recall': history['val_recall'][best_epoch],
            'best_epoch': best_epoch + 1,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_train_recall': history['train_recall'][-1]
        }
    
    def _save_experiment_results(self, params: Dict[str, Any], metrics: Dict[str, float], exp_dir: Path):
        """Save results of a single experiment"""
        result = {**params, **metrics}
        self.results.append(result)

        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=4)
            
        logging.info("Results saved successfully")

        