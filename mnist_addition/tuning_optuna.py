
#Note that this file has issues and is not working properly.

import optuna
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import logging
from datetime import datetime
import shutil

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from mnist_addition.utils.config import load_config
from mnist_addition.models.Mnist_Addition_nn import create_model
from mnist_addition.data.dataset import get_dataloaders
from mnist_addition.training.train import ModelTrainer

CONFIG_PATH = current_dir / 'config.toml'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_experiment_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = project_root.parent / "experiments" / f"optuna_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_experiment_results(results_dir: Path, study, model, config, results):
    results_dir.mkdir(parents=True, exist_ok=True)
    
    study_results = {
        "best_params": study.best_params,
        "best_accuracy": study.best_value,
        "n_trials": len(study.trials),
        "optimization_history": [{
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name
        } for t in study.trials]
    }
    
    # Save results and configs
    with open(results_dir / "study_results.json", "w") as f:
        json.dump(study_results, f, indent=4)
    torch.save(model.state_dict(), results_dir / "best_model.pt")
    with open(results_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=4)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Calculate and save statistics
    trials_df = study.trials_dataframe()
    stats = {
        'mean_accuracy': trials_df['value'].mean(),
        'std_accuracy': trials_df['value'].std(),
        'max_accuracy': trials_df['value'].max(),
        'min_accuracy': trials_df['value'].min(),
        'total_trials': len(study.trials),
        'completed_trials': len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    }
    with open(results_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Save visualization plots
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        
        logging.info("Generating visualization plots...")
        figures = {
            "optimization_history": plot_optimization_history(study),
            "param_importances": plot_param_importances(study),
            "parallel_coordinate": plot_parallel_coordinate(study),
            "slice_plot": plot_slice(study)
        }
        
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for name, fig in figures.items():
            fig.write_html(str(plots_dir / f"{name}.html"))
        logging.info(f"Plots saved to {plots_dir}")
                
    except Exception as e:
        logging.error(f"Failed to save visualization plots: {e}")

def run_optimization(n_trials=100):
    exp_dir = setup_experiment_dir()
    logging.info(f"Experiment directory: {exp_dir}")
    print("\n=== Starting Optimization ===")
    
    # Create experiment directory
    base_config = load_config(CONFIG_PATH)
    

    def objective(trial):
        trial_dir = exp_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        # Suggest parameters
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 512, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        }
        logging.info(f"\nTrial {trial.number}/{n_trials}")
        logging.info(f"Parameters: {json.dumps(params, indent=2)}")
        try:
            # Update configuration
            trial_config = base_config.copy()
            trial_config['model'].update(params)
            trial_config['data']['batch_size'] = params['batch_size']
            trial_config['train']['save_dir'] = str(trial_dir)
            
            with open(trial_dir / 'config.json', 'w') as f:
                json.dump(trial_config, f, indent=4)
            
            # Create dataloaders with current batch size
            dataloaders = get_dataloaders(trial_config, root=str(current_dir / "data"))
            
            # Create and train model
            model = create_model(trial_config)
            trainer = ModelTrainer(
                model=model,
                dataloaders=dataloaders,
                config=trial_config,
                patience=3  # Removed save_dir argument
            )
            
            trained_model, metrics_history = trainer.train()
            accuracy = max(metrics_history['val_accuracy'])

             # Save trial results
            result = {
                'trial_number': trial.number,
                'parameters': params,
                'best_val_accuracy': accuracy,
                'history': metrics_history
            }
            with open(trial_dir / 'results.json', 'w') as f:
                json.dump(result, f, indent=4)
            
            print(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
            logging.info(f"Trial {trial.number} completed: accuracy = {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best params
    logging.info("\nTraining final model with best parameters...")
    final_config = base_config.copy()
    final_config['model'].update(study.best_params)
    final_config['data']['batch_size'] = study.best_params['batch_size']
    
    dataloaders = get_dataloaders(final_config, root=str(current_dir / "data"))
    best_model = create_model(final_config)
    best_trainer = ModelTrainer(best_model, dataloaders, final_config)
    trained_model, training_results = best_trainer.train()
    
    save_experiment_results(exp_dir, study, trained_model, final_config, training_results)
    logging.info(f"Results saved to: {exp_dir}")
    
    return study, trained_model, exp_dir


if __name__ == "__main__":
    try:
        study, best_model, results_dir = run_optimization(n_trials=50)
        logging.info(f"\nBest accuracy: {study.best_value:.4f}")
        logging.info(f"Best parameters: {json.dumps(study.best_params, indent=2)}")
        
        importance = optuna.importance.get_param_importances(study)
        logging.info("\nParameter importance:")
        for param, score in importance.items():
            logging.info(f"{param}: {score:.3f}")
            
    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}")