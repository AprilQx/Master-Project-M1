
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
def save_experiment_results(results_dir: Path, study, model, config, results):
    """Save all experiment results and artifacts"""
    print(f"\nSaving results to {results_dir}")
    
    # Create experiment directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study results
    study_results = {
        "best_params": study.best_params,
        "best_accuracy": study.best_value,
        "n_trials": len(study.trials),
        "optimization_history": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name
            }
            for t in study.trials
        ]
    }
    
    with open(results_dir / "study_results.json", "w") as f:
        json.dump(study_results, f, indent=4)
    
    # Save best model state
    torch.save(model.state_dict(), results_dir / "best_model.pt")
    
    # Save model configuration
    with open(results_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Save full results including training metrics
    with open(results_dir / "full_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Copy the original config file
    shutil.copy2(CONFIG_PATH, results_dir / "original_config.toml")
    
    # Save optimization plots
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        figures = {
            "optimization_history": plot_optimization_history(study),
            "param_importances": plot_param_importances(study),
            "parallel_coordinate": plot_parallel_coordinate(study),
            "slice_plot": plot_slice(study)
        }
        
        for name, fig in figures.items():
            fig.write_html(str(results_dir / f"{name}.html"))
                
    except Exception as e:
        print(f"Failed to save visualization plots: {e}")
    
    
    print("Successfully saved all experiment results")

def run_optimization(n_trials=100):
    print("\n=== Starting Optimization ===")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = project_root / "experiments" / timestamp
    
    try:
        print("Loading base configuration...")
        base_config = load_config(CONFIG_PATH)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    def objective(trial):
        # Suggest parameters
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 512, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        }
        
        try:
            # Update configuration
            trial_config = base_config.copy()
            trial_config['model'] = trial_config.get('model', {})
            trial_config['model'].update(params)
            trial_config['data']['batch_size'] = params['batch_size']
            
            print(f"\nTrial {trial.number}: Testing parameters: {params}")
            
            # Create dataloaders with current batch size
            dataloaders = get_dataloaders(trial_config, root=str(current_dir / "data"))
            
            # Create and train model
            model = create_model(trial_config)
            trainer = ModelTrainer(
                model=model,
                dataloaders=dataloaders,
                config=trial_config,
                patience=3
            )
            
            trained_model, metrics_history = trainer.train()
            accuracy = max(metrics_history['val_accuracy'])
            
            print(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
            
            # Report intermediate values for pruning
            trial.report(accuracy, step=1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            return accuracy
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"\nStarting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    print("\nOptimization completed!")

    # Train final model with best parameters
    final_config = base_config.copy()
    final_config['model'].update(study.best_params)
    final_config['data']['batch_size'] = study.best_params['batch_size']
    
    print("\nTraining final model with best parameters...")
    dataloaders = get_dataloaders(final_config, root=str(current_dir / "data"))
    best_model = create_model(final_config)
    best_trainer = ModelTrainer(
        model=best_model,
        dataloaders=dataloaders,
        config=final_config
    )
    
    trained_model, training_results = best_trainer.train()
    
    # Save all results
    save_experiment_results(results_dir, study, trained_model, final_config, training_results)
    
    return study, trained_model, results_dir

if __name__ == "__main__":
    try:
        study, best_model, results_dir = run_optimization(n_trials=50)
        print(f"\nOptimization completed successfully!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"\nResults saved to: {results_dir}")
        
        # Print parameter importance
        print("\nParameter importance:")
        importance = optuna.importance.get_param_importances(study)
        for param, score in importance.items():
            print(f"{param}: {score:.3f}")
            
    except Exception as e:
        print(f"Optimization failed with error: {e}")