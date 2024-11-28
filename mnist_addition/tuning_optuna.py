import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import optuna
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import logging
from datetime import datetime

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from mnist_addition.utils.config import load_config
from mnist_addition.models.Mnist_Addition_nn import create_model
from mnist_addition.data.dataset import get_dataloaders
from mnist_addition.training.train import ModelTrainer

CONFIG_PATH = current_dir / 'config.toml'
print(f"Current working directory: {os.getcwd()}")


def run_optimization(n_trials=100):
    print("\n=== Starting Optimization ===")
    print("Loading base configuration...")
    base_config = load_config(CONFIG_PATH)
    print("Success!")
    print("\nDataset initialization stage...")

    print("\nInitializing dataloaders...")
    dataloaders = get_dataloaders(base_config, root=str(current_dir / "data"))
    print("Dataloaders created successfully")

    
    
    def objective(trial):
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 512),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        }
        
        trial_config = base_config.copy()
        trial_config['model'].update(params)
        
        model = create_model(trial_config)
        trainer = ModelTrainer(
            model=model,
            dataloaders=dataloaders,
            config=trial_config,
            patience=3
        )
        
        try:
            _, metrics_history = trainer.train()
            return max(metrics_history['val_accuracy'])
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Save results
    results_dir = project_root / "experiments" / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_params": study.best_params,
        "best_accuracy": study.best_value,
        "n_trials": n_trials,
        "optimization_history": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ]
    }
    
    with open(results_dir / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Train final model with best parameters
    final_config = base_config.copy()
    final_config['model'].update(study.best_params)
    
    best_model = create_model(final_config)
    best_trainer = ModelTrainer(
        model=best_model,
        dataloaders=dataloaders,
        config=final_config
    )
    
    trained_model, _ = best_trainer.train()
    torch.save(trained_model.state_dict(), results_dir / "best_model.pt")
    
    return results

if __name__ == "__main__":
    results = run_optimization()
    print(f"Best parameters: {results['best_params']}")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
