import torch
import logging
from pathlib import Path
import json
import sys
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Get project root and add to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import load_config
from models.Mnist_Addition_nn import create_model
from data.dataset import get_dataloaders
from training.train import ModelTrainer

def setup_experiment_dir() -> Path:
    """Create and return experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / f"tuning_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def run_hyperparameter_tuning():
    """Run hyperparameter tuning experiments"""
    # Load base configuration
    config = load_config(str(PROJECT_ROOT / 'config.toml'))
    
    # Define hyperparameter search space
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64]
    }
    
    # Create experiment directory
    exp_dir = setup_experiment_dir()
    logging.info(f"Experiment directory: {exp_dir}")
    
    # Save hyperparameter search space
    with open(exp_dir / 'param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Track best model and performance
    best_val_acc = 0.0
    best_params = None
    results = []
    
    # Generate all parameter combinations
    from itertools import product
    param_combinations = list(product(
        param_grid['hidden_size'],
        param_grid['num_layers'],
        param_grid['dropout_rate'],
        param_grid['learning_rate'],
        param_grid['batch_size']
    ))

    total_experiments = len(param_combinations)
    logging.info(f"Running {total_experiments} experiments...")

    # Save total number of experiments
    with open(exp_dir / 'experiment_count.txt', 'w') as f:
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Total parameter combinations: {len(param_combinations)}\n")
        f.write("\nParameter grid:\n")
        f.write(json.dumps(param_grid, indent=2))

    for i, (hidden_size, num_layers, dropout_rate, lr, batch_size) in enumerate(param_combinations, 1):
        # Update config with current parameters
        exp_config = config.copy()
        exp_config['model'].update({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': lr
        })
        exp_config['data']['batch_size'] = batch_size

        # Create experiment-specific directory
        current_exp_dir = exp_dir / f"exp_{i}"
        current_exp_dir.mkdir(exist_ok=True)
        exp_config['train']['save_dir'] = str(current_exp_dir)
        
        logging.info(f"\nExperiment {i}/{total_experiments}")
        logging.info(f"Parameters: {json.dumps(exp_config['model'])}")

        try:
            # Get dataloaders with balanced dataset
            dataloaders = get_dataloaders(exp_config, root=str(PROJECT_ROOT / "data"))
            
            # Create and train model
            model = create_model(exp_config)
            trainer = ModelTrainer(model, dataloaders, exp_config)
            _, history = trainer.train()
            
            # Evaluate best validation performance
            best_val_epoch = min(range(len(history['val_loss'])), 
                               key=lambda i: history['val_loss'][i])
            val_acc = history['val_accuracy'][best_val_epoch]
            
            # Track results
            result = {
                'experiment': i,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout_rate': dropout_rate,
                'learning_rate': lr,
                'batch_size': batch_size,
                'best_val_accuracy': val_acc,
                'best_val_loss': history['val_loss'][best_val_epoch],
                'best_epoch': best_val_epoch + 1,
                'final_train_loss': history['train_loss'][-1],
                'final_train_accuracy': history['train_accuracy'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_val_accuracy': history['val_accuracy'][-1]
            }
            results.append(result)
            # Update best model if applicable
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = result.copy()
                # Save best model
                torch.save(model.state_dict(), exp_dir / 'best_model.pt')
                # Save best model configuration
                with open(exp_dir / 'best_model_config.json', 'w') as f:
                    json.dump(exp_config, f, indent=4)
            
            # Save current experiment results
            with open(current_exp_dir / 'results.json', 'w') as f:
                json.dump(result, f, indent=4)
            
            # Save intermediate results after each experiment
            with open(exp_dir / 'current_results.json', 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            logging.error(f"Experiment {i} failed: {str(e)}")
            with open(exp_dir / 'errors.log', 'a') as f:
                f.write(f"\nExperiment {i} failed:\n{str(e)}\n")
            continue
    # Save all results and best parameters
    with open(exp_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    with open(exp_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Calculate and save some basic statistics
    df = pd.DataFrame(results)
    stats = {
        'mean_val_accuracy': df['best_val_accuracy'].mean(),
        'std_val_accuracy': df['best_val_accuracy'].std(),
        'max_val_accuracy': df['best_val_accuracy'].max(),
        'min_val_accuracy': df['best_val_accuracy'].min(),
        'total_experiments': len(results),
        'successful_experiments': len(df)
    }
    
    with open(exp_dir / 'experiment_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    logging.info("\nHyperparameter tuning completed!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
    logging.info(f"Results saved in: {exp_dir}")

if __name__ == "__main__":
    run_hyperparameter_tuning()