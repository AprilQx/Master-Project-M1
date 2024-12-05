import json
import os
from pathlib import Path
from typing import Dict, List, Optional

#this script helps to extract the best results from the experiment tuning directories

def count_epochs(experiment_path: Path) -> int:
    """
    Count the number of epoch directories in an experiment.
    
    Args:
        experiment_path (Path): Path to the experiment directory
        
    Returns:
        int: Number of epoch directories
    """
    # Count directories that start with 'epoch_'
    epoch_dirs = [d for d in experiment_path.iterdir() 
                 if d.is_dir() and d.name.startswith('epoch_')]
    return len(epoch_dirs)
def extract_experiment_results(experiment_path: Path) -> Optional[Dict]:
    """
    Extract the best model results from a single experiment directory.
    
    Args:
        experiment_path (Path): Path to the experiment directory
        
    Returns:
        Optional[Dict]: Dictionary containing the experiment results, or None if error
    """
    try:
        # Get experiment number from directory name
        exp_num = int(experiment_path.name.split('_')[1])
        
        # Path to best model summary and model info
        best_model_path = experiment_path / "best_model" / "best_summary.json"
        model_info_path = experiment_path / "model_info.json"
        
        if not best_model_path.exists() or not model_info_path.exists():
            print(f"Missing required files in experiment {exp_num}")
            return None
            
        # Load best model results and model info
        with open(best_model_path, 'r') as f:
            best_results = json.load(f)
        
        # Count actual epochs run
        epochs_run = count_epochs(experiment_path)
            
        # Extract results in desired format
        results = {
            "experiment_id": exp_num,
            "epochs_completed": epochs_run,
            "model_config": best_results["config"]["model"],
            "data_config": best_results["config"]["data"],
            "train_config": best_results["config"]["train"],
            "metrics": {
                "loss": best_results["metrics"]["loss"],
                "accuracy": best_results["metrics"]["accuracy"],
                "recall": best_results["metrics"]["recall"]
            },
            "best_epoch": best_results["epoch"]
        }
        
        return results
    
    except Exception as e:
        print(f"Error processing experiment {experiment_path.name}: {str(e)}")
        return None

def compile_all_results(tuning_dir: str) -> None:
    """
    Compile results from all experiments in the tuning directory.
    
    Args:
        tuning_dir (str): Path to the tuning directory
    """
    tuning_path = Path(tuning_dir)
    
    if not tuning_path.exists():
        raise ValueError(f"Tuning directory not found: {tuning_dir}")
    
    # Get all experiment directories
    exp_dirs = [d for d in tuning_path.iterdir() 
                if d.is_dir() and d.name.startswith('trial_')]
    
    # Extract results from each experiment
    all_results = []
    for exp_dir in sorted(exp_dirs, key=lambda x: int(x.name.split('_')[1])):
        results = extract_experiment_results(exp_dir)
        if results:
            all_results.append(results)
    
    # Sort results by experiment ID
    all_results.sort(key=lambda x: x["experiment_id"])
    
    # Find best model based on accuracy
    best_model = max(all_results, key=lambda x: x["metrics"]["accuracy"])
    
    # Calculate some statistics about epochs
    epoch_stats = {
        "min_epochs": min(r["epochs_completed"] for r in all_results),
        "max_epochs": max(r["epochs_completed"] for r in all_results),
        "avg_epochs": sum(r["epochs_completed"] for r in all_results) / len(all_results)
    }
    
    # Save compiled results
    output = {
        "experiments": all_results,
        "total_experiments": len(all_results),
        "epoch_statistics": epoch_stats,
        "timestamp": tuning_path.name.split('_')[1],
        "best_model": {
            "experiment_id": best_model["experiment_id"],
            "accuracy": best_model["metrics"]["accuracy"],
            "epochs_completed": best_model["epochs_completed"],
            "config": {
                "model": best_model["model_config"],
                "data": best_model["data_config"],
                "train": best_model["train_config"]
            }
        }
    }
    
    output_path = tuning_path / "all_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results compiled and saved to: {output_path}")
    print(f"Successfully processed {len(all_results)} experiments")
    print(f"Best model found in experiment {best_model['experiment_id']} "
          f"with accuracy {best_model['metrics']['accuracy']:.4f}")
    print("\nEpoch Statistics:")
    print(f"Min epochs: {epoch_stats['min_epochs']}")
    print(f"Max epochs: {epoch_stats['max_epochs']}")
    print(f"Average epochs: {epoch_stats['avg_epochs']:.2f}")

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).resolve().parent.parent
    exp_path = project_root.parent/'experiments'/ 'optuna_20241204_164643'
    compile_all_results(exp_path)