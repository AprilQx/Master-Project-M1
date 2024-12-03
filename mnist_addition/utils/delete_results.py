#this script deletes the result.json files in the experiment finetuning directories
import os
from pathlib import Path

def delete_results_files(tuning_dir: str) -> None:
    """
    Delete all results.json files from experiment directories.
    
    Args:
        tuning_dir (str): Path to the tuning directory
    """
    tuning_path = Path(tuning_dir)
    
    if not tuning_path.exists():
        raise ValueError(f"Tuning directory not found: {tuning_dir}")
    
    # Get all experiment directories
    exp_dirs = [d for d in tuning_path.iterdir() 
                if d.is_dir() and d.name.startswith('exp_')]
    
    deleted_count = 0
    for exp_dir in exp_dirs:
        results_file = exp_dir / "results.json"
        if results_file.exists():
            try:
                results_file.unlink()  # Delete the file
                deleted_count += 1
                print(f"Deleted: {results_file}")
            except Exception as e:
                print(f"Error deleting {results_file}: {str(e)}")
    
    print(f"\nTotal files deleted: {deleted_count}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exp_path = project_root.parent/'experiments'/'tuning_20241127_201445'
    delete_results_files(exp_path)