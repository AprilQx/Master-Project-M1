import torch
from pathlib import Path
import os
import sys

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))

from mnist_addition.utils.config import load_config
from mnist_addition.models.Mnist_Addition_nn import create_model
from mnist_addition.data.dataset import get_dataloaders
from mnist_addition.training.train import ModelTrainer

CONFIG_PATH = PROJECT_ROOT / 'config.toml'
print(f"Current working directory: {os.getcwd()}")

def main():
    try:
        config = load_config(CONFIG_PATH)
        # Create dataloaders
        dataloaders = get_dataloaders(config)

        # Create model
        model = create_model(config)

        # Initialize trainer
        trainer = ModelTrainer(
                    model=model,
                    dataloaders=dataloaders,
                    config=config
                )
        trained_model, metrics_history = trainer.train()
        # Evaluate on test set
        test_metrics = trainer.evaluate(dataloaders['test'])
        print(f"\nTest set metrics:")
        for metric_name, value in test_metrics.items():
                print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()      