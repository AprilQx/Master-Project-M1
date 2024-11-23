from pathlib import Path
from utils.config import load_config
from data.dataset import MNISTAdditionDataset,get_dataloaders

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR

def main():
    config = load_config(str(PROJECT_ROOT / 'config.toml'))
    dataloaders = get_dataloaders(config)
    # Test the dataloaders
    for split, loader in dataloaders.items():
        print(f"{split} dataset size: {len(loader.dataset)}")

if __name__ == "__main__":
    main()
