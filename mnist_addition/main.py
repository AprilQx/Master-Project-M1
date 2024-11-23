from pathlib import Path
from utils.config import load_config
from data.dataset import MNISTAdditionDataset,get_dataloaders


def main():
    config = load_config('config.toml')
    dataloaders = get_dataloaders(config)
    # Test the dataloaders
    for split, loader in dataloaders.items():
        print(f"{split} dataset size: {len(loader.dataset)}")

if __name__ == "__main__":
    main()
