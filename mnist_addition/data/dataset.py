# This is used for dataset creation and processing
import torch
import logging
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torchvision.datasets import MNIST
import sys

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Add the project root to Python path
sys.path.append(str(PROJECT_ROOT))
#print(f"Added {str(PROJECT_ROOT)} to path.")

# Import config with correct path
from utils.config import load_config
config = load_config(str(PROJECT_ROOT / 'config.toml'))

class MNISTAdditionDataset(Dataset):
    """MNIST Addition Dataset with statistical guarantees"""
    def __init__(self,
        root: str = "mnist_addition/data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        seed: int = 42,
        balanced: bool = True,
    ):
        """Here, initialize the dataset"""
        print(f"\nInitializing {'training' if train else 'test'} dataset...")

        self.train = train
        self.transform = transform
        self.balanced=balanced
        self.root=Path(root)
        self.data_dir = self.root / 'processed'
        self.data_dir.mkdir(parents=True, exist_ok=True)


        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Loading MNIST dataset...")
        mnist_path = Path(root) / "MNIST" / "processed"
        train_path = mnist_path / "training.pt"
        test_path = mnist_path / "test.pt"

        # Check if MNIST dataset is already downloaded
        if download and (not train_path.exists() or not test_path.exists()):
            print("Downloading complete MNIST dataset...")
            # Download training set
            MNIST(root=root, train=True, download=True, transform=None)
            print("Training set downloaded successfully")
            # Download test set
            MNIST(root=root, train=False, download=True, transform=None)
            print("Test set downloaded successfully")

        print(f"Loading {'training' if train else 'test'} split...")
        # Load MNIST directly

        self.mnist = MNIST(
            root=root,
            train=train,
            download=True,
            transform=None
        )
        print(f"Successfully loaded {len(self.mnist)} samples")
        #create digit to indices mapping
        print("Creating digit indices mapping...")
        self.digit_to_indices = {}
        targets = self.mnist.targets.numpy()
        for digit in range(10):
            self.digit_to_indices[digit] = np.where(targets == digit)[0]
            print(f"Found {len(self.digit_to_indices[digit])} samples for digit {digit}")
    

        #Generating datasets
        print("Generating pairs...")
        self.pairs,self.targets=self._generate_balanced_pairs()
        print(f"Generated {len(self.pairs)} pairs")

        # Verify statistical properties
        #self._verify_statistics()
        #print("Dataset initialization complete")

    def _create_digit_indices(self) -> Dict[int, np.ndarray]:
        """Creata a mapping from digit to their indices"""
        digit_to_indices={i: [] for i in range(10)}
        for idx, (_, target) in enumerate(self.mnist):
            digit_to_indices[target].append(idx)
        return {k:np.array(v) for k,v in digit_to_indices.items()}

    # def _verify_statistics(self):

    #     """Here, we verify the statics and print some logging info"""
    #     sum_counts=Counter(self.targets)
    #     total_pairs=len(self.targets)

    #     #Here we print some statistics results
    #     mean_sum=np.mean(self.targets)
    #     std_sum=np.std(self.targets)

    #     #Here we print some logging info
    #     logging.info("Dataset statistics:")
    #     logging.info(f"Total pairs: {total_pairs}")
    #     logging.info(f"Mean sum: {mean_sum:.2f}")
    #     logging.info(f"Std sum: {std_sum:.2f}")

    #     #Here we plot the distribution of the sums
    #     plt.figure(figsize=(10,5))
    #     sums,counts=zip(*sorted(sum_counts.items()))
    #     plt.bar(sums,counts)
    #     plt.xlabel("Sum")
    #     plt.ylabel("Count")
    #     plt.title("Distribution of sums")
    #     plt.show()

    
    def _generate_balanced_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced pairs"""
        pairs = []
        targets = []
        
        if self.balanced:
            target_sums=list(range(19)) # here, we created target sums from 0 to 18
            pairs_per_sum=1000 if self.train else 200
            expected_size = pairs_per_sum * len(target_sums)
            print(f"Expected dataset size: {expected_size} pairs")



            for target_sum in target_sums:
                possible_pairs=[ 
                    (i, target_sum-i) 
                    for i in range(10) 
                    if 0<=target_sum-i<10
                ]
                if not possible_pairs:
                    print(f"Warning: No possible pairs for sum {target_sum}")
                    continue
                for _ in range(pairs_per_sum):
                    d1,d2=possible_pairs[np.random.randint(len(possible_pairs))]
                    index1=np.random.choice(self.digit_to_indices[d1])
                    index2=np.random.choice(self.digit_to_indices[d2])
                    pairs.append([index1,index2])
                    targets.append(target_sum)
        else:
            # Random sampling (for comparison)
            num_pairs = 19000 if self.train else 3800
            print(f"Generating {num_pairs} random pairs...")
            for _ in range(num_pairs):
                d1=np.random.randint(10)
                d2=np.random.randint(10)
                idx1=np.random.choice(self.digit_to_indices[d1])
                idx2=np.random.choice(self.digit_to_indices[d2])
                pairs.append([idx1, idx2])
                targets.append(d1 + d2)
                
        print(f"Actually generated {len(pairs)} pairs")
        return np.array(pairs), np.array(targets)
    
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        idx1, idx2 =self.pairs[idx]
        img1, _ =self.mnist[idx1]
        img2, _ =self.mnist[idx2]
        to_tensor = transforms.ToTensor()
        img1_tensor = to_tensor(img1)
        img2_tensor = to_tensor(img2)
        img=torch.cat([img1_tensor, img2_tensor], dim=2)#56*28
        target=self.targets[idx]
        if self.transform:
            img=self.transform(img)
        return img, target



def get_dataloaders(
        config: Dict,
        root: str="data",
) -> Dict[str, DataLoader]:
    """Create dataloaders with proper splits"""
    print("\nInitializing dataloaders...")
    print(f"Using data root: {root}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("Created transforms")
    
    try:
        # Create train dataset with reduced number of workers for debugging
        print("\nCreating training dataset...")
        train_dataset = MNISTAdditionDataset(
            root=root,
            train=True,
            transform=transform,
            download=True,
            seed=42,
            balanced=True
        )
        print("Training dataset created successfully")
        
        # Create test dataset
        print("\nCreating test dataset...")
        test_dataset = MNISTAdditionDataset(
            root=root,
            train=False,
            transform=transform,
            download=True,
            seed=42,
            balanced=True
        )
        print("Test dataset created successfully")
        
        # Calculate splits with more logging
        # Get total length of training dataset
        total_train_size = len(train_dataset)
        print(f"\nTotal training dataset size: {total_train_size}")

        # Calculate split sizes based on actual dataset size
        val_size = int(total_train_size * 0.15)
        train_size = total_train_size - val_size

        print(f"\nSplit sizes:")
        print(f"- Train: {train_size}")
        print(f"- Validation: {val_size}")
        print(f"- Test: {len(test_dataset)}")

         # Verify split sizes
        if train_size + val_size != total_train_size:
            raise ValueError(f"Split sizes {train_size} + {val_size} != {total_train_size}")

        # Create splits
        generator = torch.Generator().manual_seed(config['data']['random_seed'])
        train_data, val_data = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=generator
        )
        print("Dataset splits created successfully")

        print(f"\nDataset Split Ratios:")
        total_size = train_size + val_size + len(test_dataset)
        print(f"Train: {train_size} ({train_size/total_size*100:.1f}%)")
        print(f"Validation: {val_size} ({val_size/total_size*100:.1f}%)")
        print(f"Test: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")

        # Create dataloaders with minimal number of workers for debugging
        print("\nCreating dataloaders...")
        dataloaders = {
            'train': DataLoader(
                train_data,
                batch_size=config['data']['batch_size'],
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True
            ),
            'val': DataLoader(
                val_data,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True
            )
        }
        print("Dataloaders created successfully")
        return dataloaders
        
    except Exception as e:
        print(f"\nError in get_dataloaders: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        raise