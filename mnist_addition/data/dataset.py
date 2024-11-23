# This is used for dataset creation and processing
import torch
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Import config with correct path
from utils.config import load_config
config = load_config(str(PROJECT_ROOT / 'config.toml'))

class MNISTAdditionDataset(Dataset):
    """MNIST Addition Dataset with statistical guarantees"""
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        seed: int = 43,
        balanced: bool = True,
    ):
     """Here, initialize the dataset"""
    self.root = root
    self.train = train
    self.transform = transform
    self.download = download
    self.seed = seed
    self.balanced = balanced

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    #load datasets
    self.minist = datasets.MNIST(
        root=str(self.root),
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )
    #create digit to indices mapping
    self.digit_to_indices=self._create_digit_indices()

    #Generating datasets
    self.pairs,self.targets=self._generate_balanced_pairs()

    # Verify statistical properties
    self._verify_statistics()

    def _create_digit_indices(self) -> Dict[int, np.ndarray]:
        """Creata a mapping from digit to their indices"""
        digit_to_indices={i: [] for i in range(10)}
        for idx, (_, target) in enumerate(self.minist):
            digit_to_indices[target].append(idx)
        return {k:np.array(v) for k,v in digit_to_indices.items()}
    
    def _generate_balanced_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced pairs"""
        pairs = []
        targets = []
        if self.balanced:
            target_sums=list(range(19)) # here, we created target sums from 0 to 18
            pairs_per_sum=1000 if self.train else 200
            for target_sum in target_sums:
                possible_pairs=[ 
                    (i, target_sum-i) 
                    for i in range(10) 
                    if 0<=target_sum-i<10
                ]
                for _ in range(pairs_per_sum):
                    d1,d2=possible_pairs[np.random.randint(len(possible_pairs))]
                    index1=np.random.choice(self.digit_to_indices[d1])
                    index2=np.random.choice(self.digit_to_indices[d2])
                    pairs.append([index1,index2])
                    targets.append(target_sum)
        else:
            # Random sampling (for comparison)
            num_pairs = 19000 if self.train else 3800
            for _ in range(num_pairs):
                d1=np.random.randint(10)
                d2=np.random.randint(10)
                idx1=np.random.choice(self.digit_to_indices[d1])
                idx2=np.random.choice(self.digit_to_indices[d2])
                pairs.append([idx1, idx2])
                targets.append(d1 + d2)

        return np.array(pairs), np.array(targets)
    
    def _verify_statistics(self):
        """Here, we verify the statics and print some logging info"""
        sum_counts=Counter(self.targets)
        total_pairs=len(self.targets)

        #Here we print some statistics results
        mean_sum=np.mean(self.targets)
        std_sum=np.std(self.targets)

        #Here we print some logging info
        logging.info("Dataset statistics:")
        logging.info(f"Total pairs: {total_pairs}")
        logging.info(f"Mean sum: {mean_sum:.2f}")
        logging.info(f"Std sum: {std_sum:.2f}")

        #Here we plot the distribution of the sums
        plt.figure(figsize=(10,5))
        sums,counts=zip(*sorted(sum_counts.items()))
        plt.bar(sums,counts)
        plt.xlabel("Sum")
        plt.ylabel("Count")
        plt.title("Distribution of sums")
        plt.show()
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        idx1, idx2 =self.pairs[idx]
        img1, _ =self.minist[idx1]
        img2, _ =self.minist[idx2]
        img=torch.cat([img1, img2], dim=2)#56*28
        target=self.targets[idx]
        if self.transform:
            img=self.transform(img)
        return img, target



def get_dataloaders(
        config: Dict,
        root: str="data",
) -> Dict[str, DataLoader]:
    """Create dataloaders with proper splits"""
    # Create transforms
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    # Create train dataset
    train_dataset=MNISTAdditionDataset(
        root=root,
        train=True,
        transform=transform,
        download=True,
        seed=config['data']['random_seed'],
        balanced=True
    )
    #create test dataset
    test_dataset=MNISTAdditionDataset(
        root=root,
        train=False,
        transform=transform,
        download=True,
        seed=config['data']['random_seed'],
        balanced=True
    )
    total_desired_size = int(3800 / 0.15)  # using previuos methods of balanced sampling, we have 3800 samples
    train_val_size = total_desired_size - 3800  # we need to add more samples to reach the desired ratio of 0.7/0.15/0.15
    # Create validation and train splits
    train_size = int(total_desired_size * 0.70)
    val_size = total_desired_size - train_size - 3800 

    generator = torch.Generator().manual_seed(config['dataset']['random_seed'])

    train_data, val_data = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )

    #verify the split ratios
    total_data = train_size + val_size + 3800
    print(f"\nDataset Split Ratios:")
    print(f"Train: {train_size} ({train_size/total_data*100:.1f}%)")
    print(f"Validation: {val_size} ({val_size/total_data*100:.1f}%)")
    print(f"Test: 3800 ({3800/total_data*100:.1f}%)")

    #create dataloaders
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_data,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers']
        ),
        'val': DataLoader(
            val_data,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers']
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers']
        )
    }
    return dataloaders
    