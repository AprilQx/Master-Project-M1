# This is used for dataset creation and processing
import torch
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter
import matplotlib.pyplot as plt

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
    self.digit_to_indices = self._create_digit_indices()

    #Generating datasets
    self.pairs,self.targets= self._generate_balanced_pairs()

    # Verify statistical properties
    self._verify_statistics()

    def _create_digit_indices(self) -> Dict[int, np.ndarray]:
        """Creata a mapping from digit to their indices"""
        digit_to_indices = {i: [] for i in range(10)}
        for idx, (_, target) in enumerate(self.minist):
            digit_to_indices[target].append(idx)
        return {k:np.array(v) for k,v in digit_to_indices.items()}
    
    def _generate_balanced_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced pairs"""
        pairs = []
        targets = []
        if self.balanced:
            target_sums=list(range(19)) # here, we created target sums from 0 to 18
            pairs_per_sum=1000 if self.train else 100
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
                d1 = np.random.randint(10)
                d2 = np.random.randint(10)
                idx1 = np.random.choice(self.digit_to_indices[d1])
                idx2 = np.random.choice(self.digit_to_indices[d2])
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
        idx1, idx2 = self.pairs[idx]
        img1, _ = self.minist[idx1]
        img2, _ = self.minist[idx2]
        img = torch.cat([img1, img2], dim=2)#56*28
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target



def get_dataloaders():
    pass