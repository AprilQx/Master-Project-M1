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

