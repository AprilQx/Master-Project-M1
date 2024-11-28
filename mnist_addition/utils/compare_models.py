import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import torch
from typing import Dict, Tuple, Optional

def load_mnist_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST addition dataset from the saved numpy arrays
    """
    # Load training and test data
    train_data = np.load(f"{data_path}/train_data.npy")
    train_labels = np.load(f"{data_path}/train_labels.npy")
    test_data = np.load(f"{data_path}/test_data.npy")
    test_labels = np.load(f"{data_path}/test_labels.npy")
    
    # Reshape the data for sklearn models (flatten the images)
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    
    return train_data, train_labels, test_data, test_labels