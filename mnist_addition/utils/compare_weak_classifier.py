import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import seaborn as sns

def load_and_split_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data and split into single digits for sequential approach"""

    X_train = np.load(data_path / 'train_data.npy')
    y_train = np.load(data_path / 'train_labels.npy')
    X_test = np.load(data_path / 'test_data.npy')
    y_test = np.load(data_path / 'test_labels.npy')
    
    X_train_left = X_train[:, :784]
    X_train_right = X_train[:, 784:]
    X_test_left = X_test[:, :784]
    X_test_right = X_test[:, 784:]
    
    return (X_train_left, X_train_right, X_test_left, X_test_right), (y_train, y_test)

def train_combined_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray,
                            n_samples: int) -> Dict:
    """Train classifier on combined images"""
    clf = LogisticRegression(max_iter=1000)
    
    # Use subset of training data
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    # Train and evaluate
    clf.fit(X_train_subset, y_train_subset)
    y_pred = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'probabilities': probas,
        'predictions': y_pred
    }
def train_sequential_classifier(X_train_left: np.ndarray, X_train_right: np.ndarray,
                              X_test_left: np.ndarray, X_test_right: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              n_samples: int)-> Dict:
    """Train classifier on sequential images"""
    clf = LogisticRegression(max_iter=1000)
    indices = np.random.choice(len(X_train_left), n_samples, replace=False)
    X_train_left_subset = X_train_left[indices]
    X_train_right_subset = X_train_right[indices]
    y_train_subset = y_train[indices]
