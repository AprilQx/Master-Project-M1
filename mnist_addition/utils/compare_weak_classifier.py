import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import seaborn as sns
from tqdm import tqdm
import sys,os

def train_combined_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray,
                            n_samples: int) -> Dict:
    clf = LogisticRegression(max_iter=1000)
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    
    clf.fit(X_train[indices], y_train[indices])
    y_pred = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'probabilities': probas,
        'predictions': y_pred,
        'model': clf
    }

def train_sequential_classifier(X_train_left: np.ndarray, y_train_left: np.ndarray,
                              X_train_right: np.ndarray, y_train_right: np.ndarray,
                              X_test_left: np.ndarray, X_test_right: np.ndarray,
                              y_test: np.ndarray, n_samples: int) -> Dict:
    clf = LogisticRegression(max_iter=1000)
    #here we try to train one classifier that learns digit recognition and reusing it for both left and right digits
    indices = np.random.choice(len(X_train_left), n_samples, replace=False)
    
    X_train_digits = np.vstack([X_train_left[indices], X_train_right[indices]]) #here we random choose samples
    y_train_digits = np.hstack([y_train_left[indices], y_train_right[indices]])
    
    clf.fit(X_train_digits, y_train_digits)
    
    left_pred = clf.predict(X_test_left)
    right_pred = clf.predict(X_test_right)
    y_pred = left_pred + right_pred
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'left_probabilities': clf.predict_proba(X_test_left),
        'right_probabilities': clf.predict_proba(X_test_right),
        'predictions': y_pred,
        'model': clf
    }

def prepare_data_splits(data_path: Path) -> Tuple[Dict, Dict]:
    X_train = np.load(data_path / 'train_data.npy')
    y_train_left = np.load(data_path / 'train_left_labels.npy')
    y_train_right = np.load(data_path / 'train_right_labels.npy')
    y_train_sum = np.load(data_path / 'train_sum_labels.npy')
    
    X_test = np.load(data_path / 'test_data.npy')
    y_test_left = np.load(data_path / 'test_left_labels.npy')
    y_test_right = np.load(data_path / 'test_right_labels.npy')
    y_test_sum = np.load(data_path / 'test_sum_labels.npy')
    
    X_train_left = X_train[:, :784]
    X_train_right = X_train[:, 784:]
    X_test_left = X_test[:, :784]
    X_test_right = X_test[:, 784:]
    
    return {
        'combined': {
            'X_train': X_train,
            'y_train': y_train_sum,
            'X_test': X_test,
            'y_test': y_test_sum
        },
        'sequential': {
            'X_train_left': X_train_left,
            'y_train_left': y_train_left,
            'X_train_right': X_train_right,
            'y_train_right': y_train_right,
            'X_test_left': X_test_left,
            'X_test_right': X_test_right,
            'y_test': y_test_sum
        }
    }
def plot_results(results: Dict, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot([r['n_samples'] for r in results['combined']], 
             [r['accuracy'] for r in results['combined']], 
             'o-', label='Combined')
    plt.plot([r['n_samples'] for r in results['sequential']], 
             [r['accuracy'] for r in results['sequential']], 
             'o-', label='Sequential')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance vs Training Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / 'accuracy_comparison.png')
    plt.close()
    

def compare_models(data_path: Path, sample_sizes: list = [50, 100, 500, 1000]):
    data = prepare_data_splits(data_path)
    results = {'combined': [], 'sequential': []}
    
    for n_samples in tqdm(sample_sizes, desc="Training models"):
        combined_results = train_combined_classifier(
            data['combined']['X_train'],
            data['combined']['y_train'],
            data['combined']['X_test'],
            data['combined']['y_test'],
            n_samples
        )
        
        sequential_results = train_sequential_classifier(
            data['sequential']['X_train_left'],
            data['sequential']['y_train_left'],
            data['sequential']['X_train_right'],
            data['sequential']['y_train_right'],
            data['sequential']['X_test_left'],
            data['sequential']['X_test_right'],
            data['sequential']['y_test'],
            n_samples
        )
        
        results['combined'].append({
            'n_samples': n_samples,
            'accuracy': combined_results['accuracy'],
            'probabilities': combined_results['probabilities']
        })
        
        results['sequential'].append({
            'n_samples': n_samples,
            'accuracy': sequential_results['accuracy'],
            'left_probabilities': sequential_results['left_probabilities'],
            'right_probabilities': sequential_results['right_probabilities']
        })
    
    save_path = data_path.parent / 'weak_classifier_results'
    save_path.mkdir(exist_ok=True)
    plot_results(results, save_path)
    
    print("\nResults Summary:")
    for method in ['combined', 'sequential']:
        print(f"\n{method.capitalize()} Approach:")
        for result in results[method]:
            print(f"Samples: {result['n_samples']}, Accuracy: {result['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed_classification"
    results = compare_models(data_path)