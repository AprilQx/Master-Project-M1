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
import sys, os

def load_mnist_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST addition dataset from the saved numpy arrays
    """
    # Load training and test data
    train_data = np.load(f"{data_path}/train_data.npy")
    train_labels = np.load(f"{data_path}/train_labels.npy")
    test_data = np.load(f"{data_path}/test_data.npy")
    test_labels = np.load(f"{data_path}/test_labels.npy")
    
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    
    return train_data, train_labels, test_data, test_labels

def train_and_evaluate_models(train_data: np.ndarray, train_labels: np.ndarray, 
                            test_data: np.ndarray, test_labels: np.ndarray,
                            save_dir: Optional[Path] = None) -> Dict:
    """
    Train and evaluate Random Forest and SVM models
    """
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    svm_model = SVC(kernel='rbf', random_state=42)
    

    print("Training Random Forest...")
    rf_model.fit(train_data, train_labels)
    print("Training SVM...")
    svm_model.fit(train_data, train_labels)
    

    rf_pred = rf_model.predict(test_data)
    svm_pred = svm_model.predict(test_data)
    

    rf_accuracy = accuracy_score(test_labels, rf_pred)
    svm_accuracy = accuracy_score(test_labels, svm_pred)
    

    rf_report = classification_report(test_labels, rf_pred, output_dict=True)
    svm_report = classification_report(test_labels, svm_pred, output_dict=True)
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest confusion matrix
    rf_cm = confusion_matrix(test_labels, rf_pred)
    sns.heatmap(rf_cm, annot=True, fmt='d', ax=ax1)
    ax1.set_title(f'Random Forest\nAccuracy: {rf_accuracy:.4f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # SVM confusion matrix
    svm_cm = confusion_matrix(test_labels, svm_pred)
    sns.heatmap(svm_cm, annot=True, fmt='d', ax=ax2)
    ax2.set_title(f'SVM\nAccuracy: {svm_accuracy:.4f}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'confusion_matrices.png')
    plt.show()
    
    # Compare with neural network results (if available)
    try:
        #nn_results = torch.load(save_dir.parent / 'best_model' / 'best_summary.json')
        nn_accuracy = 0.8737 #test result
        
        # Plot accuracy comparison
        models = ['Neural Network', 'Random Forest', 'SVM']
        accuracies = [nn_accuracy, rf_accuracy, svm_accuracy]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        if save_dir:
            plt.savefig(save_dir / 'accuracy_comparison.png')
        plt.show()
        
    except:
        print("Neural network results not found. Skipping comparison.")
    
    return {
        'random_forest': {
            'accuracy': rf_accuracy,
            'report': rf_report,
            'model': rf_model
        },
        'svm': {
            'accuracy': svm_accuracy,
            'report': svm_report,
            'model': svm_model
        }
    }

def print_results(results: Dict):
    """Print detailed results for each model"""
    print("\n=== Model Performance Comparison ===")
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print("\nDetailed Classification Report:")
        report = model_results['report']
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        for label in sorted([k for k in report.keys() if k.isdigit()]):
            metrics = report[label]
            print(f"\nClass {label}:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1-score']:.4f}")
            print(f"Support: {metrics['support']}")

def main():
    # Set paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed"
    save_dir = Path( project_root.parent/"experiments/sklearn_comparison")
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(data_path)
    
    # Train and evaluate models
    results = train_and_evaluate_models(train_data, train_labels, test_data, test_labels, save_dir)
    
    # Print detailed results
    print_results(results)

if __name__ == "__main__":
    main()