import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys,os
from torchvision import transforms
# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))
print(f"Current working directory: {os.getcwd()}")


from data.dataset import get_dataloaders
from data.dataset import MNISTAdditionDataset
from models.Mnist_Addition_nn import MNISTAdditionNN 



def initialize_model(config: Dict) -> MNISTAdditionNN:
    """Initialize model with configuration"""
    model = MNISTAdditionNN(
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout_rate=config['model']['dropout_rate']
    )
    return model

def create_test_loader(test_dataset, batch_size: int) -> DataLoader:
    """Create test loader with specific batch size"""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

def load_model_and_run_inference(exp_dir: Path, test_dataset, device: str) -> Dict:
    """
    Load a model from experiment directory and run inference on test data
    """
    try:
        # Load model configuration first
        model_info_path = exp_dir / "model_info.json"
        if not model_info_path.exists():
            raise FileNotFoundError(f"model_info.json not found in {exp_dir}")
            
        with open(model_info_path, 'r') as f:
            model_config = json.load(f)
        
        # Extract configurations
        model_params = model_config.get('config', {}).get('model', {})
        data_params = model_config.get('config', {}).get('data', {})
        
        if not model_params or not data_params:
            raise KeyError(f"Invalid config structure in {exp_dir}")
        
        # Get batch size from config
        batch_size = data_params.get('batch_size', 32)  # default to 32 if not found
        
        # Create test loader
        test_loader = create_test_loader(test_dataset, batch_size)
        
        # Load model state
        model_path = exp_dir / "best_model" / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt not found in {exp_dir}/best_model")
            
        model_state = torch.load(model_path, map_location=device)
        
        # Initialize and load model
        model = initialize_model({
            'model': {
                'hidden_size': model_params.get('hidden_size', 256),
                'num_layers': model_params.get('num_layers', 3),
                'dropout_rate': model_params.get('dropout_rate', 0.3),
                'learning_rate': model_params.get('learning_rate', 0.001)
            }
        })
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        # Run inference
        correct = 0
        total = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        return {
            "experiment_id": int(exp_dir.name.split('_')[1]),
            "accuracy": accuracy,
            "predictions": predictions,
            "targets": targets,
            "model_config": {
                "model": model_params,
                "data": data_params
            }
        }
    
    except Exception as e:
        print(f"Error processing {exp_dir.name}: {str(e)}")
        print(f"Full error: {type(e).__name__}")
        return None

def visualize_results(results: List[Dict], save_dir: Optional[Path] = None):
    """Create visualizations of inference results"""
    if not results:
        print("No valid results to visualize!")
        return
        
    # Convert results to DataFrame
    df = pd.DataFrame([{
        'experiment_id': r['experiment_id'],
        'accuracy': r['accuracy'],
        'hidden_size': r['model_config']['model']['hidden_size'],
        'num_layers': r['model_config']['model']['num_layers'],
        'dropout_rate': r['model_config']['model']['dropout_rate'],
        'learning_rate': r['model_config']['model']['learning_rate']
    } for r in results])
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='accuracy', bins=20)
    plt.title('Distribution of Model Accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    if save_dir:
        plt.savefig(save_dir / 'accuracy_distribution.png')
    plt.close()
    
    # Plot accuracy vs hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    params = ['hidden_size', 'num_layers', 'dropout_rate', 'learning_rate','batch_size']
    for i, param in enumerate(params):
        sns.scatterplot(data=df, x=param, y='accuracy', ax=axes[i])
        axes[i].set_title(f'Accuracy vs {param}')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'accuracy_vs_params.png')
    plt.close()
    
    # Save detailed results
    if save_dir:
        # Sort by accuracy
        df_sorted = df.sort_values('accuracy', ascending=False)
        df_sorted.to_csv(save_dir / 'inference_results.csv', index=False)
        
        # Save top 10 configurations
        top_10 = df_sorted.head(10)
        with open(save_dir / 'top_10_models.json', 'w') as f:
            json.dump(top_10.to_dict('records'), f, indent=2)
        
        print(f"\nTop 5 Models:")
        for _, row in top_10.head().iterrows():
            print(f"\nExperiment {row['experiment_id']}:")
            print(f"Accuracy: {row['accuracy']:.4f}")
            print(f"Configuration:")
            print(f"- Hidden Size: {row['hidden_size']}")
            print(f"- Num Layers: {row['num_layers']}")
            print(f"- Dropout Rate: {row['dropout_rate']}")
            print(f"- Learning Rate: {row['learning_rate']}")

def run_all_experiments(base_dir: Path, test_dataset, device: str) -> List[Dict]:
    """Run inference on all experiment models"""
    results = []
    exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')],
                     key=lambda x: int(x.name.split('_')[1]))
    
    for exp_dir in tqdm(exp_dirs, desc="Running inference"):
        result = load_model_and_run_inference(exp_dir, test_dataset, device)
        if result:
            results.append(result)
    
    return results

def main():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    exp_path = project_root.parent.parent/'experiments'/ 'optuna_20241204_164643'
    save_dir = exp_path / "inference_results"
    save_dir.mkdir(exist_ok=True)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Initializing test dataset...")
    test_dataset = MNISTAdditionDataset(
        root="data",
        train=False,
        transform=transform,
        download=True,
        seed=42,
        balanced=False
    )
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Run inference on all models
    print("\nEvaluating models on test set...")
    results = run_all_experiments(exp_path, test_dataset, device)
    
    if not results:
        print("No successful results were obtained!")
        return
        
    # Visualize and save results
    print("\nGenerating visualizations...")
    visualize_results(results, save_dir)

if __name__ == "__main__":
    main()