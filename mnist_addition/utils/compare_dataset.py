import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))
print(f"Current working directory: {os.getcwd()}")


from data.dataset import get_dataloaders
from data.dataset import MNISTAdditionDataset
from models.Mnist_Addition_nn import MNISTAdditionNN 

from utils.config import load_config

def load_metrics(experiment_path):
    """Load metrics from final_metrics_history.json"""
    with open(experiment_path / "final_metrics_history.json", "r") as f:
        return json.load(f)
    
def load_model_info(model_path):
    """Load model configuration"""
    with open(model_path / "model_info.json", "r") as f:
        return json.load(f)

def load_model(model_path):
    """Load the saved model with proper architecture"""
    # Load model info to get architecture parameters
    model_info = load_model_info(model_path)

    model = MNISTAdditionNN(
        hidden_size=model_info.get('hidden_size', 128),
        num_layers=model_info.get('num_layers', 2),
        dropout_rate=model_info.get('dropout_rate', 0.2)
    )
    
    # Load state dictionary
    state_dict = torch.load(model_path / "best_model" / "model.pt")
    model.load_state_dict(state_dict)
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total, correct, total

def plot_comparison(balanced_metrics, unbalanced_metrics, save_path):
    """Plot training history comparison"""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training and validation loss
    ax1.plot(balanced_metrics['train_loss'], 'b-', label='Balanced (Train)')
    ax1.plot(balanced_metrics['val_loss'], 'b--', label='Balanced (Val)')
    ax1.plot(unbalanced_metrics['train_loss'], 'r-', label='Unbalanced (Train)')
    ax1.plot(unbalanced_metrics['val_loss'], 'r--', label='Unbalanced (Val)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(balanced_metrics['train_accuracy'], 'b-', label='Balanced (Train)')
    ax2.plot(balanced_metrics['val_accuracy'], 'b--', label='Balanced (Val)')
    ax2.plot(unbalanced_metrics['train_accuracy'], 'r-', label='Unbalanced (Train)')
    ax2.plot(unbalanced_metrics['val_accuracy'], 'r--', label='Unbalanced (Val)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Balanced vs Unbalanced Training Comparison', y=1.05)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Set paths
    experiments_dir = Path("experiments")
    balanced_path = experiments_dir / "test1"
    unbalanced_path = experiments_dir / "test2"
    
    # Load metrics
    balanced_metrics = load_metrics(balanced_path)
    unbalanced_metrics = load_metrics(unbalanced_path)
    
    plot_comparison(balanced_metrics, unbalanced_metrics, 
                   experiments_dir / "training_comparison.png")
    
    # Load config
    config = load_config("mnist_addition/config.toml")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("\nInitializing test dataset...")
    test_dataset = MNISTAdditionDataset(
        root="data",
        train=False,
        transform=transform,
        download=True,
        seed=config['data']['random_seed'],
        balanced=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load and evaluate both models
    print("\nEvaluating models on balanced test set...")
    
    # Evaluate balanced model
    print("Loading balanced model...")
    balanced_model = load_model(balanced_path).to(device)
    balanced_acc, balanced_correct, balanced_total = evaluate_model(
        balanced_model, test_loader, device
    )
    
    # Evaluate unbalanced model
    print("Loading unbalanced model...")
    unbalanced_model = load_model(unbalanced_path).to(device)
    unbalanced_acc, unbalanced_correct, unbalanced_total = evaluate_model(
        unbalanced_model, test_loader, device
    )
    
    print("\nTest Set Performance for balanced test dataset:")
    print(f"{'Model':<15} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-" * 45)
    print(f"{'Balanced':<15} {balanced_acc:.4f}    {balanced_correct:<10} {balanced_total:<10}")
    print(f"{'Unbalanced':<15} {unbalanced_acc:.4f}    {unbalanced_correct:<10} {unbalanced_total:<10})")
if __name__ == "__main__":
    main()