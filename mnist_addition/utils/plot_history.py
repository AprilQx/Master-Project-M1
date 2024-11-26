import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch

def plot_training_history(save_dir: str):
    """Plot training and validation metrics from saved history"""
    save_path = Path(save_dir)
    
    # Initialize lists to store metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # Read metrics from each epoch
    epoch = 0
    while True:
        epoch_dir = save_path / f'epoch_{epoch}'
        if not epoch_dir.exists():
            break
            
        metrics_file = epoch_dir / 'metrics.json'
        if not metrics_file.exists():
            break
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        train_loss.append(metrics['train']['loss'])
        val_loss.append(metrics['validation']['loss'])
        train_acc.append(metrics['train']['accuracy'])
        val_acc.append(metrics['validation']['accuracy'])
        
        epoch += 1
    
    if epoch == 0:
        print("No training history found!")
        return
        
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_loss, label='Training Loss', marker='o')
    ax1.plot(val_loss, label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_acc, label='Training Accuracy', marker='o')
    ax2.plot(val_acc, label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Add overall title
    plt.suptitle('Training History', y=1.05)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Training history plot saved to {save_path / 'training_history.png'}")
    
    # Print best metrics
    best_val_loss_epoch = val_loss.index(min(val_loss))
    best_val_acc_epoch = val_acc.index(max(val_acc))
    
    print("\nBest Validation Results:")
    print(f"Lowest validation loss: {min(val_loss):.4f} (epoch {best_val_loss_epoch})")
    print(f"Highest validation accuracy: {max(val_acc):.4f} (epoch {best_val_acc_epoch})")

if __name__ == "__main__":
    # Use the same save directory as in your config
    plot_training_history("experiments/test1")