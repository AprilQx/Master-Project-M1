import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_training_history(json_path: str):
    """Plot training and validation metrics from metrics history JSON file"""
    # Load the metrics history
    with open(json_path, 'r') as f:
        history = json.load(f)
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Training Loss', marker='o', markersize=4)
    ax1.plot(history['val_loss'], label='Validation Loss', marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracy'], label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Add overall title
    plt.suptitle('Training History', y=1.05)
    
    # Save the plot
    save_dir = Path(json_path).parent
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Training history plot saved to {save_dir / 'training_history.png'}")
    
    # Print best metrics
    best_val_loss_epoch = history['val_loss'].index(min(history['val_loss']))
    best_val_acc_epoch = history['val_accuracy'].index(max(history['val_accuracy']))
    
    print("\nBest Validation Results:")
    print(f"Lowest validation loss: {min(history['val_loss']):.4f} (epoch {best_val_loss_epoch})")
    print(f"Highest validation accuracy: {max(history['val_accuracy']):.4f} (epoch {best_val_acc_epoch})")
    
    # Early stopping analysis
    if best_val_loss_epoch < len(history['val_loss']) - 1:
        print(f"\nNote: Best validation loss occurred at epoch {best_val_loss_epoch}")
        print("Model might have benefited from early stopping around this point.")

if __name__ == "__main__":
    # Use the metrics history JSON file
    project_root = Path(__file__).resolve().parent.parent.parent
    json_path=project_root/"experiments"/"test2"/"final_metrics_history.json"
    plot_training_history(json_path)