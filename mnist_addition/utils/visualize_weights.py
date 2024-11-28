import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import sys,os

class WeightVisualizer:
    def __init__(self, model_path: str):
        """
        Initialize weight visualizer with path to model weights
        
        Args:
            model_path (str): Path to the PyTorch model state dict
        """
        self.weights = torch.load(model_path)
        self.layer_names = [name for name in self.weights.keys() if 'weight' in name]
        
    def plot_weight_distributions(self, save_path: Optional[Path] = None):
        """Plot the distribution of weights for each layer"""
        n_layers = len(self.layer_names)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
        
        if n_layers == 1:
            axes = [axes]
            
        for ax, name in zip(axes, self.layer_names):
            weights = self.weights[name].cpu().numpy().flatten()
            sns.histplot(weights, kde=True, ax=ax)
            ax.set_title(f'Layer: {name}')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_weight_statistics(self, save_path: Optional[Path] = None):
        """Plot statistical measures of weights across layers"""
        stats = []
        for name in self.layer_names:
            weights = self.weights[name].cpu().numpy().flatten()
            stats.append({
                'layer': name,
                'mean': np.mean(weights),
                'std': np.std(weights),
                'median': np.median(weights),
                'max': np.max(weights),
                'min': np.min(weights)
            })
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot mean and std
        layers = [s['layer'] for s in stats]
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax1.bar(x - width/2, means, width, label='Mean')
        ax1.bar(x + width/2, stds, width, label='Std')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers, rotation=45)
        ax1.legend()
        ax1.set_title('Weight Statistics by Layer')
        
        # Plot min, median, max
        medians = [s['median'] for s in stats]
        maxs = [s['max'] for s in stats]
        mins = [s['min'] for s in stats]
        
        ax2.plot(layers, maxs, 'g-', label='Max')
        ax2.plot(layers, medians, 'b-', label='Median')
        ax2.plot(layers, mins, 'r-', label='Min')
        ax2.set_xticklabels(layers, rotation=45)
        ax2.legend()
        ax2.set_title('Weight Ranges by Layer')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_weight_heatmaps(self, save_path: Optional[Path] = None):
        """Plot heatmaps of weight matrices for each layer"""
        n_layers = len(self.layer_names)
        fig = plt.figure(figsize=(5*n_layers, 5))
        
        for i, name in enumerate(self.layer_names, 1):
            weights = self.weights[name].cpu().numpy()
            
            ax = fig.add_subplot(1, n_layers, i)
            im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Layer: {name}')
            ax.set_xlabel('Output Features')
            ax.set_ylabel('Input Features')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def generate_weight_summary(self):
        """Print summary statistics for model weights"""
        print("=== Model Weight Summary ===")
        
        total_params = 0
        for name in self.layer_names:
            weights = self.weights[name].cpu().numpy()
            n_params = weights.size
            total_params += n_params
            
            print(f"\nLayer: {name}")
            print(f"Shape: {weights.shape}")
            print(f"Parameters: {n_params:,}")
            print(f"Mean: {np.mean(weights):.6f}")
            print(f"Std: {np.std(weights):.6f}")
            print(f"Range: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
            
        print(f"\nTotal trainable parameters: {total_params:,}")

def main():
    # Path to your best model's state dict
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT.parent))
    print(f"Current working directory: {os.getcwd()}")
    model_path = "experiments/tuning_20241127_201445/exp_102/best_model/model.pt"
    save_dir = Path("experiments/tuning_20241127_201445/weight_visualizations")
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(exist_ok=True)
    
    viz = WeightVisualizer(model_path)
    
    # Generate visualizations
    viz.plot_weight_distributions(save_path=save_dir / "weight_distributions.png")
    viz.plot_weight_statistics(save_path=save_dir / "weight_statistics.png")
    viz.plot_weight_heatmaps(save_path=save_dir / "weight_heatmaps.png")
    
    # Print summary statistics
    viz.generate_weight_summary()

if __name__ == "__main__":
    main()