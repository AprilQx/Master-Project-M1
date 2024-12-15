import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List,Optional
import sys,os


class TuningVisualizer:
    def __init__(self, results_path: str):
        """
        Initialize visualizer with path to all_results.json
        
        Args:
            results_path (str): Path to the all_results.json file
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        # Convert experiments to DataFrame for easier analysis
        self.df = pd.DataFrame([
            {
                'experiment_id': exp['experiment_id'],
                'accuracy': exp['metrics']['accuracy'],
                'loss': exp['metrics']['loss'],
                'recall': exp['metrics']['recall'],
                'epochs_completed': exp['epochs_completed'],
                'hidden_size': exp['model_config']['hidden_size'],
                'num_layers': exp['model_config']['num_layers'],
                'dropout_rate': exp['model_config']['dropout_rate'],
                'learning_rate': exp['model_config']['learning_rate'],
                'batch_size': exp['data_config']['batch_size']
            }
            for exp in self.results['experiments']
        ])

    def plot_parameter_distributions(self, save_path: str = None):
        """Plot distributions of hyperparameters and their relationship with accuracy"""
        params = ['hidden_size', 'num_layers', 'dropout_rate', 'learning_rate', 'batch_size']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, param in enumerate(params):
            if i < len(axes):
                sns.scatterplot(data=self.df, x=param, y='accuracy', ax=axes[i])
                axes[i].set_title(f'Accuracy vs {param}')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Accuracy')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        

    def plot_parameter_importance(self, save_path: str = None):
        """Plot correlation between parameters and accuracy"""
        params = ['hidden_size', 'num_layers', 'dropout_rate', 'learning_rate', 'batch_size']
        correlations = [self.df[param].corr(self.df['accuracy']) for param in params]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(params))
        
        ax.barh(y_pos, correlations)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Correlation with Accuracy')
        ax.set_title('Parameter Importance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_epochs_analysis(self, save_path: str = None):
        """Plot relationship between epochs and performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Epochs vs Accuracy
        sns.scatterplot(data=self.df, x='epochs_completed', y='accuracy', ax=ax1)
        ax1.set_title('Epochs vs Accuracy')
        
        # Distribution of epochs
        sns.histplot(data=self.df, x='epochs_completed', ax=ax2)
        ax2.set_title('Distribution of Training Epochs')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


    def generate_summary_report(self):
        """Print a summary report of the hyperparameter tuning results"""
        best_model = self.df.loc[self.df['accuracy'].idxmax()]
        
        print("=== Hyperparameter Tuning Summary ===")
        print(f"\nTotal experiments: {len(self.df)}")
        print(f"Best accuracy: {best_model['accuracy']:.4f}")
        print("\nBest model configuration:")
        print(f"- Experiment ID: {best_model['experiment_id']}")
        print(f"- Hidden size: {best_model['hidden_size']}")
        print(f"- Number of layers: {best_model['num_layers']}")
        print(f"- Dropout rate: {best_model['dropout_rate']}")
        print(f"- Learning rate: {best_model['learning_rate']}")
        print(f"- Batch size: {best_model['batch_size']}")
        print(f"- Epochs completed: {best_model['epochs_completed']}")
        
        print("\nParameter ranges:")
        for param in ['hidden_size', 'num_layers', 'dropout_rate', 'learning_rate', 'batch_size']:
            print(f"- {param}: [{self.df[param].min()}, {self.df[param].max()}]")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
  
    results_path = project_root/"experiments/optuna_20241204_164643/all_results.json"
    save_dir = Path(project_root/"experiments"/"weight visualizations")
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(exist_ok=True,parents=True)
    
    viz = TuningVisualizer(results_path)
    
    # Generate all plots
    viz.plot_parameter_distributions(save_dir / "parameter_distributions.png")
    #viz.plot_top_models(save_dir / "top_models.png")
    viz.plot_parameter_importance(save_dir / "parameter_importance.png")
    viz.plot_epochs_analysis(save_dir / "epochs_analysis.png")
    
    # Print summary report
    viz.generate_summary_report()

if __name__ == "__main__":
    main()
    