import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import sys,os

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))
from data.dataset import get_dataloaders
from data.dataset import MNISTAdditionDataset
from models.Mnist_Addition_nn import MNISTAdditionNN 

def load_best_model_and_data(exp_path):
    """Load the best model and prepare data loader"""
    
    config_path = exp_path / 'exp_102' / 'best_model' / 'best_summary.json'
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    dataloaders = get_dataloaders(exp_config['config'], root="mnist_addition/data")
    model_path = exp_path / 'exp_102' / 'best_model' / 'model.pt'

    model_path = exp_path / 'exp_102' / 'best_model' / 'model.pt'
    state_dict = torch.load(model_path)
    
    # Extract only the model architecture parameters (excluding learning_rate)
    model_params = {
        'hidden_size': exp_config['config']['model']['hidden_size'],
        'num_layers': exp_config['config']['model']['num_layers'],
        'dropout_rate': exp_config['config']['model']['dropout_rate']
    }
    
    # Create a new model instance
    model = MNISTAdditionNN(**model_params)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, dataloaders['test']

def extract_embedding_features(model, dataloader, device='cpu'):
    """Extract features from the embedding layer (second-to-last layer)"""
    features = []
    labels = []
    model = model.to(device)
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Extracting embedding features"):
            data = data.to(device)
            # Forward pass through network until the embedding layer
            x = data
            for layer in model.network[:-1]:  # All layers except last Linear layer
                x = layer(x)
            # Get the embedding from the second-to-last layer
            embedding = model.embedding(x)
            
            features.append(embedding.cpu().numpy())
            labels.append(target.cpu().numpy())
            
    return np.vstack(features), np.concatenate(labels)

def find_optimal_perplexity(data, perplexities=[5, 10, 30, 50, 75]):
    """Find optimal perplexity value with fallback"""
    best_kl = float('inf')
    best_perp = None
    best_embedding = None
    
    print("Optimizing perplexity...")
    for perp in tqdm(perplexities):
        try:
            tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                        random_state=42, init='pca')
            embedding = tsne.fit_transform(data)
            kl_div = tsne.kl_divergence_
            
            if kl_div < best_kl:
                best_kl = kl_div
                best_perp = perp
                best_embedding = embedding
        except Exception as e:
            print(f"Warning: Failed with perplexity {perp}: {str(e)}")
            continue
    
    # Fallback if no valid result found
    if best_embedding is None:
        print("Warning: Using default perplexity as optimization failed")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                    random_state=42, init='pca')
        best_embedding = tsne.fit_transform(data)
        best_perp = 30
    
    return best_embedding, best_perp

def extract_raw_features(dataloader):
    """Extract raw features from the dataloader"""
    features = []
    labels = []
    
    for data, target in tqdm(dataloader, desc="Extracting raw features"):
        features.append(data.reshape(data.shape[0], -1).numpy())
        labels.append(target.numpy())
            
    return np.vstack(features), np.concatenate(labels)

def plot_tsne_comparison(raw_embedding, model_embedding, labels, perp_raw, perp_model, save_path):
    """Create comparison plot of raw and model embeddings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot raw input t-SNE
    scatter1 = ax1.scatter(raw_embedding[:, 0], raw_embedding[:, 1], 
                          c=labels, cmap='tab20', alpha=0.6)
    ax1.set_title(f'Raw Input t-SNE (perplexity={perp_raw})')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot model embedding t-SNE
    scatter2 = ax2.scatter(model_embedding[:, 0], model_embedding[:, 1], 
                          c=labels, cmap='tab20', alpha=0.6)
    ax2.set_title(f'Model Embedding t-SNE (perplexity={perp_model})')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_features_and_visualize(model, dataloader, save_dir, device='cpu'):
    """Extract features and create visualization with consistent scaling"""
    # Extract features
    model_features, labels = extract_embedding_features(model, dataloader, device)
    raw_features, _ = extract_raw_features(dataloader)
    
    # Use single scaler for consistency
    scaler = StandardScaler()
    raw_features_scaled = scaler.fit_transform(raw_features)
    model_features_scaled = scaler.transform(model_features)  # Use same scaler
    
    # Handle potential NaN values
    raw_features_scaled = np.nan_to_num(raw_features_scaled)
    model_features_scaled = np.nan_to_num(model_features_scaled)
    
    # Find optimal perplexity and get embeddings
    raw_embedding, perp_raw = find_optimal_perplexity(raw_features_scaled)
    model_embedding, perp_model = find_optimal_perplexity(model_features_scaled)
    
    # Create visualization
    plot_tsne_comparison(raw_embedding, model_embedding, labels,
                        perp_raw, perp_model,
                        save_dir / "tsne_comparison.png")
    
    return {
        'raw_perplexity': perp_raw,
        'model_perplexity': perp_model,
        'raw_embedding': raw_embedding,
        'model_embedding': model_embedding
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    exp_path = project_root.parent/'experiments'/'tuning_20241127_201445'
    save_dir = exp_path / "tsne_visualization"
    save_dir.mkdir(exist_ok=True)

    try:
        model, dataloader = load_best_model_and_data(exp_path)
        print("Model loaded successfully")

        results = extract_features_and_visualize(model, dataloader, save_dir, device)
        
        print(f"Optimal perplexity for raw features: {results['raw_perplexity']}")
        print(f"Optimal perplexity for model features: {results['model_perplexity']}")
        print(f"Visualizations saved to {save_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()