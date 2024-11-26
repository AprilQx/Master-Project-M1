#this is the model file for the neural network model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

#First version
class MNISTAdditionNN(nn.Module):
    def __init__(self, hidden_size=128,num_layers=2,dropout_rate=0.2):
        super().__init__()
        layers=[nn.Flatten(),
                nn.Linear(1568,hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)]
        for _ in range(num_layers-1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        layers.append(nn.Linear(hidden_size, 19))
        self.network=nn.Sequential(*layers)
        self.embedding=layers[-2]

    def forward(self, x):
        return self.network(x)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network[:-1]:
            x = layer(x)
        return x

def create_model(config: Dict[str, Any]) -> MNISTAdditionNN:
    """Create a new MNIST Addition model from config."""
    return MNISTAdditionNN(
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout_rate=config['model']['dropout_rate']
    )
# Make sure to export create_model
__all__ = ['MNISTAdditionNN', 'create_model']
