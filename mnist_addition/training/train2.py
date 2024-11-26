import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Recall
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader

class ModelTrainer:
    """
    Handles training and evaluation of MNIST Addition model
    """
    def __init__(self, model: nn.Module, dataloaders: Dict[str, DataLoader], config: Dict[str, Any]):
        self.model=model
        self.dataloaders=dataloaders
        self.config=config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model=self.model.to(self.device)

        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['model']['learning_rate']
        )
        # Initialize metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=19).to(self.device)
        self.train_recall = Recall(task='multiclass', num_classes=19).to(self.device)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=19).to(self.device)
        self.val_recall = Recall(task='multiclass', num_classes=19).to(self.device)

        # Initialize tracking
        self.best_val_loss = float('inf')
        self.metrics_history = {
            'train_loss': [], 'train_recall': [],
            'val_loss': [], 'val_recall': [], 'val_accuracy': []
        }
        
        # Setup save directory
        self.save_dir = Path(config['train']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Tuple[nn.Module, Dict[str, list]]:
        """
        Here, we run the complete training process
        """
        num_epochs = self.config['train']['epochs']
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update metrics history
            self._update_metrics(train_metrics, val_metrics)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoints
            self._save_checkpoint(epoch, val_metrics['loss'])
        
        # Save final metrics history
        self._save_metrics_history()
        
        return self.model, self.metrics_history
