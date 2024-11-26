import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Recall
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ModelTrainer:
    """
    Handles training and evaluation of MNIST Addition model
    """
    def __init__(self, model: nn.Module, dataloaders: Dict[str, DataLoader], config: Dict[str, Any], patience: int = 5,
                 min_delta: float = 0.001):
        self.model=model
        self.dataloaders=dataloaders
        self.config=config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model=self.model.to(self.device)

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

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
             'train_loss': [], 'train_accuracy': [], 'train_recall': [],
            'val_loss': [], 'val_accuracy': [], 'val_recall': []
        }
        
        # Setup save directory
        self.save_dir = Path(config['train']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save initial model architecture and configuration
        self._save_model_info()

    def _plot_metrics(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.metrics_history['train_loss'], label='Training Loss')
        ax1.plot(self.metrics_history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.metrics_history['train_accuracy'], label='Training Accuracy')
        ax2.plot(self.metrics_history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should be stopped early"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

        

    def _save_model_info(self):
        """
        Save model architecture and configuration
        """
        model_info = {
            'config': self.config,
            'model_architecture': str(self.model),  # Save model architecture as string
            'model_parameters': sum(p.numel() for p in self.model.parameters()),  # Total parameters
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        with open(self.save_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)

    def _save_checkpoint(self, epoch: int, metrics:Dict[str, float],is_best: bool = False):
        """Save model architecture and hyperparameter information"""
        serializable_metrics = self._convert_tensors(metrics)
        serializable_history = self._convert_tensors(self.metrics_history)
    
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer,  # Save complete optimizer
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': serializable_metrics,
            'metrics_history': serializable_history
        }
        # Save regular checkpoint
        #checkpoint_path = self.save_dir / f'epoch_{epoch}'
       #checkpoint_path.mkdir(parents=True, exist_ok=True)

        #torch.save(checkpoint, checkpoint_path / 'full_checkpoint.pt')  # Complete checkpoint
        #torch.save(self.model, checkpoint_path / 'model.pt')  # Just the model

        # Save best model if applicable
        if is_best:
            best_path = self.save_dir / 'best_model'
            best_path.mkdir(parents=True, exist_ok=True)

            torch.save(checkpoint, best_path / 'full_checkpoint.pt')
            torch.save(self.model.state_dict(), best_path / 'model.pt')
            
            # Save a summary of best performance
            best_summary = {
                'epoch': epoch,
                'metrics': serializable_metrics,
                'config': self.config
            }
            with open(best_path / 'best_summary.json', 'w') as f:
                json.dump(best_summary, f, indent=4)
    def _convert_tensors(self, obj):
        """Recursively convert tensors in nested structures to Python numbers"""
        if torch.is_tensor(obj):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors(item) for item in obj]
        return obj
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for the current epoch"""
        epoch_dir = self.save_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
         # Convert any tensor values to Python numbers
        train_metrics = self._convert_tensors(train_metrics)
        val_metrics = self._convert_tensors(val_metrics)
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics
        }
        
        # Save metrics to JSON file
        with open(epoch_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nEpoch {epoch + 1}")
        print("Training Metrics:")
        for name, value in train_metrics.items():
            print(f"  {name}: {value.item() if torch.is_tensor(value) else value:.4f}")
        print("Validation Metrics:")
        for name, value in val_metrics.items():
            print(f"  {name}: {value.item() if torch.is_tensor(value) else value:.4f}")
    
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
            
            # Combine metrics
            metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }

            # Update metrics history
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                logging.info(f"{key}: {value:.4f}")
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Plot current progress
            #self._plot_metrics()
            
            # Check early stopping
            if self._check_early_stopping(val_metrics['loss']):
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break
            

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, 
                                   {**train_metrics, **val_metrics}, 
                                   is_best=True)
            
            
        
        # Save final metrics history
        with open(self.save_dir / 'final_metrics_history.json', 'w') as f:
            json.dump(self._convert_tensors(self.metrics_history), f, indent=4)
        
        return self.model, self.metrics_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Run one epoch of training with multiple metrics
        """
        self.model.train()
        running_loss=0.0
        running_accuracy=0.0
        running_recall=0.0

        for data,targets in self.dataloaders['train']:
            data,targets=data.to(self.device),targets.to(self.device)
            #forward pass
            self.optimizer.zero_grad()
            outputs=self.model(data)
            loss=self.criterion(outputs,targets)

            #backward pass
            loss.backward()
            self.optimizer.step()

            #update metrics
            running_loss+=loss.item()
            running_accuracy += self.train_accuracy(outputs, targets)
            running_recall += self.train_recall(outputs, targets)
        # Calculate average metrics
        num_batches = len(self.dataloaders['train'])
        return {
            'loss': running_loss / num_batches,
            'accuracy': running_accuracy / num_batches,
            'recall': running_recall / num_batches
        }
    def _validate_epoch(self)->Dict[str,float]:
        """
        Run one validation epoch with multiple metrics
        """
        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        running_recall = 0.0
        with torch.no_grad():
            for data, targets in self.dataloaders['val']:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                running_accuracy += self.val_accuracy(outputs, targets)
                running_recall += self.val_recall(outputs, targets)
        
        # Calculate average metrics
        num_batches = len(self.dataloaders['val'])
        return {
            'loss': running_loss / num_batches,
            'accuracy': running_accuracy / num_batches,
            'recall': running_recall / num_batches
        }
    def evalute(self,dataloader:DataLoader)->Dict[str,float]:
        """
        Evaluate the model on the test dataset
        """
        self.model.eval()
        running_loss = 0.0
        accuracy = Accuracy(task='multiclass', num_classes=19).to(self.device)
        recall = Recall(task='multiclass', num_classes=19).to(self.device)

        running_accuracy = 0.0
        running_recall = 0.0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                running_accuracy += accuracy(outputs, targets)
                running_recall += recall(outputs, targets)
        
        num_batches = len(dataloader)
        return {
            'loss': running_loss / num_batches,
            'accuracy': running_accuracy / num_batches,
            'recall': running_recall / num_batches
        }

