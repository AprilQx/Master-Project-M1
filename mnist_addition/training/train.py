import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Recall
from pathlib import Path
import  json
import logging
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader



def train_model(model: nn.Module, dataloaders,criterion, hyperparams,  num_epochs: int = 10, save_dir='checkpoints' ):
    """
    hyperparams = {
        'hidden_size': int,
        'num_layers': int, 
        'dropout_rate': float,
        'learning_rate': float,
        'batch_size': int
    }
    """
    save_dir=Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=model.to(device)

    best_val_loss = float('inf')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    metrics_history = {
        'train_loss': [], 'train_recall': [],
        'val_loss': [], 'val_recall': [], 'val_accuracy': []
    }
    
    for epoch in range(10): #fixed epochs for parameter search
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        metrics = train_epoch(model, dataloaders, 
                            criterion, optimizer, device)
        for key, value in metrics.items():
            metrics_history[key].append(value)
            logging.info(f"{key}: {value:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, save_dir / f'epoch_{epoch}.pt')
        
        # Save best model
        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            torch.save(checkpoint, save_dir / 'best_model.pt')
            
    # Save metrics history
    with open(save_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f)
        
    return model, metrics_history

def train_epoch(model, dataloaders, criterion, optimizer, device):
    metrics = {}
    recall = Recall(task='multiclass', num_classes=19).to(device)
    
    # Training
    model.train()
    train_loss = train_recall = 0
    for targets, data in dataloaders['train']:
        targets, data = targets.to(device), data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recall += recall(outputs, targets)
    
    # Validation
    model.eval()
    val_loss = val_recall = correct = total = 0
    with torch.no_grad():
        for targets, data in dataloaders['val']:
            targets, data = targets.to(device), data.to(device)
            outputs = model(data)
            val_loss += criterion(outputs, targets).item()
            val_recall += recall(outputs, targets)
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    metrics['train_loss'] = train_loss / len(dataloaders['train'])
    metrics['train_recall'] = train_recall / len(dataloaders['train'])
    metrics['val_loss'] = val_loss / len(dataloaders['val'])
    metrics['val_recall'] = val_recall / len(dataloaders['val'])
    metrics['val_accuracy'] = 100. * correct / total
    
    return metrics