import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Recall
from pathlib import Path
import  json

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, output, target):
        return self.loss(output, target)



def train_model(model, train_loader,val_loader,criterion, optimizer, num_epoches=10, save_dir='checkpoints', device="mps" if torch.backends.mps.is_available() else "cpu" ):
    
    save_dir=Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    model=model.to(device)
    best_val_loss = float('inf')
    criterion=CustomLoss()
    recall = Recall(task='multiclass', num_classes=19).to(device)

    metrics_history = {
        'train_loss': [], 'train_recall': [],
        'val_loss': [], 'val_recall': [], 'val_accuracy': []
    }
    
    for epoch in range(num_epoches):
        metrics = train_epoch(model, train_loader, val_loader, 
                            criterion, optimizer, recall, device)
        for key, value in metrics.items():
            metrics_history[key].append(value)
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, save_dir / f'epoch_{epoch}.pt')
        
        # Save best model
        if epoch == 0 or metrics['val_loss'] < min(metrics_history['val_loss'][:-1]):
            torch.save(checkpoint, save_dir / 'best_model.pt')
            
    # Save metrics history
    with open(save_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f)
        
    return model, metrics_history

def train_epoch(model, train_loader, val_loader, criterion, optimizer, recall, device):
    model.train()
    metrics = {}
    
    # Training
    train_loss = train_recall = 0
    for targets, data in train_loader:
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
        for targets, data in val_loader:
            targets, data = targets.to(device), data.to(device)
            outputs = model(data)
            val_loss += criterion(outputs, targets).item()
            val_recall += recall(outputs, targets)
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    metrics['train_loss'] = train_loss / len(train_loader)
    metrics['train_recall'] = train_recall / len(train_loader)
    metrics['val_loss'] = val_loss / len(val_loader)
    metrics['val_recall'] = val_recall / len(val_loader)
    metrics['val_accuracy'] = 100. * correct / total
    
    return metrics
    
