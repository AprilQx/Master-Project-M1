import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader,val_loader,criterion, optimizer, num_epoches=10, device="mps" if torch.backends.mps.is_available() else "cpu" ):
    model=model.to(device)
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(num_epoches):
        model.train()
        train_loss=0
        for targets, data in train_loader:
            targets=targets.to(device)
            data=data.to(device)
            optimizer.zero_grad()
            outputs=model(data)
            loss=criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        
        model.eval()
        val_loss=0
        correct=0
        total=0
        with torch.no_grad():
            for targets, data in val_loader:
                targets=targets.to(device)
                data=data.to(device)
                outputs=model(data)
                loss=criterion(outputs,targets)
                val_loss+=loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict().copy()
        return best_weights, best_val_loss, val_accuracy
    
def eval_model(model, test_loader, device="mps" if torch.backends.mps.is_available() else "cpu"):
    model=model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for targets, data in test_loader:
            targets=targets.to(device)
            data=data.to(device)
            outputs=model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
