import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTNet
from datetime import datetime
import os
from torch.utils.data import random_split

def train():
    # Set device
    device = torch.device("cpu")
    
    # More aggressive data augmentation while maintaining accuracy
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(5),  # Keep rotation
        transforms.RandomErasing(p=0.05, scale=(0.01, 0.03))  # Reduced probability and scale
    ])
    
    # Simpler transform for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load and split dataset with more training data
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_size = int(0.9 * len(full_dataset))  # Increased training size
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Smaller batch size for better training
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.NLLLoss()
    
    # Modified optimizer settings for better training accuracy
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,  # Back to previous value
        betas=(0.95, 0.999),  # Back to previous value
        eps=1e-8,
        weight_decay=0  # Keep weight decay at 0
    )
    
    total_steps = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,  # Back to previous value
        steps_per_epoch=total_steps,
        epochs=1,
        pct_start=0.05,  # Back to previous value
        div_factor=5,  # Back to previous value
        final_div_factor=50,  # Back to previous value
        anneal_strategy='linear'  # Keep linear annealing
    )
    
    # Training
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    best_accuracy = 0.0
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        total += target.size(0)
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Train Batch: {batch_idx}/{total_steps}, '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Acc: {100. * correct / total:.2f}%, '
                  f'LR: {current_lr:.6f}')
    
    train_accuracy = 100. * correct / total
    
    # Validation loop
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)
    
    val_accuracy = 100. * val_correct / val_total
    print(f'\nTraining Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    
    # Save model if validation accuracy is better
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'models/mnist_model_{timestamp}_acc{val_accuracy:.2f}.pth'
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), save_path)
    
    return val_accuracy, model

if __name__ == "__main__":
    train() 