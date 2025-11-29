import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

class Subgraph_0(nn.Module):
    def __init__(self):
        super(Subgraph_0, self).__init__()
        self.layer_11 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer_14 = nn.Linear(25088, 10)

    
    def forward(self, in_0):
        out_11 = self.layer_11(in_0)
        out_12 = F.relu(out_11)
        out_14 = self.layer_14(out_12.view(out_12.size(0), -1))
        return out_14

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_2 = Subgraph_0()

    
    def forward(self, x):
        out_2 = self.layer_2(x)
        return out_2

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if isinstance(criterion, nn.CrossEntropyLoss) and output.dim() == 3:
             output = output.transpose(1, 2)
             
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)


def validate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if output.dim() == 3:
                pred = output.argmax(dim=2)
                correct += (pred == target).sum().item()
                total += target.numel()
            else:
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, test_loader = get_dataloaders()
    
    model = Model().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    
    for epoch in range(1, 10 + 1):
        print(f'\nEpoch {epoch}/10')
        
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy = validate(model, test_loader, device)
        
        print(f'Epoch {epoch} - Loss: {avg_loss:.6f}, Validation Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Saved best model with accuracy: {best_accuracy:.4f}')
    
    print(f'\nTraining completed! Best validation accuracy: {best_accuracy:.4f}')

if __name__ == '__main__':
    main()
