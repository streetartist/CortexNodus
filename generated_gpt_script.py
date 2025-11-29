import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_2 = nn.Embedding(100, 128)
        self.layer_3 = self._create_positional_encoding(128, 512, 0.1)
        self.layer_4 = self._create_gpt_block(128, 4, 512, 0.1)
        self.layer_5 = self._create_gpt_block(128, 4, 512, 0.1)
        self.layer_6 = self._create_gpt_block(128, 4, 512, 0.1)
        self.layer_7 = nn.LayerNorm([128])
        self.layer_8 = nn.Linear(128, 100)

    def _create_positional_encoding(self, d_model, max_len, dropout):
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                
                self.register_buffer('pe', pe)
                
            def forward(self, x):
                seq_len = x.size(1)
                x = x + self.pe[:, :seq_len, :]
                return self.dropout(x)
        
        return PositionalEncoding(d_model, max_len, dropout)
    def _create_gpt_block(self, d_model, nhead, dim_feedforward, dropout):
        class GPTBlock(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                self.ln2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
                
            def forward(self, x):
                seq_len = x.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                
                x_norm = self.ln1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
                x = x + attn_out
                
                x = x + self.ffn(self.ln2(x))
                
                return x
        
        return GPTBlock(d_model, nhead, dim_feedforward, dropout)
    
    def forward(self, x):
        out_2 = self.layer_2(x.long())
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)
        out_6 = self.layer_6(out_5)
        out_7 = self.layer_7(out_6)
        out_8 = self.layer_8(out_7)
        return out_8

def get_dataloaders():
    # Placeholder for WikiText-2
    print("WikiText-2 dataset loading is a placeholder in this generated script.")
    
    # Dummy data for demonstration
    train_loader = [(torch.randint(0, 100, (20, 32)), torch.randint(0, 100, (20, 32))) for _ in range(10)]
    test_loader = [(torch.randint(0, 100, (20, 32)), torch.randint(0, 100, (20, 32))) for _ in range(2)]
    
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
    
    for epoch in range(1, 20 + 1):
        print(f'\nEpoch {epoch}/20')
        
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
