import os
import torch
import collections
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

def load_wikitext_dataset(data_dir="./data/wikitext-2", seq_length=35):
    """
    Load WikiText-2 dataset for language modeling.
    Returns train_ds, val_ds, test_ds, vocab_size
    """
    def read_tokens(file_path):
        if not os.path.exists(file_path):
            # Fallback if file doesn't exist (e.g. first run)
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Filter out empty lines and strip
        tokens = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('='):  # Skip headers
                tokens.extend(line.split())
        return tokens
    
    # Read files
    train_tokens = read_tokens(os.path.join(data_dir, 'wiki.train.tokens'))
    val_tokens = read_tokens(os.path.join(data_dir, 'wiki.valid.tokens'))
    test_tokens = read_tokens(os.path.join(data_dir, 'wiki.test.tokens'))
    
    if not train_tokens:
        # If no data found, return dummy data for testing
        print(f"Warning: WikiText-2 data not found in {data_dir}. Using dummy data.")
        vocab_size = 100
        dummy_seq = torch.randint(0, vocab_size, (100, seq_length))
        dummy_tgt = torch.randint(0, vocab_size, (100, seq_length))
        ds = TensorDataset(dummy_seq, dummy_tgt)
        return ds, ds, ds, vocab_size

    # Build vocabulary
    counter = collections.Counter(train_tokens)
    vocab = ['<unk>', '<eos>'] + sorted(counter.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    
    def tokens_to_ids(tokens):
        return [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
    
    # Convert to ids
    train_ids = tokens_to_ids(train_tokens)
    val_ids = tokens_to_ids(val_tokens)
    test_ids = tokens_to_ids(test_tokens)
    
    # Create sequences
    def create_sequences(ids, seq_len):
        sequences = []
        targets = []
        for i in range(len(ids) - seq_len):
            seq = ids[i:i+seq_len]
            tgt = ids[i+1:i+seq_len+1]
            sequences.append(seq)
            targets.append(tgt)
        if not sequences:
            return torch.empty(0, seq_len, dtype=torch.long), torch.empty(0, seq_len, dtype=torch.long)
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    train_seq, train_tgt = create_sequences(train_ids, seq_length)
    val_seq, val_tgt = create_sequences(val_ids, seq_length)
    test_seq, test_tgt = create_sequences(test_ids, seq_length)
    
    train_ds = TensorDataset(train_seq, train_tgt)
    val_ds = TensorDataset(val_seq, val_tgt)
    test_ds = TensorDataset(test_seq, test_tgt)
    
    return train_ds, val_ds, test_ds, vocab_size

def get_dataset(dataset_name, batch_size, data_props=None):
    """
    Factory function to get dataloaders and metadata for a dataset.
    """
    if data_props is None:
        data_props = {}
        
    train_ds = None
    test_ds = None
    in_channels = 1
    num_classes = 10
    
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        
    elif dataset_name == "Fashion-MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        
    elif dataset_name == "CIFAR-10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        in_channels = 3
        num_classes = 10
        
    elif dataset_name in ["WikiText-2", "WikiText-103", "PennTreebank"]:
        seq_length = int(data_props.get("seq_length", 35))
        train_ds, val_ds, test_ds, vocab_size = load_wikitext_dataset(data_dir=f"./data/{dataset_name.lower()}", seq_length=seq_length)
        # For language modeling, in_channels is seq_length (input size) and num_classes is vocab_size
        in_channels = seq_length
        num_classes = vocab_size
        
    elif dataset_name == "CustomData":
        data_path = data_props.get("path", "")
        data_type = data_props.get("type", "ImageFolder")
        
        if not data_path or not os.path.exists(data_path):
             # Fallback for demo if path invalid
             pass 

        if data_type == "ImageFolder" and os.path.exists(data_path):
            transform = transforms.Compose([
                transforms.Resize((128, 128)), 
                transforms.ToTensor(),
            ])
            full_ds = datasets.ImageFolder(root=data_path, transform=transform)
            in_channels = 3
            num_classes = len(full_ds.classes)
            
            total_len = len(full_ds)
            train_len = int(0.8 * total_len)
            val_len = total_len - train_len
            train_ds, test_ds = random_split(full_ds, [train_len, val_len])
            
        elif data_type == "CSV" and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            x = df.iloc[:, :-1].values.astype('float32')
            y = df.iloc[:, -1].values.astype('int64')
            
            full_ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
            in_channels = x.shape[1]
            num_classes = len(np.unique(y))
            
            total_len = len(full_ds)
            train_len = int(0.8 * total_len)
            val_len = total_len - train_len
            train_ds, test_ds = random_split(full_ds, [train_len, val_len])
            
        elif data_type == "Numpy" and os.path.exists(data_path):
            data = np.load(data_path)
            if 'x' in data and 'y' in data:
                x = torch.from_numpy(data['x'].astype('float32'))
                y = torch.from_numpy(data['y'].astype('int64'))
                full_ds = TensorDataset(x, y)
                in_channels = x.shape[1]
                num_classes = len(torch.unique(y))
                
                total_len = len(full_ds)
                train_len = int(0.8 * total_len)
                val_len = total_len - train_len
                train_ds, test_ds = random_split(full_ds, [train_len, val_len])
        
        # Fallback demo data if custom loading failed
        if train_ds is None:
             # Create dummy data
             print("Using dummy custom data")
             x = torch.randn(100, 3, 32, 32)
             y = torch.randint(0, 10, (100,))
             train_ds = TensorDataset(x, y)
             test_ds = TensorDataset(x, y)
             in_channels = 3
             num_classes = 10

    if train_ds is None:
        raise ValueError(f"Could not load dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, in_channels, num_classes
