"""
PyTorch Code Generator
Generates standalone PyTorch training scripts from graph definitions
"""

from typing import Dict, Any, List, Tuple
import json


def generate_pytorch_script(plan: Dict[str, Any]) -> str:
    """Generate a standalone PyTorch training script from plan"""
    
    # Parse model structure
    model_struct = plan.get("model", {})
    connections = model_struct.get("connections", {})
    
    # Check for non-linear structures (branches/merges)
    for node_id, sources in connections.items():
        if len(sources) > 1:
            raise ValueError("Code generation for models with branches or merges is not currently supported.")
    
    # Check for multiple outputs from a single node (another form of branching)
    source_counts = {}
    for node_id, sources in connections.items():
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
            if source_counts[source] > 1:
                raise ValueError("Code generation for models with branches is not currently supported.")
    nodes = model_struct.get("nodes", {})
    execution_order = model_struct.get("execution_order", [])
    connections = model_struct.get("connections", {})
    
    # Dataset info
    data_info = plan.get("data", {})
    dataset_name = data_info.get("dataset", "MNIST")
    batch_size = data_info.get("batch_size", 32)
    
    # Training info
    train_configs = plan.get("train", [])
    if not train_configs:
        train_configs = [{"epochs": 10, "optimizer": "Adam", "lr": 0.001, "loss": "CrossEntropy"}]
    
    main_config = train_configs[0]
    epochs = main_config.get("epochs", 10)
    optimizer_name = main_config.get("optimizer", "Adam")
    lr = main_config.get("lr", 0.001)
    loss_name = main_config.get("loss", "CrossEntropy")
    
    # Generate code sections
    imports = _generate_imports(dataset_name)
    model_class = _generate_model_class(nodes, execution_order, connections, dataset_name)
    dataset_code = _generate_dataset_code(dataset_name, batch_size)
    training_code = _generate_training_code(epochs, optimizer_name, lr, loss_name)
    
    # Combine all sections
    script = f"""{imports}

{model_class}

{dataset_code}

{training_code}

if __name__ == '__main__':
    main()
"""
    
    return script


def _generate_imports(dataset_name: str) -> str:
    """Generate import statements"""
    base_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms"""
    
    return base_imports


def _generate_model_class(nodes: Dict[str, Any], execution_order: List[int], 
                          connections: Dict[str, List], dataset_name: str) -> str:
    """Generate the model class definition"""
    
    # Determine input shape
    if dataset_name == "CIFAR-10":
        in_channels = 3
        num_classes = 10
        spatial_size = 32
    else:  # MNIST, Fashion-MNIST
        in_channels = 1
        num_classes = 10
        spatial_size = 28
    
    # Build layers
    layer_defs = []
    layer_forwards = []
    current_channels = in_channels
    current_spatial_size = spatial_size
    is_flattened = False
    
    for i, node_id in enumerate(execution_order):
        node_id_str = str(node_id)
        node = nodes.get(node_id_str, {})
        node_type = node.get("type", "")
        
        # Skip data and loss nodes
        if node_type in ["MNIST", "Fashion-MNIST", "CIFAR-10", "Loss", "CustomData"]:
            continue
        
        layer_name = f"layer_{i}"
        layer_code, forward_code, current_channels, current_spatial_size, is_flattened = _generate_layer_code(
            node_type, node, layer_name, current_channels, current_spatial_size, is_flattened, num_classes
        )
        
        if layer_code:
            layer_defs.append(layer_code)
        if forward_code:
            layer_forwards.append(forward_code)
    
    # Check if we need helper methods
    needs_pos_encoding = any("_create_positional_encoding" in ld for ld in layer_defs)
    needs_gpt_block = any("_create_gpt_block" in ld for ld in layer_defs)
    
    # Build class
    layers_init = "\n        ".join(layer_defs) if layer_defs else "pass"
    forward_calls = "\n        ".join(layer_forwards) if layer_forwards else "pass"
    
    # Generate helper methods
    helper_methods = []
    
    if needs_pos_encoding:
        helper_methods.append("""
    def _create_positional_encoding(self, d_model, max_len, dropout):
        \"\"\"Create Positional Encoding layer\"\"\"
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                
                self.register_buffer('pe', pe)
                
            def forward(self, x):
                seq_len = x.size(1)
                x = x + self.pe[:, :seq_len, :]
                return self.dropout(x)
        
        return PositionalEncoding(d_model, max_len, dropout)""")
    
    if needs_gpt_block:
        helper_methods.append("""
    def _create_gpt_block(self, d_model, nhead, dim_feedforward, dropout):
        \"\"\"Create GPT Block layer\"\"\"
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
                # Pre-norm architecture
                seq_len = x.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                
                x_norm = self.ln1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
                x = x + attn_out
                
                x = x + self.ffn(self.ln2(x))
                
                return x
        
        return GPTBlock(d_model, nhead, dim_feedforward, dropout)""")
    
    helper_methods_code = "".join(helper_methods)
    
    model_class = f"""class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        {layers_init}
{helper_methods_code}
    
    def forward(self, x):
        {forward_calls}
        return x"""
    
    return model_class


def _generate_layer_code(node_type: str, node: Dict[str, Any], layer_name: str,
                         current_channels: int, current_spatial_size: int, is_flattened: bool, 
                         num_classes: int) -> Tuple[str, str, int, int, bool]:
    """Generate code for a single layer"""
    
    layer_def = ""
    forward_code = ""
    new_channels = current_channels
    new_spatial_size = current_spatial_size
    new_flattened = is_flattened
    
    if node_type == "Conv2D":
        out_channels = int(node.get("out_channels", 32))
        kernel_size = int(node.get("kernel_size", 3))
        stride = int(node.get("stride", 1))
        padding = int(node.get("padding", kernel_size // 2))
        
        layer_def = f"self.{layer_name} = nn.Conv2d({current_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = out_channels
        new_spatial_size = (new_spatial_size + 2 * padding - kernel_size) // stride + 1
        
    elif node_type == "ReLU":
        forward_code = f"x = F.relu(x)"
        
    elif node_type == "LeakyReLU":
        negative_slope = float(node.get("negative_slope", 0.01))
        forward_code = f"x = F.leaky_relu(x, negative_slope={negative_slope})"
        
    elif node_type == "Sigmoid":
        forward_code = f"x = torch.sigmoid(x)"
        
    elif node_type == "Tanh":
        forward_code = f"x = torch.tanh(x)"
        
    elif node_type == "MaxPool":
        kernel_size = int(node.get("kernel_size", 2))
        stride = int(node.get("stride", kernel_size))
        forward_code = f"x = F.max_pool2d(x, kernel_size={kernel_size}, stride={stride})"
        new_spatial_size = (new_spatial_size - kernel_size) // stride + 1
        
    elif node_type == "AvgPool":
        kernel_size = int(node.get("kernel_size", 2))
        stride = int(node.get("stride", kernel_size))
        forward_code = f"x = F.avg_pool2d(x, kernel_size={kernel_size}, stride={stride})"
        new_spatial_size = (new_spatial_size - kernel_size) // stride + 1
        
    elif node_type == "AdaptiveAvgPool":
        output_size = int(node.get("output_size", 1))
        forward_code = f"x = F.adaptive_avg_pool2d(x, ({output_size}, {output_size}))"
        new_spatial_size = output_size
        
    elif node_type == "BatchNorm2d":
        layer_def = f"self.{layer_name} = nn.BatchNorm2d({current_channels})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "Dropout":
        p = float(node.get("p", 0.5))
        forward_code = f"x = F.dropout(x, p={p}, training=self.training)"
        
    elif node_type == "Dropout2d":
        p = float(node.get("p", 0.5))
        forward_code = f"x = F.dropout2d(x, p={p}, training=self.training)"
        
    elif node_type == "Flatten":
        forward_code = f"x = x.view(x.size(0), -1)"
        new_channels = current_channels * new_spatial_size * new_spatial_size
        new_spatial_size = 1
        new_flattened = True
        
    elif node_type in ["Dense", "Linear"]:
        out_features = int(node.get("out_features", num_classes))
        
        # If not flattened, need to flatten first
        if not is_flattened:
            in_features = current_channels * new_spatial_size * new_spatial_size
            forward_code = f"x = x.view(x.size(0), -1)\n        "
            new_flattened = True
            new_spatial_size = 1
        else:
            in_features = current_channels
        
        layer_def = f"self.{layer_name} = nn.Linear({in_features}, {out_features})"
        forward_code += f"x = self.{layer_name}(x)"
        new_channels = out_features
        
    elif node_type == "Softmax":
        dim = int(node.get("dim", 1))
        forward_code = f"x = F.softmax(x, dim={dim})"
        
    elif node_type == "ConvTranspose2d":
        out_channels = int(node.get("out_channels", 32))
        kernel_size = int(node.get("kernel_size", 3))
        stride = int(node.get("stride", 1))
        padding = int(node.get("padding", 0))
        output_padding = int(node.get("output_padding", 0))
        
        layer_def = f"self.{layer_name} = nn.ConvTranspose2d({current_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = out_channels
        new_spatial_size = (new_spatial_size - 1) * stride - 2 * padding + kernel_size + output_padding
        
    elif node_type == "BatchNorm1d":
        layer_def = f"self.{layer_name} = nn.BatchNorm1d({current_channels})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "InstanceNorm2d":
        layer_def = f"self.{layer_name} = nn.InstanceNorm2d({current_channels})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "GroupNorm":
        num_groups = int(node.get("num_groups", 4))
        layer_def = f"self.{layer_name} = nn.GroupNorm({num_groups}, {current_channels})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "LayerNorm":
        if not is_flattened and new_spatial_size > 1:
            normalized_shape = f"[{current_channels}, {new_spatial_size}, {new_spatial_size}]"
        else:
            normalized_shape = f"[{current_channels}]"
        layer_def = f"self.{layer_name} = nn.LayerNorm({normalized_shape})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "ELU":
        alpha = float(node.get("alpha", 1.0))
        forward_code = f"x = F.elu(x, alpha={alpha})"
        
    elif node_type == "GELU":
        forward_code = f"x = F.gelu(x)"
        
    elif node_type == "SiLU":
        forward_code = f"x = F.silu(x)"
        
    elif node_type == "LogSoftmax":
        dim = int(node.get("dim", 1))
        forward_code = f"x = F.log_softmax(x, dim={dim})"
        
    elif node_type == "AlphaDropout":
        p = float(node.get("p", 0.5))
        layer_def = f"self.{layer_name} = nn.AlphaDropout(p={p})"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "Upsample":
        scale_factor = float(node.get("scale_factor", 2.0))
        mode = node.get("mode", "nearest")
        forward_code = f"x = F.interpolate(x, scale_factor={scale_factor}, mode='{mode}')"
        new_spatial_size = int(new_spatial_size * scale_factor)
        
    elif node_type == "PixelShuffle":
        upscale_factor = int(node.get("upscale_factor", 2))
        layer_def = f"self.{layer_name} = nn.PixelShuffle({upscale_factor})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = current_channels // (upscale_factor ** 2)
        new_spatial_size = new_spatial_size * upscale_factor
        
    elif node_type == "ZeroPad2d":
        padding = int(node.get("padding", 1))
        forward_code = f"x = F.pad(x, ({padding}, {padding}, {padding}, {padding}))"
        new_spatial_size = new_spatial_size + 2 * padding
        
    elif node_type == "PReLU":
        layer_def = f"self.{layer_name} = nn.PReLU()"
        forward_code = f"x = self.{layer_name}(x)"
        
    elif node_type == "Softplus":
        forward_code = f"x = F.softplus(x)"
        
    elif node_type == "Hardswish":
        forward_code = f"x = F.hardswish(x)"
        
    elif node_type == "Hardsigmoid":
        forward_code = f"x = F.hardsigmoid(x)"
        
    elif node_type == "Resize":
        size_val = node.get("size", 28)
        mode = node.get("mode", "bilinear")
        align_corners = node.get("align_corners", False)
        
        if isinstance(size_val, (list, tuple)):
            size = tuple(map(int, size_val))
        else:
            size = (int(size_val), int(size_val))
        
        if mode in ['bilinear', 'bicubic']:
            forward_code = f"x = F.interpolate(x, size={size}, mode='{mode}', align_corners={align_corners})"
        else:
            forward_code = f"x = F.interpolate(x, size={size}, mode='{mode}')"
        new_spatial_size = size[0]
        
    elif node_type == "Embedding":
        num_embeddings = int(node.get("num_embeddings", num_classes))
        embedding_dim = int(node.get("embedding_dim", 128))
        
        layer_def = f"self.{layer_name} = nn.Embedding({num_embeddings}, {embedding_dim})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = embedding_dim
        new_spatial_size = 0  # Not spatial anymore
        new_flattened = True  # Sequence data
        
    elif node_type == "PositionalEncoding":
        max_len = int(node.get("max_len", 512))
        d_model = current_channels if current_channels > 0 else int(node.get("d_model", 128))
        dropout_p = float(node.get("dropout", 0.1))
        
        # Generate PositionalEncoding class
        layer_def = f"self.{layer_name} = self._create_positional_encoding({d_model}, {max_len}, {dropout_p})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = d_model
        
    elif node_type in ["TransformerEncoderLayer", "TransformerEncoder"]:
        d_model = current_channels if current_channels > 0 else int(node.get("d_model", 128))
        nhead = int(node.get("nhead", 8))
        dim_feedforward = int(node.get("dim_feedforward", 512))
        dropout_p = float(node.get("dropout", 0.1))
        
        if node_type == "TransformerEncoderLayer":
            layer_def = f"self.{layer_name} = nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout_p}, batch_first=True)"
        else:  # TransformerEncoder
            num_layers = int(node.get("num_layers", 6))
            layer_def = f"self.{layer_name} = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout_p}, batch_first=True), num_layers={num_layers})"
        
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = d_model
        
    elif node_type in ["TransformerDecoderLayer", "TransformerDecoder"]:
        d_model = current_channels if current_channels > 0 else int(node.get("d_model", 128))
        nhead = int(node.get("nhead", 8))
        dim_feedforward = int(node.get("dim_feedforward", 512))
        dropout_p = float(node.get("dropout", 0.1))
        
        if node_type == "TransformerDecoderLayer":
            layer_def = f"self.{layer_name} = nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout_p}, batch_first=True)"
            forward_code = f"# Note: TransformerDecoderLayer requires a 'memory' input from an encoder.\n        # This generated code assumes self-attention (x is passed as memory).\n        # You may need to modify this for your specific use case.\n        memory = x\n        x = self.{layer_name}(x, memory)"
        else:  # TransformerDecoder
            num_layers = int(node.get("num_layers", 6))
            layer_def = f"self.{layer_name} = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout_p}, batch_first=True), num_layers={num_layers})"
            forward_code = f"# Note: TransformerDecoder requires a 'memory' input from an encoder.\n        # This generated code assumes self-attention (x is passed as memory).\n        # You may need to modify this for your specific use case.\n        memory = x\n        x = self.{layer_name}(x, memory)"
        
        new_channels = d_model
        
    elif node_type == "MultiheadAttention":
        embed_dim = current_channels if current_channels > 0 else int(node.get("embed_dim", 128))
        num_heads = int(node.get("num_heads", 8))
        dropout_p = float(node.get("dropout", 0.0))
        
        layer_def = f"self.{layer_name} = nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout_p}, batch_first=True)"
        forward_code = f"x, _ = self.{layer_name}(x, x, x)  # Self-attention"
        new_channels = embed_dim
        
    elif node_type == "GPTBlock":
        d_model = current_channels if current_channels > 0 else int(node.get("d_model", 128))
        nhead = int(node.get("nhead", 8))
        dim_feedforward = int(node.get("dim_feedforward", 512))
        dropout_p = float(node.get("dropout", 0.1))
        
        # GPTBlock is complex, we'll create it as a helper method
        layer_def = f"self.{layer_name} = self._create_gpt_block({d_model}, {nhead}, {dim_feedforward}, {dropout_p})"
        forward_code = f"x = self.{layer_name}(x)"
        new_channels = d_model
        
    elif node_type == "Identity":
        forward_code = f"# Identity layer - no operation"
        
    # Note: Merge operations (Add, Multiply, Concat) require multiple inputs
    # These are not yet fully supported in linear architectures
    # Would require DAG-aware code generation
        
    return layer_def, forward_code, new_channels, new_spatial_size, new_flattened


def _generate_dataset_code(dataset_name: str, batch_size: int) -> str:
    """Generate dataset loading code"""
    
    if dataset_name == "MNIST":
        code = f"""def get_dataloaders():
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
    
    train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
    
    return train_loader, test_loader"""
    
    elif dataset_name == "Fashion-MNIST":
        code = f"""def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
    
    return train_loader, test_loader"""
    
    else:  # CIFAR-10
        code = f"""def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
    
    return train_loader, test_loader"""
    
    return code


def _generate_training_code(epochs: int, optimizer_name: str, lr: float, loss_name: str) -> str:
    """Generate training loop code"""
    
    # Optimizer setup
    if optimizer_name == "SGD":
        optimizer_code = f"optimizer = torch.optim.SGD(model.parameters(), lr={lr}, momentum=0.9)"
    elif optimizer_name == "AdamW":
        optimizer_code = f"optimizer = torch.optim.AdamW(model.parameters(), lr={lr})"
    else:  # Adam
        optimizer_code = f"optimizer = torch.optim.Adam(model.parameters(), lr={lr})"
    
    # Loss function
    if loss_name == "MSE":
        criterion_code = "criterion = nn.MSELoss()"
    elif loss_name == "SmoothL1":
        criterion_code = "criterion = nn.SmoothL1Loss()"
    else:  # CrossEntropy
        criterion_code = "criterion = nn.CrossEntropyLoss()"
    
    code = f"""def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {{batch_idx}}/{{len(train_loader)}}, Loss: {{loss.item():.6f}}')
    
    return total_loss / len(train_loader)


def validate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {{device}}')
    
    # Data
    train_loader, test_loader = get_dataloaders()
    
    # Model
    model = Model().to(device)
    print(f'Model parameters: {{sum(p.numel() for p in model.parameters()):,}}')
    
    # Optimizer and loss
    {optimizer_code}
    {criterion_code}
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, {epochs} + 1):
        print(f'\\nEpoch {{epoch}}/{epochs}')
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_accuracy = validate(model, test_loader, device)
        
        print(f'Epoch {{epoch}} - Loss: {{avg_loss:.6f}}, Validation Accuracy: {{val_accuracy:.4f}}')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Saved best model with accuracy: {{best_accuracy:.4f}}')
    
    print(f'\\nTraining completed! Best validation accuracy: {{best_accuracy:.4f}}')"""
    
    return code
