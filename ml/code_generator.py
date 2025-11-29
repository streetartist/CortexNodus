"""
PyTorch Code Generator
Generates standalone PyTorch training scripts from graph definitions
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import json
import hashlib

# Import subgraph parser from designer
try:
    from ml.designer import parse_subgraph_dag
except ImportError:
    # Fallback or mock if designer is not available in path
    def parse_subgraph_dag(subgraph):
        return {}

def generate_pytorch_script(plan: Dict[str, Any]) -> str:
    """Generate a standalone PyTorch training script from plan"""
    
    # Parse model structure
    model_struct = plan.get("model", {})
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
    
    # Determine output node (the node connected to Loss)
    output_node_id = None
    loss_node_id = main_config.get("node_id")
    if loss_node_id:
        loss_sources = connections.get(str(loss_node_id), [])
        if loss_sources:
            output_node_id = str(loss_sources[0])
    
    if not output_node_id and execution_order:
        valid_nodes = [n for n in execution_order if nodes.get(str(n), {}).get("type") != "Loss"]
        if valid_nodes:
            output_node_id = str(valid_nodes[-1])
    
    # Subgraph Registry: stores (class_name, code)
    # We use a list to maintain order of generation
    subgraph_registry = []
    # Cache to avoid regenerating same subgraph with same shapes: key=(json_str, input_shapes_str) -> class_name
    subgraph_cache = {}

    # Generate code sections
    imports = _generate_imports()
    
    # Generate Model Class (and recursively subgraphs)
    # Initial shape for Model
    if dataset_name == "CIFAR-10":
        initial_shape = (3, 32, 32)
    elif dataset_name in ["MNIST", "Fashion-MNIST"]:
        initial_shape = (1, 28, 28)
    elif dataset_name == "WikiText-2":
        initial_shape = (1,) 
    else:
        initial_shape = (1, 28, 28)
        
    # We treat the main model as a module with "data" as input
    # But _generate_module_class expects input_ids and input_shapes
    # For main model, input is implicit "x" from dataset.
    
    # We need a wrapper to start the recursion
    # The main model generation will trigger subgraph generation
    
    model_class_code, needed_helpers = _generate_module_class(
        class_name="Model",
        nodes=nodes,
        execution_order=execution_order,
        connections=connections,
        input_ids=["data"], # Virtual input ID
        output_ids=[output_node_id] if output_node_id else [],
        input_shapes=[initial_shape],
        subgraph_registry=subgraph_registry,
        subgraph_cache=subgraph_cache,
        num_classes=100 if dataset_name == "WikiText-2" else 10, # Approximate
        is_main_model=True
    )
    
    # Collect all subgraph codes
    subgraph_codes = "\n\n".join([code for _, code in subgraph_registry])
    
    dataset_code = _generate_dataset_code(dataset_name, batch_size)
    training_code = _generate_training_code(epochs, optimizer_name, lr, loss_name)
    
    # Helper methods (PositionalEncoding etc) need to be inside the classes that use them?
    # Or global?
    # The current implementation puts them inside the class.
    # If subgraphs need them, they should be inside subgraph classes.
    # _generate_module_class returns needed_helpers for THAT class.
    # We handle this inside _generate_module_class.
    
    # Combine all sections
    script = f"""{imports}

{subgraph_codes}

{model_class_code}

{dataset_code}

{training_code}

if __name__ == '__main__':
    main()
"""
    
    return script


def _generate_imports() -> str:
    return """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math"""


def _generate_module_class(class_name: str, nodes: Dict[str, Any], execution_order: List[int], 
                           connections: Dict[str, List], input_ids: List[str], output_ids: List[str],
                           input_shapes: List[Tuple], subgraph_registry: List, subgraph_cache: Dict,
                           num_classes: int, is_main_model: bool = False) -> Tuple[str, Set[str]]:
    
    # Track shapes: node_id -> shape tuple
    node_shapes = {}
    
    # Initialize input shapes
    # input_ids correspond to the arguments of forward()
    # For main model, input_ids=["data"], input_shapes=[initial_shape]
    for i, nid in enumerate(input_ids):
        if i < len(input_shapes):
            node_shapes[nid] = input_shapes[i]
        else:
            node_shapes[nid] = (1,) # Fallback
            
    layer_defs = []
    forward_lines = []
    needed_helpers = set()
    
    # Map input_ids to variable names in forward
    # Main model: forward(self, x) -> x maps to "data"
    # Subgraph: forward(self, in_0, in_1) -> in_0 maps to input_ids[0]
    
    if is_main_model:
        forward_args = "x"
        # Map "data" to "x"
        # We can just set node_shapes["data"] and use "x" when referencing "data"
        var_map = {"data": "x"}
    else:
        arg_names = [f"in_{i}" for i in range(len(input_ids))]
        forward_args = ", ".join(arg_names)
        var_map = {nid: arg for nid, arg in zip(input_ids, arg_names)}

    for node_id in execution_order:
        node_id_str = str(node_id)
        node = nodes.get(node_id_str, {})
        node_type = node.get("type", "")
        
        # Skip Data/Loss/Input/Output nodes in layer generation
        # But we need to track their shapes/flow
        if node_type in ["MNIST", "Fashion-MNIST", "CIFAR-10", "Loss", "CustomData", "WikiText-2", "graph/input", "graph/output"]:
            # For graph/input, shape is already set via input_ids
            # For graph/output, it just passes through
            
            # If it's a graph/output, we need to ensure it gets the value from its source
            if node_type == "graph/output":
                source_ids = connections.get(node_id_str, [])
                if source_ids:
                    src = str(source_ids[0])
                    if src in var_map:
                        var_map[node_id_str] = var_map[src]
                    if src in node_shapes:
                        node_shapes[node_id_str] = node_shapes[src]
            
            # For Data nodes in main model, shape is set
            continue
            
        # Get inputs
        source_ids = connections.get(node_id_str, [])
        current_input_shapes = []
        current_input_vars = []
        
        for src in source_ids:
            src_str = str(src)
            if src_str in node_shapes:
                current_input_shapes.append(node_shapes[src_str])
            else:
                # Fallback or error
                current_input_shapes.append((1,))
                
            if src_str in var_map:
                current_input_vars.append(var_map[src_str])
            else:
                # If source is not mapped, maybe it's "data" in main model?
                if is_main_model and src_str == "data":
                    current_input_vars.append("x")
                    current_input_shapes[-1] = node_shapes.get("data", (1, 28, 28))
                else:
                    current_input_vars.append(f"out_{src_str}") # Default naming
        
        if not current_input_shapes:
            continue
            
        layer_name = f"layer_{node_id}"
        
        # Generate Layer Info
        definition, forward_expr, out_shape, helpers = _get_layer_info(
            node_type, node, layer_name, current_input_shapes, current_input_vars, num_classes,
            subgraph_registry, subgraph_cache
        )
        
        if definition:
            layer_defs.append(f"self.{layer_name} = {definition}")
        
        # Assign output to variable
        out_var = f"out_{node_id_str}"
        forward_lines.append(f"{out_var} = {forward_expr}")
        var_map[node_id_str] = out_var
        
        node_shapes[node_id_str] = out_shape
        if helpers:
            needed_helpers.update(helpers)
    
    # Build class
    layers_init = "\n        ".join(layer_defs) if layer_defs else "pass"
    forward_body = "\n        ".join(forward_lines) if forward_lines else "pass"
    
    # Helper methods
    helper_code = _generate_helper_methods(needed_helpers)
    
    # Return statement
    if not output_ids:
        # Fallback: return last calculated variable
        if forward_lines:
            last_var = forward_lines[-1].split(" = ")[0].strip()
            return_stmt = f"return {last_var}"
        else:
            return_stmt = "return x" # Should not happen
    else:
        # Return tuple of outputs or single output
        ret_vars = []
        for oid in output_ids:
            if oid in var_map:
                ret_vars.append(var_map[oid])
            else:
                ret_vars.append("None")
        
        if len(ret_vars) == 1:
            return_stmt = f"return {ret_vars[0]}"
        else:
            return_stmt = f"return ({', '.join(ret_vars)})"
    
    class_code = f"""class {class_name}(nn.Module):
    def __init__(self):
        super({class_name}, self).__init__()
        {layers_init}
{helper_code}
    
    def forward(self, {forward_args}):
        {forward_body}
        {return_stmt}"""
    
    return class_code, needed_helpers


def _get_layer_info(node_type: str, node: Dict[str, Any], layer_name: str,
                    input_shapes: List[Tuple], input_vars: List[str], num_classes: int,
                    subgraph_registry: List, subgraph_cache: Dict) -> Tuple[str, str, Tuple, Set[str]]:
    
    definition = ""
    forward_expr = ""
    output_shape = input_shapes[0] if input_shapes else (1,)
    helpers = set()
    
    in_shape = input_shapes[0] if input_shapes else (1,)
    in_channels = in_shape[0]
    spatial_size = in_shape[1] if len(in_shape) == 3 else 0
    
    # --- Subgraph Handling ---
    if node_type == "graph/subgraph" or node_type == "graph/embedded_subgraph":
        subgraph_data = node.get("subgraph", {})
        # If empty, try to load from path? (Not implemented here for simplicity, assuming embedded)
        
        if subgraph_data:
            # Parse subgraph
            dag = parse_subgraph_dag(subgraph_data)
            
            # Generate unique class name based on content and input shapes
            # We include input_shapes in key because the generated class might depend on them (e.g. Linear layers)
            json_str = json.dumps(subgraph_data, sort_keys=True)
            shapes_str = str(input_shapes)
            cache_key = (json_str, shapes_str)
            
            if cache_key in subgraph_cache:
                sub_class_name = subgraph_cache[cache_key]
            else:
                # Generate new class
                sub_class_name = f"Subgraph_{len(subgraph_registry)}"
                sub_code, _ = _generate_module_class(
                    sub_class_name,
                    dag["nodes"],
                    dag["execution_order"],
                    dag["connections"],
                    dag["input_ids"],
                    dag["output_ids"],
                    input_shapes,
                    subgraph_registry,
                    subgraph_cache,
                    num_classes
                )
                subgraph_registry.append((sub_class_name, sub_code))
                subgraph_cache[cache_key] = sub_class_name
            
            definition = f"{sub_class_name}()"
            # Pass all inputs
            args = ", ".join(input_vars)
            forward_expr = f"self.{layer_name}({args})"
            
            # Determine output shape
            # We need to know the output shape of the subgraph
            # This is tricky because we just generated it but didn't get the output shape back from _generate_module_class easily
            # without running it.
            # Ideally _generate_module_class should return output shapes too.
            # For now, let's assume it preserves shape or use a heuristic?
            # Or we can simulate the subgraph flow here?
            # Let's assume output shape is same as input for now (Identity), 
            # OR we can improve _generate_module_class to return output shapes.
            # Given the complexity, let's assume (in_channels, spatial, spatial) if 3D, else (in_channels,)
            # This is a limitation. To fix, we'd need to run shape inference on the subgraph DAG here.
            output_shape = in_shape # Placeholder approximation
            
    # --- Standard Layers ---
    elif node_type == "Conv2D":
        out_channels = int(node.get("out_channels", 32))
        kernel_size = int(node.get("kernel_size", 3))
        stride = int(node.get("stride", 1))
        padding = int(node.get("padding", kernel_size // 2))
        
        definition = f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
        forward_expr = f"self.{layer_name}({input_vars[0]})"
        
        new_spatial = (spatial_size + 2 * padding - kernel_size) // stride + 1
        output_shape = (out_channels, new_spatial, new_spatial)
        
    elif node_type == "ReLU":
        forward_expr = f"F.relu({input_vars[0]})"
        
    elif node_type == "LeakyReLU":
        negative_slope = float(node.get("negative_slope", 0.01))
        forward_expr = f"F.leaky_relu({input_vars[0]}, negative_slope={negative_slope})"
        
    elif node_type == "Sigmoid":
        forward_expr = f"torch.sigmoid({input_vars[0]})"
        
    elif node_type == "Tanh":
        forward_expr = f"torch.tanh({input_vars[0]})"
        
    elif node_type == "MaxPool":
        kernel_size = int(node.get("kernel_size", 2))
        stride = int(node.get("stride", kernel_size))
        padding = int(node.get("padding", 0))
        
        forward_expr = f"F.max_pool2d({input_vars[0]}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
        new_spatial = (spatial_size + 2 * padding - kernel_size) // stride + 1
        output_shape = (in_channels, new_spatial, new_spatial)
        
    elif node_type == "AvgPool":
        kernel_size = int(node.get("kernel_size", 2))
        stride = int(node.get("stride", kernel_size))
        padding = int(node.get("padding", 0))
        
        forward_expr = f"F.avg_pool2d({input_vars[0]}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
        new_spatial = (spatial_size + 2 * padding - kernel_size) // stride + 1
        output_shape = (in_channels, new_spatial, new_spatial)
        
    elif node_type == "AdaptiveAvgPool":
        output_size = int(node.get("output_size", 1))
        forward_expr = f"F.adaptive_avg_pool2d({input_vars[0]}, ({output_size}, {output_size}))"
        output_shape = (in_channels, output_size, output_size)
        
    elif node_type == "BatchNorm2d":
        definition = f"nn.BatchNorm2d({in_channels})"
        forward_expr = f"self.{layer_name}({input_vars[0]})"
        
    elif node_type == "Dropout":
        p = float(node.get("p", 0.5))
        forward_expr = f"F.dropout({input_vars[0]}, p={p}, training=self.training)"
        
    elif node_type == "Flatten":
        forward_expr = f"{input_vars[0]}.view({input_vars[0]}.size(0), -1)"
        flat_size = in_channels
        if len(in_shape) == 3:
            flat_size = in_channels * spatial_size * spatial_size
        output_shape = (flat_size,)
        
    elif node_type in ["Dense", "Linear"]:
        out_features = int(node.get("out_features", num_classes))
        needs_flatten = len(in_shape) == 3
        
        if needs_flatten:
            in_features = in_channels * spatial_size * spatial_size
            forward_expr = f"self.{layer_name}({input_vars[0]}.view({input_vars[0]}.size(0), -1))"
        else:
            in_features = in_channels
            forward_expr = f"self.{layer_name}({input_vars[0]})"
            
        definition = f"nn.Linear({in_features}, {out_features})"
        output_shape = (out_features,)
        
    elif node_type == "Softmax":
        dim = int(node.get("dim", 1))
        forward_expr = f"F.softmax({input_vars[0]}, dim={dim})"
        
    elif node_type == "Add":
        if len(input_vars) > 1:
            summands = " + ".join(input_vars)
            forward_expr = summands
        else:
            forward_expr = input_vars[0]
            
    elif node_type == "Concat":
        dim = int(node.get("dim", 1))
        inputs_str = ", ".join(input_vars)
        forward_expr = f"torch.cat([{inputs_str}], dim={dim})"
        if dim == 1:
            total_channels = sum(s[0] for s in input_shapes)
            if len(in_shape) == 3:
                output_shape = (total_channels, spatial_size, spatial_size)
            else:
                output_shape = (total_channels,)
                
    elif node_type == "Multiply":
        if len(input_vars) > 1:
            factors = " * ".join(input_vars)
            forward_expr = factors
        else:
            forward_expr = input_vars[0]
            
    elif node_type == "Embedding":
        num_embeddings = int(node.get("num_embeddings", 1000))
        embedding_dim = int(node.get("embedding_dim", 128))
        definition = f"nn.Embedding({num_embeddings}, {embedding_dim})"
        forward_expr = f"self.{layer_name}({input_vars[0]}.long())"
        output_shape = (embedding_dim,)
        
    elif node_type == "PositionalEncoding":
        max_len = int(node.get("max_len", 512))
        d_model = in_channels
        dropout = float(node.get("dropout", 0.1))
        
        definition = f"self._create_positional_encoding({d_model}, {max_len}, {dropout})"
        forward_expr = f"self.{layer_name}({input_vars[0]})"
        helpers.add("PositionalEncoding")
        
    elif node_type == "GPTBlock":
        d_model = in_channels
        nhead = int(node.get("nhead", 8))
        dim_feedforward = int(node.get("dim_feedforward", 512))
        dropout = float(node.get("dropout", 0.1))
        
        definition = f"self._create_gpt_block({d_model}, {nhead}, {dim_feedforward}, {dropout})"
        forward_expr = f"self.{layer_name}({input_vars[0]})"
        helpers.add("GPTBlock")
        
    elif node_type == "LayerNorm":
        if len(in_shape) == 3:
            norm_shape = f"[{in_channels}, {spatial_size}, {spatial_size}]"
        else:
            norm_shape = f"[{in_channels}]"
            
        definition = f"nn.LayerNorm({norm_shape})"
        forward_expr = f"self.{layer_name}({input_vars[0]})"
        
    elif node_type == "Identity":
        forward_expr = input_vars[0]
        
    elif node_type == "Resize":
        size_val = node.get("size", 28)
        mode = node.get("mode", "bilinear")
        align_corners = node.get("align_corners", False)
        
        if isinstance(size_val, (list, tuple)):
            size = tuple(map(int, size_val))
        else:
            size = (int(size_val), int(size_val))
            
        if mode in ['bilinear', 'bicubic']:
            forward_expr = f"F.interpolate({input_vars[0]}, size={size}, mode='{mode}', align_corners={align_corners})"
        else:
            forward_expr = f"F.interpolate({input_vars[0]}, size={size}, mode='{mode}')"
            
        output_shape = (in_channels, size[0], size[1])

    else:
        if not forward_expr:
            forward_expr = input_vars[0]
            
    return definition, forward_expr, output_shape, helpers


def _generate_helper_methods(needed_helpers: Set[str]) -> str:
    methods = []
    
    if "PositionalEncoding" in needed_helpers:
        methods.append("""
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
        
        return PositionalEncoding(d_model, max_len, dropout)""")

    if "GPTBlock" in needed_helpers:
        methods.append("""
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
        
        return GPTBlock(d_model, nhead, dim_feedforward, dropout)""")

    return "".join(methods)


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
    
    elif dataset_name == "CIFAR-10":
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
    
    elif dataset_name == "WikiText-2":
        code = f"""def get_dataloaders():
    # Placeholder for WikiText-2
    print("WikiText-2 dataset loading is a placeholder in this generated script.")
    
    # Dummy data for demonstration
    train_loader = [(torch.randint(0, 100, ({batch_size}, 32)), torch.randint(0, 100, ({batch_size}, 32))) for _ in range(10)]
    test_loader = [(torch.randint(0, 100, ({batch_size}, 32)), torch.randint(0, 100, ({batch_size}, 32))) for _ in range(2)]
    
    return train_loader, test_loader"""
    
    else:
        code = f"""def get_dataloaders():
    # Custom dataset placeholder
    train_loader = []
    test_loader = []
    return train_loader, test_loader"""
    
    return code


def _generate_training_code(epochs: int, optimizer_name: str, lr: float, loss_name: str) -> str:
    """Generate training loop code"""
    
    if optimizer_name == "SGD":
        optimizer_code = f"optimizer = torch.optim.SGD(model.parameters(), lr={lr}, momentum=0.9)"
    elif optimizer_name == "AdamW":
        optimizer_code = f"optimizer = torch.optim.AdamW(model.parameters(), lr={lr})"
    else:
        optimizer_code = f"optimizer = torch.optim.Adam(model.parameters(), lr={lr})"
    
    if loss_name == "MSE":
        criterion_code = "criterion = nn.MSELoss()"
    elif loss_name == "SmoothL1":
        criterion_code = "criterion = nn.SmoothL1Loss()"
    else:
        criterion_code = "criterion = nn.CrossEntropyLoss()"
    
    code = f"""def train_epoch(model, train_loader, optimizer, criterion, device):
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
    print(f'Using device: {{device}}')
    
    train_loader, test_loader = get_dataloaders()
    
    model = Model().to(device)
    print(f'Model parameters: {{sum(p.numel() for p in model.parameters()):,}}')
    
    {optimizer_code}
    {criterion_code}
    
    best_accuracy = 0.0
    
    for epoch in range(1, {epochs} + 1):
        print(f'\\nEpoch {{epoch}}/{epochs}')
        
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy = validate(model, test_loader, device)
        
        print(f'Epoch {{epoch}} - Loss: {{avg_loss:.6f}}, Validation Accuracy: {{val_accuracy:.4f}}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Saved best model with accuracy: {{best_accuracy:.4f}}')
    
    print(f'\\nTraining completed! Best validation accuracy: {{best_accuracy:.4f}}')"""
    
    return code
