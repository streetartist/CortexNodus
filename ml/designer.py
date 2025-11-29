from typing import Any, Dict, List, Tuple, Callable, Optional
from torch import nn
import torch

# --- Layer Registry ---

class LayerContext:
    def __init__(self, in_channels: int, spatial_size: int, num_classes: int):
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.out_channels = in_channels  # Updated by layers
        self.out_spatial_size = spatial_size # Updated by layers
        self.shapes = {} # node_id -> (C, H, W) or (C,)
        self.input_shapes = [] # List of shapes for the current node being built

class SubgraphModule(nn.Module):
    def __init__(self, layers_map: Dict[str, nn.Module], execution_order: List[str], connections: Dict[str, List[str]], input_nodes: List[str], output_nodes: List[str]):
        super().__init__()
        self.layers = nn.ModuleDict(layers_map)
        self.execution_order = execution_order
        self.connections = connections
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def forward(self, *inputs):
        outputs = {}
        # Map inputs
        for i, nid in enumerate(self.input_nodes):
            if i < len(inputs):
                outputs[nid] = inputs[i]
        
        for nid in self.execution_order:
            # If input node, skip (already in outputs)
            if nid in outputs and nid not in self.layers:
                continue
                
            input_ids = self.connections.get(nid, [])
            
            node_inputs = []
            for iid in input_ids:
                val = outputs.get(iid)
                if val is not None:
                    node_inputs.append(val)
            
            # If node has connections but no inputs were resolved, skip
            if input_ids and not node_inputs:
                continue
                
            if nid in self.layers:
                mod = self.layers[nid]
                if isinstance(mod, (MergeAdd, MergeConcat, MergeMultiply)):
                    out = mod(node_inputs)
                elif isinstance(mod, nn.Identity):
                    out = node_inputs[0] if node_inputs else None
                else:
                    out = mod(node_inputs[0]) if node_inputs else None
                
                if out is not None:
                    outputs[nid] = out
                
        res = []
        for nid in self.output_nodes:
            res.append(outputs.get(nid))
            
        if len(res) == 1:
            return res[0]
        return tuple(res)

class MergeAdd(nn.Module):
    def forward(self, inputs):
        if not inputs:
            return None
        # Ensure all inputs are tensors
        valid_inputs = [i for i in inputs if isinstance(i, torch.Tensor)]
        if not valid_inputs:
            return None
        # Check shapes? For now assume they match or broadcast.
        # Start with first
        res = valid_inputs[0]
        for i in range(1, len(valid_inputs)):
            res = res + valid_inputs[i]
        return res

class MergeConcat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        
    def forward(self, inputs):
        if not inputs:
            return None
        valid_inputs = [i for i in inputs if isinstance(i, torch.Tensor)]
        if not valid_inputs:
            return None
        return torch.cat(valid_inputs, dim=self.dim)

class MergeMultiply(nn.Module):
    def forward(self, inputs):
        if not inputs:
            return None
        valid_inputs = [i for i in inputs if isinstance(i, torch.Tensor)]
        if not valid_inputs:
            return None
        res = valid_inputs[0]
        for i in range(1, len(valid_inputs)):
            res = res * valid_inputs[i]
        return res

class GraphModule(nn.Module):
    def __init__(self, layers_map: Dict[str, nn.Module], execution_order: List[int], connections: Dict[str, List[Any]]):
        super().__init__()
        self.layers = nn.ModuleDict(layers_map)
        self.execution_order = execution_order
        self.connections = connections

    def forward(self, x):
        # 'data' is the reserved ID for input data
        outputs = {'data': x}
        
        for node_id in self.execution_order:
            nid_str = str(node_id)
            if nid_str not in self.layers:
                continue
                
            # Get inputs
            input_ids = self.connections.get(nid_str, [])
            if not input_ids:
                continue
            
            # Collect all available inputs
            inputs = []
            for iid in input_ids:
                val = outputs.get(str(iid))
                if val is not None:
                    inputs.append(val)
            
            if not inputs:
                continue
                
            mod = self.layers[nid_str]
            
            # Check if module expects list of inputs
            if isinstance(mod, (MergeAdd, MergeConcat, MergeMultiply)):
                out = mod(inputs)
            else:
                # Default: take first input
                out = mod(inputs[0])
            
            outputs[nid_str] = out
            
        return outputs

LayerBuilder = Callable[[Dict[str, Any], LayerContext], nn.Module]
LAYER_REGISTRY: Dict[str, LayerBuilder] = {}
SUBGRAPH_CACHE: Dict[str, Dict[str, Any]] = {}

def register_layer(name: str):
    def decorator(func: LayerBuilder):
        LAYER_REGISTRY[name] = func
        return func
    return decorator

# --- Layer Implementations ---

@register_layer("Conv2D")
def build_conv2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    out_c = int(props.get("out_channels", 8))
    k = int(props.get("kernel_size", 3))
    stride = int(props.get("stride", 1))
    groups = int(props.get("groups", 1))
    # Default padding logic: if not specified, use k//2 (same padding for stride 1)
    # But if user specifies, use it.
    if "padding" in props:
        padding = int(props["padding"])
    else:
        padding = k // 2
        
    layer = nn.Conv2d(ctx.out_channels, out_c, kernel_size=k, stride=stride, padding=padding, groups=groups)
    
    ctx.out_channels = out_c
    # H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    # Assuming dilation=1
    ctx.out_spatial_size = (ctx.out_spatial_size + 2 * padding - 1 * (k - 1) - 1) // stride + 1
    
    return layer

@register_layer("ConvTranspose2d")
def build_conv_transpose2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    out_c = int(props.get("out_channels", 8))
    k = int(props.get("kernel_size", 3))
    stride = int(props.get("stride", 1))
    padding = int(props.get("padding", 0))
    output_padding = int(props.get("output_padding", 0))
    
    layer = nn.ConvTranspose2d(ctx.out_channels, out_c, kernel_size=k, stride=stride, padding=padding, output_padding=output_padding)
    
    # Update context
    ctx.out_channels = out_c
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # Assuming dilation=1
    ctx.out_spatial_size = (ctx.out_spatial_size - 1) * stride - 2 * padding + 1 * (k - 1) + output_padding + 1
    
    return layer

@register_layer("ReLU")
def build_relu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.ReLU(inplace=True)

@register_layer("LeakyReLU")
def build_leaky_relu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    negative_slope = float(props.get("negative_slope", 0.01))
    return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

@register_layer("Sigmoid")
def build_sigmoid(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Sigmoid()

@register_layer("Tanh")
def build_tanh(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Tanh()

@register_layer("MaxPool")
def build_maxpool(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    k = int(props.get("kernel_size", 2))
    stride = int(props.get("stride", k)) # Default stride is kernel_size
    padding = int(props.get("padding", 0))
    
    layer = nn.MaxPool2d(kernel_size=k, stride=stride, padding=padding)
    
    # H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    ctx.out_spatial_size = (ctx.out_spatial_size + 2 * padding - 1 * (k - 1) - 1) // stride + 1
    return layer

@register_layer("CustomLayer")
def build_custom_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    code = props.get("code", "")
    if not code:
        raise ValueError("CustomLayer requires 'code' property with Python code defining the layer.")
    
    # Execute the code in a controlled environment
    local_vars = {"nn": nn, "torch": torch}
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        layer = local_vars.get("layer")
        if not isinstance(layer, nn.Module):
            raise ValueError("CustomLayer code must define a variable 'layer' that is an nn.Module.")
        return layer
    except Exception as e:
        raise ValueError(f"Error executing CustomLayer code: {e}")

@register_layer("AvgPool")
def build_avgpool(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    k = int(props.get("kernel_size", 2))
    stride = int(props.get("stride", k))
    padding = int(props.get("padding", 0))
    
    layer = nn.AvgPool2d(kernel_size=k, stride=stride, padding=padding)
    
    ctx.out_spatial_size = (ctx.out_spatial_size + 2 * padding - 1 * (k - 1) - 1) // stride + 1
    return layer

@register_layer("AdaptiveAvgPool")
def build_adaptive_avgpool(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    output_size = int(props.get("output_size", 1))
    layer = nn.AdaptiveAvgPool2d((output_size, output_size))
    ctx.out_spatial_size = output_size
    return layer

@register_layer("BatchNorm2d")
def build_batchnorm2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.BatchNorm2d(ctx.out_channels)

@register_layer("BatchNorm1d")
def build_batchnorm1d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.BatchNorm1d(ctx.out_channels)

@register_layer("InstanceNorm2d")
def build_instancenorm2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.InstanceNorm2d(ctx.out_channels)

@register_layer("GroupNorm")
def build_groupnorm(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    num_groups = int(props.get("num_groups", 4))
    # GroupNorm requires num_channels to be divisible by num_groups
    return nn.GroupNorm(num_groups, ctx.out_channels)

@register_layer("LayerNorm")
def build_layernorm(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    if ctx.out_spatial_size > 1:
        normalized_shape = [ctx.out_channels, ctx.out_spatial_size, ctx.out_spatial_size]
    else:
        normalized_shape = [ctx.out_channels]
    return nn.LayerNorm(normalized_shape)

@register_layer("ELU")
def build_elu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    alpha = float(props.get("alpha", 1.0))
    return nn.ELU(alpha=alpha, inplace=True)

@register_layer("GELU")
def build_gelu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.GELU()

@register_layer("SiLU")
def build_silu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.SiLU(inplace=True)

@register_layer("Softmax")
def build_softmax(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    dim = int(props.get("dim", 1))
    return nn.Softmax(dim=dim)

@register_layer("LogSoftmax")
def build_logsoftmax(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    dim = int(props.get("dim", 1))
    return nn.LogSoftmax(dim=dim)

@register_layer("Dropout")
def build_dropout(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    p = float(props.get("p", 0.5))
    return nn.Dropout(p=p)

@register_layer("Dropout2d")
def build_dropout2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    p = float(props.get("p", 0.5))
    return nn.Dropout2d(p=p)

@register_layer("AlphaDropout")
def build_alphadropout(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    p = float(props.get("p", 0.5))
    return nn.AlphaDropout(p=p)

@register_layer("Flatten")
def build_flatten(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Flatten()

@register_layer("Identity")
def build_identity(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Identity()

@register_layer("Dense")
def build_dense(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    # Auto-flatten if needed (heuristic: if spatial size > 1 or channels > 1, assume we need to flatten first? 
    # Actually, nn.Linear expects (N, *, H_in). If we come from Conv2d (N, C, H, W), we need Flatten.
    # But here we return a single module. If flatten is needed, it should be a separate layer or we use Sequential internally.
    # To keep it simple, we assume user adds Flatten, OR we check context.
    # Let's be smart: if we are "conceptually" 2D, we add Flatten implicitly if this is the first Dense layer.
    # However, to support "Dense -> Dense", we shouldn't flatten again.
    # A robust way: check if we are coming from a Conv layer state? 
    # For now, let's just return Linear. The user or the graph parser should handle Flatten.
    # BUT, to make it easy for users, we can return a Sequential(Flatten, Linear) if we detect we are in 2D mode.
    
    layers = []
    # Heuristic: if spatial size is valid (from convs), we might need to flatten.
    # But ctx.out_channels tracks features.
    # Let's assume ctx.out_channels * ctx.out_spatial_size^2 is the input features.
    
    # Correctly calculate input features even for non-square spatial dimensions
    # We need to get the actual H and W from the input shape.
    # Let's assume the shape is (C, H, W) and is stored in ctx.input_shapes[0]
    in_feat = ctx.out_channels
    if ctx.out_spatial_size > 0 and ctx.input_shapes:
        s = ctx.input_shapes[0]
        if len(s) == 3: # C, H, W
            in_feat = s[0] * s[1] * s[2]
        else: # Fallback for older logic
            in_feat = ctx.out_channels * ctx.out_spatial_size * ctx.out_spatial_size

    out_feat = int(props.get("out_features", ctx.num_classes))
    
    # If we have spatial dims, we likely need to flatten.
    if ctx.out_spatial_size > 0:
        layers.append(nn.Flatten())
        ctx.out_spatial_size = 0 # Flattened
        ctx.out_channels = in_feat
    else:
        # If already flat, in_feat is just ctx.out_channels
        in_feat = ctx.out_channels
    
    layers.append(nn.Linear(ctx.out_channels, out_feat))
    ctx.out_channels = out_feat
    
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)

@register_layer("Upsample")
def build_upsample(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    scale_factor = float(props.get("scale_factor", 2.0))
    mode = props.get("mode", "nearest")
    layer = nn.Upsample(scale_factor=scale_factor, mode=mode)
    ctx.out_spatial_size = int(ctx.out_spatial_size * scale_factor)
    return layer

@register_layer("PixelShuffle")
def build_pixel_shuffle(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    upscale_factor = int(props.get("upscale_factor", 2))
    layer = nn.PixelShuffle(upscale_factor)
    ctx.out_channels = ctx.out_channels // (upscale_factor ** 2)
    ctx.out_spatial_size = ctx.out_spatial_size * upscale_factor
    return layer

@register_layer("ZeroPad2d")
def build_zeropad2d(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    padding = int(props.get("padding", 1))
    layer = nn.ZeroPad2d(padding)
    ctx.out_spatial_size = ctx.out_spatial_size + 2 * padding
    return layer

@register_layer("PReLU")
def build_prelu(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.PReLU()

@register_layer("Softplus")
def build_softplus(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Softplus()

@register_layer("Hardswish")
def build_hardswish(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Hardswish(inplace=True)

@register_layer("Hardsigmoid")
def build_hardsigmoid(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return nn.Hardsigmoid(inplace=True)

@register_layer("graph/subgraph")
def build_subgraph_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    subgraph_data = props.get("subgraph", {})
    
    # Support loading subgraph from file
    if not subgraph_data and "subgraph_path" in props:
        import json
        import os
        path = props["subgraph_path"]
        if path in SUBGRAPH_CACHE:
            subgraph_data = SUBGRAPH_CACHE[path]
        elif os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    subgraph_data = json.load(f)
                SUBGRAPH_CACHE[path] = subgraph_data
            except Exception as e:
                raise ValueError(f"Error loading subgraph from {path}: {e}")
    
    if not subgraph_data:
        return nn.Identity()
    
    # Parse subgraph DAG
    dag = parse_subgraph_dag(subgraph_data)
    
    # Build subgraph context
    sub_ctx = LayerContext(ctx.in_channels, ctx.spatial_size, ctx.num_classes)
    
    # Map external inputs to internal input nodes
    for i, nid in enumerate(dag['input_ids']):
        if i < len(ctx.input_shapes):
            sub_ctx.shapes[nid] = ctx.input_shapes[i]
        else:
            # Fallback to default input shape if available, or current context
            sub_ctx.shapes[nid] = (ctx.in_channels, ctx.spatial_size, ctx.spatial_size)

    layers_map = {}
    
    for nid in dag['execution_order']:
        node_def = dag['nodes'][nid]
        t = node_def.get("type")
        
        # Propagate shapes for Input/Output nodes
        if t == "graph/input":
            continue
        if t == "graph/output":
            srcs = dag['connections'].get(nid, [])
            if srcs and srcs[0] in sub_ctx.shapes:
                sub_ctx.shapes[nid] = sub_ctx.shapes[srcs[0]]
            # Add Identity layer for output node so it captures the input
            layers_map[nid] = nn.Identity()
            continue
            
        # Prepare context for this node
        srcs = dag['connections'].get(nid, [])
        node_input_shapes = []
        for s in srcs:
            if s in sub_ctx.shapes:
                node_input_shapes.append(sub_ctx.shapes[s])
        
        sub_ctx.input_shapes = node_input_shapes
        
        # Set standard context properties based on first input
        if node_input_shapes:
            s0 = node_input_shapes[0]
            if len(s0) == 3:
                sub_ctx.out_channels = s0[0]
                sub_ctx.out_spatial_size = s0[1]
            else:
                sub_ctx.out_channels = s0[0]
                sub_ctx.out_spatial_size = 0
        
        if t in LAYER_REGISTRY:
            builder = LAYER_REGISTRY[t]
            node_props = node_def.get("properties", {})
            try:
                mod = builder(node_props, sub_ctx)
                layers_map[nid] = mod
                
                if sub_ctx.out_spatial_size > 0:
                    sub_ctx.shapes[nid] = (sub_ctx.out_channels, sub_ctx.out_spatial_size, sub_ctx.out_spatial_size)
                else:
                    sub_ctx.shapes[nid] = (sub_ctx.out_channels,)
            except Exception as e:
                print(f"Error building subgraph layer {t}: {e}")

    # Update parent context with subgraph output shapes
    output_shapes = []
    for nid in dag['output_ids']:
        if nid in sub_ctx.shapes:
            output_shapes.append(sub_ctx.shapes[nid])
            
    if output_shapes:
        s0 = output_shapes[0]
        if len(s0) == 3:
            ctx.out_channels = s0[0]
            ctx.out_spatial_size = s0[1]
        else:
            ctx.out_channels = s0[0]
            ctx.out_spatial_size = 0

    return SubgraphModule(layers_map, dag['execution_order'], dag['connections'], dag['input_ids'], dag['output_ids'])

@register_layer("graph/subgraph_ref")
def build_subgraph_ref_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return build_subgraph_layer(props, ctx)

# Alias: embedded subgraph nodes created in the front-end should be treated the same as graph/subgraph
@register_layer("graph/embedded_subgraph")
def build_embedded_subgraph_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return build_subgraph_layer(props, ctx)

@register_layer("Add")
def build_add(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return MergeAdd()

@register_layer("Multiply")
def build_multiply(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    return MergeMultiply()

@register_layer("Concat")
def build_concat(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    dim = int(props.get("dim", 1))
    return MergeConcat(dim=dim)

@register_layer("Resize")
def build_resize(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    size_val = props.get("size", 28)
    mode = props.get("mode", "bilinear")
    align_corners = bool(props.get("align_corners", False))
    
    # Handle size parsing
    if isinstance(size_val, str):
        try:
            # Safely parse tuple or int
            if ',' in size_val:
                size_val = tuple(map(int, size_val.strip('()[]').split(',')))
            else:
                size_val = int(size_val)
        except ValueError:
            raise ValueError(f"Invalid size format for Resize layer: {size_val}")

    if isinstance(size_val, (int, float)):
        size = (int(size_val), int(size_val))
        ctx.out_spatial_size = int(size_val)
    elif isinstance(size_val, (list, tuple)):
        size = tuple(map(int, size_val))
        ctx.out_spatial_size = size[0]
    else:
        size = (28, 28)
        ctx.out_spatial_size = 28

    class ResizeLayer(nn.Module):
        def __init__(self, size, mode, align_corners):
            super().__init__()
            self.size = size
            self.mode = mode
            self.align_corners = align_corners if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
            
        def forward(self, x):
            return torch.nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
            
    return ResizeLayer(size, mode, align_corners)

# --- Transformer & NLP Layers ---

@register_layer("Embedding")
def build_embedding(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    num_embeddings = int(props.get("num_embeddings", ctx.num_classes))
    # Ensure num_embeddings is at least num_classes (for language modeling)
    num_embeddings = max(num_embeddings, ctx.num_classes)
    embedding_dim = int(props.get("embedding_dim", 128))
    
    layer = nn.Embedding(num_embeddings, embedding_dim)
    
    # Update context: output is now (seq_len, embedding_dim)
    # ctx.out_channels represents embedding_dim
    # ctx.out_spatial_size could represent seq_len, but we keep it for compatibility
    ctx.out_channels = embedding_dim
    ctx.out_spatial_size = 0  # Not spatial anymore
    
    return layer

@register_layer("PositionalEncoding")
def build_positional_encoding(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    max_len = int(props.get("max_len", 512))
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    dropout = float(props.get("dropout", 0.1))
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            # x: (batch, seq_len, d_model) or (batch, seq_len) for embeddings
            if len(x.shape) == 2:
                # If input is (batch, seq_len), assume it's token indices
                # This shouldn't happen after Embedding, but handle it
                return x
            
            # x: (batch, seq_len, d_model)
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)
    
    ctx.out_channels = d_model
    return PositionalEncoding(d_model, max_len, dropout)

@register_layer("TransformerEncoderLayer")
def build_transformer_encoder_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    nhead = int(props.get("nhead", 8))
    dim_feedforward = int(props.get("dim_feedforward", 512))
    dropout = float(props.get("dropout", 0.1))
    
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    
    ctx.out_channels = d_model
    return layer

@register_layer("TransformerEncoder")
def build_transformer_encoder(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    nhead = int(props.get("nhead", 8))
    num_layers = int(props.get("num_layers", 6))
    dim_feedforward = int(props.get("dim_feedforward", 512))
    dropout = float(props.get("dropout", 0.1))
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    
    layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    ctx.out_channels = d_model
    return layer

@register_layer("TransformerDecoderLayer")
def build_transformer_decoder_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    nhead = int(props.get("nhead", 8))
    dim_feedforward = int(props.get("dim_feedforward", 512))
    dropout = float(props.get("dropout", 0.1))
    
    layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    
    ctx.out_channels = d_model
    return layer

@register_layer("TransformerDecoder")
def build_transformer_decoder(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    nhead = int(props.get("nhead", 8))
    num_layers = int(props.get("num_layers", 6))
    dim_feedforward = int(props.get("dim_feedforward", 512))
    dropout = float(props.get("dropout", 0.1))
    
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    
    layer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    ctx.out_channels = d_model
    return layer

@register_layer("MultiheadAttention")
def build_multihead_attention(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    embed_dim = ctx.out_channels if ctx.out_channels > 0 else int(props.get("embed_dim", 128))
    num_heads = int(props.get("num_heads", 8))
    dropout = float(props.get("dropout", 0.0))
    
    class MultiheadAttentionWrapper(nn.Module):
        def __init__(self, embed_dim, num_heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            
        def forward(self, x):
            # Self-attention: query = key = value = x
            attn_output, _ = self.attn(x, x, x)
            return attn_output
    
    ctx.out_channels = embed_dim
    return MultiheadAttentionWrapper(embed_dim, num_heads, dropout)

@register_layer("Linear")
def build_linear(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    in_features = ctx.out_channels if ctx.out_channels > 0 else int(props.get("in_features", 128))
    out_features = int(props.get("out_features", ctx.num_classes))
    # Ensure out_features is at least num_classes (for classification)
    out_features = max(out_features, ctx.num_classes)
    
    layer = nn.Linear(in_features, out_features)
    
    ctx.out_channels = out_features
    return layer

@register_layer("GPTBlock")
def build_gpt_block(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    d_model = ctx.out_channels if ctx.out_channels > 0 else int(props.get("d_model", 128))
    nhead = int(props.get("nhead", 8))
    dim_feedforward = int(props.get("dim_feedforward", 512))
    dropout = float(props.get("dropout", 0.1))
    
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
            # Pre-norm architecture (like GPT-2)
            # Self-attention with causal mask
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            
            # Attention block
            x_norm = self.ln1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
            x = x + attn_out
            
            # FFN block
            x = x + self.ffn(self.ln2(x))
            
            return x
    
    ctx.out_channels = d_model
    return GPTBlock(d_model, nhead, dim_feedforward, dropout)
    

# --- Graph Parsing (Topological) ---

def parse_graph_to_plan(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析 LiteGraph JSON，通过追踪连接构建有序的 Layer 列表。
    """
    if not graph or "nodes" not in graph:
        raise ValueError("Graph is empty or invalid")

    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {l[0]: l for l in graph.get("links", [])}
    
    # Find Data Node
    data_node = None
    for n in graph["nodes"]:
        if n.get("type") in ("MNIST", "Fashion-MNIST", "CIFAR-10", "WikiText-2", "WikiText-103", "PennTreebank", "CustomData"):
            data_node = n
            break
    
    if not data_node:
        raise ValueError("No Data node found in the graph")

    plan = {
        "data": {
            "dataset": data_node.get("type"),
            "batch_size": int(data_node.get("properties", {}).get("batch_size", 64))
        },
        "model": [],
        "train": {"epochs": 3, "optimizer": "Adam", "lr": 1e-3, "loss": "CrossEntropy"}
    }

    # Traverse graph to find Loss node and build model
    model_plan, train_config = _traverse_graph_with_loss(data_node, nodes, links)
    
    if not model_plan:
        raise ValueError("Model has no layers connected to Data node")
        
    if not train_config:
        raise ValueError("Model must end with a Training Goal (Loss) node")

    plan["model"] = model_plan
    plan["train"] = train_config

    return plan

def _traverse_graph_with_loss(start_node, nodes, links) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Build Adjacency List
    adj = {} # id -> [target_id]
    rev_adj = {} # id -> [source_id]
    
    # Initialize
    for nid in nodes:
        adj[nid] = []
        rev_adj[nid] = []
        
    for link in links.values():
        # link: [id, origin_id, origin_slot, target_id, target_slot, type]
        origin = link[1]
        target = link[3]
        if origin in nodes and target in nodes:
            adj[origin].append(target)
            rev_adj[target].append(origin)
            
    # Find all Loss nodes
    loss_nodes = []
    for nid, n in nodes.items():
        if n.get("type") == "Loss":
            loss_nodes.append(n)
            
    if not loss_nodes:
        return {}, []
        
    # Topological Sort (Kahn's Algorithm)
    # Only include nodes reachable from Data and leading to Loss
    
    # 1. Forward Reachability from Data
    reachable_from_data = set()
    queue = [start_node["id"]]
    reachable_from_data.add(start_node["id"])
    
    idx = 0
    while idx < len(queue):
        curr = queue[idx]; idx += 1
        for neighbor in adj[curr]:
            if neighbor not in reachable_from_data:
                reachable_from_data.add(neighbor)
                queue.append(neighbor)
                
    # 2. Backward Reachability from Loss
    reachable_to_loss = set()
    queue = [n["id"] for n in loss_nodes]
    for q in queue: reachable_to_loss.add(q)
    
    idx = 0
    while idx < len(queue):
        curr = queue[idx]; idx += 1
        for neighbor in rev_adj[curr]:
            if neighbor not in reachable_to_loss:
                reachable_to_loss.add(neighbor)
                queue.append(neighbor)
                
    # Intersection
    valid_nodes = reachable_from_data.intersection(reachable_to_loss)
    
    # Sort valid nodes
    in_degree = {nid: 0 for nid in valid_nodes}
    for nid in valid_nodes:
        for neighbor in adj[nid]:
            if neighbor in valid_nodes:
                in_degree[neighbor] += 1
                
    sorted_nodes = []
    zero_in = [nid for nid in valid_nodes if in_degree[nid] == 0]
    # Ensure Data node is first if it's in zero_in (it should be)
    
    while zero_in:
        curr = zero_in.pop(0)
        sorted_nodes.append(curr)
        
        for neighbor in adj[curr]:
            if neighbor in valid_nodes:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in.append(neighbor)
                    
    # Build Plan
    # execution_order: list of node IDs
    # nodes: dict of node props
    # connections: dict of node_id -> [source_ids]
    
    plan_nodes = {}
    connections = {}
    
    for nid in sorted_nodes:
        n = nodes[nid]
        ntype = n.get("type")
        props = n.get("properties", {}).copy()
        
        if ntype == "graph/subgraph":
             # Prefer embedded subgraph, then check for file path
             if "subgraph" in n:
                 props["subgraph"] = n["subgraph"]
             elif "subgraph" in props: # Legacy support or if user put it in props
                 props["subgraph"] = props["subgraph"]
             
             # If still empty, check if we have a path in properties (already in props)
             
        plan_nodes[str(nid)] = {
            "type": ntype,
            **props
        }
        
        # Inputs
        # We need to map inputs to source node IDs
        # Look at rev_adj, but we need to be specific about which input slot?
        # For now, just list all valid source nodes
        sources = [src for src in rev_adj[nid] if src in valid_nodes]
        # Special case: Data node has no sources in this graph context (it IS the source)
        # But in the plan, we might want to mark it?
        # Actually, Data node is just a node.
        
        # If a node is NOT the Data node, and has no sources in valid_nodes, it's weird.
        # But Data node is in sorted_nodes[0].
        
        # Map sources to "data" if it is the Data node?
        # We MUST map the Data Node ID to "data" because GraphModule initializes outputs={'data': x}
        # and the Data Node itself is skipped in execution (not a layer).
        
        remapped_sources = []
        for src in sources:
            if src == start_node["id"]:
                remapped_sources.append("data")
            else:
                remapped_sources.append(src)

        if nid == start_node["id"]:
            connections[str(nid)] = ["data"] # Special marker (unused by GraphModule but good for debug)
        else:
            connections[str(nid)] = remapped_sources

    # Train Configs
    train_configs = []
    for ln in loss_nodes:
        if ln["id"] in valid_nodes:
            props = ln.get("properties", {})
            train_configs.append({
                "node_id": ln["id"],
                "loss": props.get("kind", "CrossEntropy"),
                "optimizer": props.get("optimizer", "Adam"),
                "lr": float(props.get("lr", 0.001)),
                "epochs": int(props.get("epochs", 10)),
                "target": props.get("target", "Label"),
                "weight": float(props.get("weight", 1.0))
            })

    model_plan = {
        "nodes": plan_nodes,
        "execution_order": sorted_nodes,
        "connections": connections
    }
    
    return model_plan, train_configs

def build_model_from_plan(plan: Dict[str, Any], in_channels: int, num_classes: int, initial_spatial_size: int = 28) -> nn.Module:
    ctx = LayerContext(in_channels, initial_spatial_size, num_classes)
    
    model_struct = plan.get("model", {})
    nodes_def = model_struct.get("nodes", {})
    execution_order = model_struct.get("execution_order", [])
    connections = model_struct.get("connections", {})
    
    layers_map = {}
    
    # Pre-populate Data node shape
    # We assume the first node in execution order is Data node (or we find it)
    # Actually, connections has "data" marker.
    
    # We need to simulate shape propagation
    # ctx.shapes stores output shape of each node
    ctx.shapes["data"] = (in_channels, initial_spatial_size, initial_spatial_size) if initial_spatial_size > 0 else (in_channels,)
    
    for nid in execution_order:
        nid_str = str(nid)
        node_def = nodes_def.get(nid_str)
        if not node_def: continue
        
        t = node_def.get("type")
        
        # Skip Data node and Loss node in layer building
        if t in ("MNIST", "Fashion-MNIST", "CIFAR-10", "CustomData", "Loss"):
            # Just propagate shape if needed?
            # Data node output is already set.
            # Loss node has no output.
            if t in ("MNIST", "Fashion-MNIST", "CIFAR-10", "CustomData"):
                ctx.shapes[nid_str] = ctx.shapes["data"]
            continue
            
        # Get input shape
        # Assume single input for now
        sources = connections.get(nid_str, [])
        if not sources: 
            # print(f"DEBUG: No sources for node {nid}")
            continue
        
        # Collect all input shapes
        current_input_shapes = []
        for src_id in sources:
            src_id = str(src_id)
            if src_id == "data":
                current_input_shapes.append(ctx.shapes["data"])
            else:
                shape = ctx.shapes.get(src_id)
                if shape:
                    current_input_shapes.append(shape)
        
        ctx.input_shapes = current_input_shapes

        src_id = str(sources[0])
        if src_id == "data":
            input_shape = ctx.shapes["data"]
        else:
            input_shape = ctx.shapes.get(src_id)
            
        if not input_shape:
            print(f"Warning: Unknown input shape for node {nid}")
            continue
            
        # Update Context for this layer
        if len(input_shape) == 3:
            ctx.out_channels = input_shape[0]
            ctx.out_spatial_size = input_shape[1]
        else:
            ctx.out_channels = input_shape[0]
            ctx.out_spatial_size = 0
            
        # Build Layer
        if t in LAYER_REGISTRY:
            builder = LAYER_REGISTRY[t]
            props = {k: v for k, v in node_def.items() if k != "type"}
            try:
                mod = builder(props, ctx)
                layers_map[nid_str] = mod
                
                # Record output shape
                if ctx.out_spatial_size > 0:
                    ctx.shapes[nid_str] = (ctx.out_channels, ctx.out_spatial_size, ctx.out_spatial_size)
                else:
                    ctx.shapes[nid_str] = (ctx.out_channels,)
            except Exception as e:
                print(f"Error building layer {t}: {e}")
        
    return GraphModule(layers_map, execution_order, connections)

def build_optim_loss_from_plan(plan: Dict[str, Any], model: nn.Module):
    # Use the first config for Optimizer
    train_configs = plan.get("train", [])
    if not train_configs:
        # Fallback
        return torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss()
        
    main_config = train_configs[0]
    
    optim_kind = main_config.get("optimizer", "Adam")
    lr = main_config.get("lr", 1e-3)
    
    if optim_kind == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_kind == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    # Loss functions map
    criterions = {}
    for cfg in train_configs:
        kind = cfg.get("loss", "CrossEntropy")
        if kind == "MSE":
            crit = nn.MSELoss()
        elif kind == "SmoothL1":
            crit = nn.SmoothL1Loss()
        else:
            crit = nn.CrossEntropyLoss()
        criterions[cfg["node_id"]] = crit
        
    return optimizer, criterions

def parse_subgraph_dag(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = {str(n["id"]): n for n in subgraph.get("nodes", [])}
    links = {l[0]: l for l in subgraph.get("links", [])}
    
    input_nodes = []
    output_nodes = []
    
    for nid, n in nodes.items():
        if n.get("type") == "graph/input":
            input_nodes.append(n)
        elif n.get("type") == "graph/output":
            output_nodes.append(n)
            
    # Sort by Y position then X
    def get_pos(n):
        p = n.get("pos", [0, 0])
        return (p[1], p[0])
        
    input_nodes.sort(key=get_pos)
    output_nodes.sort(key=get_pos)
    
    input_ids = [str(n["id"]) for n in input_nodes]
    output_ids = [str(n["id"]) for n in output_nodes]
    
    # Build adjacency
    adj = {nid: [] for nid in nodes}
    rev_adj = {nid: [] for nid in nodes}
    
    # Store inputs with slot index to ensure correct order
    node_inputs = {nid: [] for nid in nodes}

    for l in links.values():
        origin_id = str(l[1])
        target_id = str(l[3])
        target_slot = int(l[4])
        
        if origin_id in nodes and target_id in nodes:
            adj[origin_id].append(target_id)
            node_inputs[target_id].append((origin_id, target_slot))
            
    # Sort inputs by slot index and populate rev_adj
    for nid in nodes:
        inputs = node_inputs[nid]
        inputs.sort(key=lambda x: x[1])
        rev_adj[nid] = [x[0] for x in inputs]
            
    # Topological sort
    in_degree = {nid: len(rev_adj[nid]) for nid in nodes}
    queue = [nid for nid in nodes if in_degree[nid] == 0]
    sorted_nodes = []
    
    while queue:
        u = queue.pop(0)
        sorted_nodes.append(u)
        
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycle detected (len(sorted_nodes) < len(nodes)), we might miss nodes.
    # But for now assume DAG.
    if len(sorted_nodes) < len(nodes):
        # Add remaining nodes?
        remaining = [n for n in nodes if n not in sorted_nodes]
        sorted_nodes.extend(remaining)

    connections = {}
    for nid in nodes:
        connections[nid] = rev_adj[nid]
        
    return {
        "nodes": nodes,
        "execution_order": sorted_nodes,
        "connections": connections,
        "input_ids": input_ids,
        "output_ids": output_ids
    }

def _traverse_graph(start_node, nodes, links) -> List[Dict[str, Any]]:
    # Deprecated but kept for subgraph parsing if needed, 
    # but subgraph parsing should also be DAG-aware ideally.
    # For now, let's assume subgraphs are still linear or simple.
    # Actually, we should update parse_subgraph to use the new logic too?
    # But Subgraph layer expects a Sequential or similar.
    # If we want Subgraph to be a GraphModule, we need to update build_subgraph_layer.
    
    # Let's keep _traverse_graph simple for now or reuse the new logic.
    # But _traverse_graph returns a list.
    # Let's leave it as is for legacy/subgraph support for now.
    model_plan = []
    current_node = start_node
    visited = set()
    
    while current_node:
        nid = current_node["id"]
        if nid in visited: break
        visited.add(nid)
        
        ntype = current_node.get("type")
        props = current_node.get("properties", {}).copy()

        if ntype == "graph/subgraph":
             if "subgraph" in current_node:
                 props["subgraph"] = current_node["subgraph"]
             elif "subgraph" in props:
                 props["subgraph"] = props["subgraph"]

        # Add to Model Plan if it's a registered layer
        if ntype in LAYER_REGISTRY:
            model_plan.append({
                "type": ntype,
                **props
            })
        
        # Find next node
        outputs = current_node.get("outputs", [])
        next_node = None
        if outputs:
            for out in outputs:
                if out.get("links"):
                    for link_id in out["links"]:
                        link = links.get(link_id)
                        if link:
                            target_id = link[3]
                            if target_id in nodes:
                                next_node = nodes[target_id]
                                break 
                if next_node: break
        
        current_node = next_node
    return model_plan

def parse_subgraph(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = {n["id"]: n for n in subgraph.get("nodes", [])}
    links = {l[0]: l for l in subgraph.get("links", [])}
    
    input_node = None
    for n in subgraph.get("nodes", []):
        if n.get("type") == "graph/input":
            input_node = n
            break
            
    if not input_node:
        return {"model": []}
        
    return {"model": _traverse_graph(input_node, nodes, links)}

def _build_layers(plan: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    layers = []
    for item in plan.get("model", []):
        t = item.get("type")
        if t in LAYER_REGISTRY:
            builder = LAYER_REGISTRY[t]
            props = {k: v for k, v in item.items() if k != "type"}
            try:
                mod = builder(props, ctx)
                layers.append(mod)
            except Exception as e:
                print(f"Error building layer {t}: {e}")
        else:
            print(f"Warning: Unknown layer type {t}")
    
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)
