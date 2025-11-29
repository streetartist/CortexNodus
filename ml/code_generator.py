"""
PyTorch Code Generator (Refactored)
Generates standalone PyTorch training scripts from graph definitions
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Callable
import json
import hashlib

# Import subgraph parser from designer
try:
    from ml.designer import parse_subgraph_dag
except ImportError:
    # Fallback or mock if designer is not available in path
    def parse_subgraph_dag(subgraph):
        return {}

# --- Layer Generation Infrastructure ---

class LayerGenContext:
    def __init__(self, in_channels: int, spatial_size: int, num_classes: int, layer_name: str, input_vars: List[str], 
                 subgraph_registry: List, subgraph_cache: Dict, input_shapes: List[Tuple]):
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.layer_name = layer_name
        self.input_vars = input_vars
        self.subgraph_registry = subgraph_registry
        self.subgraph_cache = subgraph_cache
        self.input_shapes = input_shapes # Full shapes of all inputs
        
        self.helpers = set()
        self.out_channels = in_channels
        self.out_spatial_size = spatial_size

LayerGenerator = Callable[[Dict[str, Any], LayerGenContext], Tuple[str, str]]
LAYER_GENERATORS: Dict[str, LayerGenerator] = {}

def register_layer_gen(name: str):
    def decorator(func: LayerGenerator):
        LAYER_GENERATORS[name] = func
        return func
    return decorator

# --- Layer Generators ---

@register_layer_gen("Conv2D")
def gen_conv2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    out_channels = int(node.get("out_channels", 32))
    kernel_size = int(node.get("kernel_size", 3))
    stride = int(node.get("stride", 1))
    padding = int(node.get("padding", kernel_size // 2))
    
    definition = f"nn.Conv2d({ctx.in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    
    ctx.out_channels = out_channels
    ctx.out_spatial_size = (ctx.spatial_size + 2 * padding - kernel_size) // stride + 1
    
    return definition, forward_expr

@register_layer_gen("ConvTranspose2d")
def gen_conv_transpose2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    out_channels = int(node.get("out_channels", 32))
    kernel_size = int(node.get("kernel_size", 3))
    stride = int(node.get("stride", 1))
    padding = int(node.get("padding", 0))
    output_padding = int(node.get("output_padding", 0))
    
    definition = f"nn.ConvTranspose2d({ctx.in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    
    ctx.out_channels = out_channels
    ctx.out_spatial_size = (ctx.spatial_size - 1) * stride - 2 * padding + kernel_size + output_padding
    
    return definition, forward_expr

@register_layer_gen("ReLU")
def gen_relu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.relu({ctx.input_vars[0]})"

@register_layer_gen("LeakyReLU")
def gen_leaky_relu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    negative_slope = float(node.get("negative_slope", 0.01))
    return "", f"F.leaky_relu({ctx.input_vars[0]}, negative_slope={negative_slope})"

@register_layer_gen("PReLU")
def gen_prelu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return f"nn.PReLU()", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("ELU")
def gen_elu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    alpha = float(node.get("alpha", 1.0))
    return "", f"F.elu({ctx.input_vars[0]}, alpha={alpha})"

@register_layer_gen("GELU")
def gen_gelu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.gelu({ctx.input_vars[0]})"

@register_layer_gen("SiLU")
def gen_silu(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.silu({ctx.input_vars[0]})"

@register_layer_gen("Sigmoid")
def gen_sigmoid(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"torch.sigmoid({ctx.input_vars[0]})"

@register_layer_gen("Tanh")
def gen_tanh(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"torch.tanh({ctx.input_vars[0]})"

@register_layer_gen("Softplus")
def gen_softplus(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.softplus({ctx.input_vars[0]})"

@register_layer_gen("Hardswish")
def gen_hardswish(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.hardswish({ctx.input_vars[0]})"

@register_layer_gen("Hardsigmoid")
def gen_hardsigmoid(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", f"F.hardsigmoid({ctx.input_vars[0]})"

@register_layer_gen("MaxPool")
def gen_maxpool(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    kernel_size = int(node.get("kernel_size", 2))
    stride = int(node.get("stride", kernel_size))
    padding = int(node.get("padding", 0))
    
    forward_expr = f"F.max_pool2d({ctx.input_vars[0]}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
    ctx.out_spatial_size = (ctx.spatial_size + 2 * padding - kernel_size) // stride + 1
    return "", forward_expr

@register_layer_gen("AvgPool")
def gen_avgpool(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    kernel_size = int(node.get("kernel_size", 2))
    stride = int(node.get("stride", kernel_size))
    padding = int(node.get("padding", 0))
    
    forward_expr = f"F.avg_pool2d({ctx.input_vars[0]}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
    ctx.out_spatial_size = (ctx.spatial_size + 2 * padding - kernel_size) // stride + 1
    return "", forward_expr

@register_layer_gen("AdaptiveAvgPool")
def gen_adaptive_avgpool(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    output_size = int(node.get("output_size", 1))
    forward_expr = f"F.adaptive_avg_pool2d({ctx.input_vars[0]}, ({output_size}, {output_size}))"
    ctx.out_spatial_size = output_size
    return "", forward_expr

@register_layer_gen("BatchNorm2d")
def gen_batchnorm2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return f"nn.BatchNorm2d({ctx.in_channels})", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("BatchNorm1d")
def gen_batchnorm1d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return f"nn.BatchNorm1d({ctx.in_channels})", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("InstanceNorm2d")
def gen_instancenorm2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return f"nn.InstanceNorm2d({ctx.in_channels})", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("GroupNorm")
def gen_groupnorm(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    num_groups = int(node.get("num_groups", 4))
    return f"nn.GroupNorm({num_groups}, {ctx.in_channels})", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("LayerNorm")
def gen_layernorm(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    # Check input shape to determine normalized_shape
    if ctx.spatial_size > 0:
        norm_shape = f"[{ctx.in_channels}, {ctx.spatial_size}, {ctx.spatial_size}]"
    else:
        norm_shape = f"[{ctx.in_channels}]"
    return f"nn.LayerNorm({norm_shape})", f"self.{ctx.layer_name}({ctx.input_vars[0]})"

@register_layer_gen("Dropout")
def gen_dropout(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    p = float(node.get("p", 0.5))
    return "", f"F.dropout({ctx.input_vars[0]}, p={p}, training=self.training)"

@register_layer_gen("Dropout2d")
def gen_dropout2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    p = float(node.get("p", 0.5))
    return "", f"F.dropout2d({ctx.input_vars[0]}, p={p}, training=self.training)"

@register_layer_gen("AlphaDropout")
def gen_alphadropout(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    p = float(node.get("p", 0.5))
    return "", f"F.alpha_dropout({ctx.input_vars[0]}, p={p}, training=self.training)"

@register_layer_gen("Flatten")
def gen_flatten(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    forward_expr = f"{ctx.input_vars[0]}.view({ctx.input_vars[0]}.size(0), -1)"
    flat_size = ctx.in_channels
    if ctx.spatial_size > 0:
        flat_size = ctx.in_channels * ctx.spatial_size * ctx.spatial_size
    ctx.out_channels = flat_size
    ctx.out_spatial_size = 0
    return "", forward_expr

@register_layer_gen("Dense")
@register_layer_gen("Linear")
def gen_dense(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    out_features = int(node.get("out_features", ctx.num_classes))
    auto_flatten = bool(node.get("auto_flatten", True))
    needs_flatten = ctx.spatial_size > 0 and auto_flatten
    
    if needs_flatten:
        in_features = ctx.in_channels * ctx.spatial_size * ctx.spatial_size
        forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]}.view({ctx.input_vars[0]}.size(0), -1))"
    else:
        in_features = ctx.in_channels
        forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
        
    definition = f"nn.Linear({in_features}, {out_features})"
    ctx.out_channels = out_features
    ctx.out_spatial_size = 0
    return definition, forward_expr

@register_layer_gen("Softmax")
def gen_softmax(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    dim = int(node.get("dim", 1))
    return "", f"F.softmax({ctx.input_vars[0]}, dim={dim})"

@register_layer_gen("LogSoftmax")
def gen_logsoftmax(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    dim = int(node.get("dim", 1))
    return "", f"F.log_softmax({ctx.input_vars[0]}, dim={dim})"

@register_layer_gen("Add")
def gen_add(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    if len(ctx.input_vars) > 1:
        summands = " + ".join(ctx.input_vars)
        return "", summands
    return "", ctx.input_vars[0]

@register_layer_gen("Multiply")
def gen_multiply(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    if len(ctx.input_vars) > 1:
        factors = " * ".join(ctx.input_vars)
        return "", factors
    return "", ctx.input_vars[0]

@register_layer_gen("Concat")
def gen_concat(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    dim = int(node.get("dim", 1))
    inputs_str = ", ".join(ctx.input_vars)
    forward_expr = f"torch.cat([{inputs_str}], dim={dim})"
    
    if dim == 1:
        total_channels = sum(s[0] for s in ctx.input_shapes)
        ctx.out_channels = total_channels
        # Spatial size remains same (assuming inputs match)
    return "", forward_expr

@register_layer_gen("Embedding")
def gen_embedding(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    num_embeddings = int(node.get("num_embeddings", 1000))
    embedding_dim = int(node.get("embedding_dim", 128))
    definition = f"nn.Embedding({num_embeddings}, {embedding_dim})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]}.long())"
    ctx.out_channels = embedding_dim
    ctx.out_spatial_size = 0 # Sequence length is not tracked as spatial size
    return definition, forward_expr

@register_layer_gen("PositionalEncoding")
def gen_positional_encoding(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    max_len = int(node.get("max_len", 512))
    d_model = ctx.in_channels
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"self._create_positional_encoding({d_model}, {max_len}, {dropout})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    ctx.helpers.add("PositionalEncoding")
    return definition, forward_expr

@register_layer_gen("GPTBlock")
def gen_gpt_block(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    d_model = ctx.in_channels
    nhead = int(node.get("nhead", 8))
    dim_feedforward = int(node.get("dim_feedforward", 512))
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"self._create_gpt_block({d_model}, {nhead}, {dim_feedforward}, {dropout})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    ctx.helpers.add("GPTBlock")
    return definition, forward_expr

@register_layer_gen("MultiheadAttention")
def gen_multihead_attention(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    embed_dim = int(node.get("embed_dim", 128))
    num_heads = int(node.get("num_heads", 8))
    dropout = float(node.get("dropout", 0.0))
    
    definition = f"nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, batch_first=True)"
    if len(ctx.input_vars) == 1:
        q = k = v = ctx.input_vars[0]
    elif len(ctx.input_vars) == 2:
        q = ctx.input_vars[0]
        k = v = ctx.input_vars[1]
    else:
        q = ctx.input_vars[0]
        k = ctx.input_vars[1]
        v = ctx.input_vars[2]
        
    forward_expr = f"self.{ctx.layer_name}({q}, {k}, {v})[0]"
    ctx.out_channels = embed_dim
    return definition, forward_expr

@register_layer_gen("TransformerEncoderLayer")
def gen_transformer_encoder_layer(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    d_model = int(node.get("d_model", 128))
    nhead = int(node.get("nhead", 8))
    dim_feedforward = int(node.get("dim_feedforward", 512))
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True)"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    return definition, forward_expr

@register_layer_gen("TransformerEncoder")
def gen_transformer_encoder(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    d_model = int(node.get("d_model", 128))
    nhead = int(node.get("nhead", 8))
    num_layers = int(node.get("num_layers", 6))
    dim_feedforward = int(node.get("dim_feedforward", 512))
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True), num_layers={num_layers})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    return definition, forward_expr

@register_layer_gen("TransformerDecoderLayer")
def gen_transformer_decoder_layer(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    d_model = int(node.get("d_model", 128))
    nhead = int(node.get("nhead", 8))
    dim_feedforward = int(node.get("dim_feedforward", 512))
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True)"
    if len(ctx.input_vars) > 1:
        tgt = ctx.input_vars[0]
        memory = ctx.input_vars[1]
        forward_expr = f"self.{ctx.layer_name}({tgt}, {memory})"
    else:
        forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]}, {ctx.input_vars[0]})"
    return definition, forward_expr

@register_layer_gen("TransformerDecoder")
def gen_transformer_decoder(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    d_model = int(node.get("d_model", 128))
    nhead = int(node.get("nhead", 8))
    num_layers = int(node.get("num_layers", 6))
    dim_feedforward = int(node.get("dim_feedforward", 512))
    dropout = float(node.get("dropout", 0.1))
    
    definition = f"nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True), num_layers={num_layers})"
    if len(ctx.input_vars) > 1:
        tgt = ctx.input_vars[0]
        memory = ctx.input_vars[1]
        forward_expr = f"self.{ctx.layer_name}({tgt}, {memory})"
    else:
        forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]}, {ctx.input_vars[0]})"
    return definition, forward_expr

@register_layer_gen("Identity")
def gen_identity(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    return "", ctx.input_vars[0]

@register_layer_gen("Resize")
def gen_resize(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    if "size" not in node:
        raise ValueError("Resize layer requires 'size' property.")
        
    size_val = node["size"]
    mode = node.get("mode", "bilinear")
    align_corners = node.get("align_corners", False)
    
    # Handle string input "h,w" or "[h,w]"
    if isinstance(size_val, str):
        size_val = size_val.strip()
        if size_val.startswith("[") and size_val.endswith("]"):
            import ast
            try:
                size_val = ast.literal_eval(size_val)
            except:
                pass
        elif "," in size_val:
            try:
                size_val = [int(x.strip()) for x in size_val.split(",")]
            except:
                pass
    
    if isinstance(size_val, (list, tuple)):
        size = tuple(map(int, size_val))
        ctx.out_spatial_size = size[0]
    else:
        size = (int(size_val), int(size_val))
        ctx.out_spatial_size = int(size_val)
        
    if mode in ['bilinear', 'bicubic']:
        forward_expr = f"F.interpolate({ctx.input_vars[0]}, size={size}, mode='{mode}', align_corners={align_corners})"
    else:
        forward_expr = f"F.interpolate({ctx.input_vars[0]}, size={size}, mode='{mode}')"
        
    return "", forward_expr

@register_layer_gen("Upsample")
def gen_upsample(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    scale_factor = float(node.get("scale_factor", 2.0))
    mode = node.get("mode", "nearest")
    align_corners = None
    if mode in ['bilinear', 'bicubic']:
        align_corners = False
    
    align_str = f", align_corners={align_corners}" if align_corners is not None else ""
    forward_expr = f"F.interpolate({ctx.input_vars[0]}, scale_factor={scale_factor}, mode='{mode}'{align_str})"
    
    ctx.out_spatial_size = int(ctx.spatial_size * scale_factor)
    return "", forward_expr

@register_layer_gen("PixelShuffle")
def gen_pixel_shuffle(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    upscale_factor = int(node.get("upscale_factor", 2))
    definition = f"nn.PixelShuffle({upscale_factor})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    
    ctx.out_channels = ctx.in_channels // (upscale_factor ** 2)
    ctx.out_spatial_size = ctx.spatial_size * upscale_factor
    return definition, forward_expr

@register_layer_gen("ZeroPad2d")
def gen_zeropad2d(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    padding = int(node.get("padding", 1))
    definition = f"nn.ZeroPad2d({padding})"
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    
    ctx.out_spatial_size = ctx.spatial_size + 2 * padding
    return definition, forward_expr

@register_layer_gen("CustomLayer")
def gen_custom_layer(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    code = node.get("code", "")
    class_name = node.get("class_name", "CustomLayer")
    args = node.get("args", "{}")
    
    ctx.helpers.add(f"CUSTOM:{code}")
    
    if args == "{}":
        definition = f"self.{class_name}()"
    else:
        definition = f"self.{class_name}({args})"
        
    forward_expr = f"self.{ctx.layer_name}({ctx.input_vars[0]})"
    return definition, forward_expr

# --- Main Generation Logic ---

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

def generate_inference_script(plan: Dict[str, Any]) -> str:
    """Generate a standalone PyTorch inference script with Gradio UI"""
    
    # Parse model structure
    model_struct = plan.get("model", {})
    nodes = model_struct.get("nodes", {})
    execution_order = model_struct.get("execution_order", [])
    connections = model_struct.get("connections", {})
    
    # Dataset info
    data_info = plan.get("data", {})
    dataset_name = data_info.get("dataset", "MNIST")
    
    # Determine output node
    output_node_id = None
    train_configs = plan.get("train", [])
    if train_configs:
        loss_node_id = train_configs[0].get("node_id")
        if loss_node_id:
            loss_sources = connections.get(str(loss_node_id), [])
            if loss_sources:
                output_node_id = str(loss_sources[0])
    
    if not output_node_id and execution_order:
        valid_nodes = [n for n in execution_order if nodes.get(str(n), {}).get("type") != "Loss"]
        if valid_nodes:
            output_node_id = str(valid_nodes[-1])
    
    # Subgraph Registry
    subgraph_registry = []
    subgraph_cache = {}

    # Generate code sections
    imports = _generate_imports()
    imports += "\nimport argparse\nimport os\nfrom PIL import Image\ntry:\n    import gradio as gr\nexcept ImportError:\n    print('Please install gradio: pip install gradio')"
    
    # Initial shape
    if dataset_name == "CIFAR-10":
        initial_shape = (3, 32, 32)
    elif dataset_name in ["MNIST", "Fashion-MNIST"]:
        initial_shape = (1, 28, 28)
    elif dataset_name == "WikiText-2":
        initial_shape = (1,) 
    else:
        initial_shape = (1, 28, 28)
        
    model_class_code, needed_helpers = _generate_module_class(
        class_name="Model",
        nodes=nodes,
        execution_order=execution_order,
        connections=connections,
        input_ids=["data"],
        output_ids=[output_node_id] if output_node_id else [],
        input_shapes=[initial_shape],
        subgraph_registry=subgraph_registry,
        subgraph_cache=subgraph_cache,
        num_classes=100 if dataset_name == "WikiText-2" else 10,
        is_main_model=True
    )
    
    subgraph_codes = "\n\n".join([code for _, code in subgraph_registry])
    
    # Inference specific code
    inference_utils = f"""
def get_transform():
    # Default transform based on dataset
    if "{dataset_name}" == "MNIST":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif "{dataset_name}" == "CIFAR-10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        # Generic transform
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

# Global model instance
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path='best_model.pt'):
    global model
    if not os.path.exists(model_path):
        return False
    
    try:
        model = Model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {{model_path}}")
        return True
    except Exception as e:
        print(f"Error loading model: {{e}}")
        return False

def predict(image):
    if model is None:
        if not load_model():
            return {{"Error": "Model not found or failed to load"}}
            
    transform = get_transform()
    
    try:
        if image is None:
            return None
            
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        # Assuming classification
        if output.dim() == 2:
            probabilities = F.softmax(output, dim=1)[0]
            # Return top 5
            topk_prob, topk_indices = torch.topk(probabilities, min(5, len(probabilities)))
            return {{str(idx.item()): float(prob) for idx, prob in zip(topk_indices, topk_prob)}}
        else:
            return {{"Output Shape": str(list(output.shape))}}
            
    except Exception as e:
        return {{"Error": str(e)}}

def main():
    parser = argparse.ArgumentParser(description='Run Gradio App')
    parser.add_argument('--model', type=str, default='best_model.pt', help='Path to model weights')
    parser.add_argument('--port', type=int, default=7860, help='Port to run Gradio on')
    parser.add_argument('--share', action='store_true', help='Create a public link')
    
    args = parser.parse_args()
    
    # Try to load model
    load_model(args.model)
    
    # Create Gradio Interface
    if "{dataset_name}" in ["MNIST", "Fashion-MNIST"]:
        input_component = gr.Image(type="pil", image_mode="L", label="Input Image")
    else:
        input_component = gr.Image(type="pil", label="Input Image")
        
    iface = gr.Interface(
        fn=predict,
        inputs=input_component,
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="CortexNodus Model Inference",
        description=f"Inference using model trained on {dataset_name}",
        allow_flagging="never"
    )
    
    print(f"Starting Gradio app on port {{args.port}}...")
    iface.launch(server_port=args.port, share=args.share)

if __name__ == '__main__':
    main()
"""

    script = f"""{imports}
import json

{subgraph_codes}

{model_class_code}

{inference_utils}
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
    for i, nid in enumerate(input_ids):
        if i < len(input_shapes):
            node_shapes[nid] = input_shapes[i]
        else:
            node_shapes[nid] = (1,) # Fallback
            
    layer_defs = []
    forward_lines = []
    needed_helpers = set()
    
    if is_main_model:
        forward_args = "x"
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
        if node_type in ["MNIST", "Fashion-MNIST", "CIFAR-10", "Loss", "CustomData", "WikiText-2", "graph/input", "graph/output"]:
            if node_type == "graph/output":
                source_ids = connections.get(node_id_str, [])
                if source_ids:
                    src = str(source_ids[0])
                    if src in var_map:
                        var_map[node_id_str] = var_map[src]
                    if src in node_shapes:
                        node_shapes[node_id_str] = node_shapes[src]
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
                current_input_shapes.append((1,))
                
            if src_str in var_map:
                current_input_vars.append(var_map[src_str])
            else:
                if is_main_model and src_str == "data":
                    current_input_vars.append("x")
                    current_input_shapes[-1] = node_shapes.get("data", (1, 28, 28))
                else:
                    current_input_vars.append(f"out_{src_str}")
        
        if not current_input_shapes:
            continue
            
        layer_name = f"layer_{node_id}"
        
        # Prepare Context
        in_shape = current_input_shapes[0]
        in_channels = in_shape[0]
        spatial_size = in_shape[1] if len(in_shape) == 3 else 0
        
        ctx = LayerGenContext(in_channels, spatial_size, num_classes, layer_name, current_input_vars, 
                              subgraph_registry, subgraph_cache, current_input_shapes)
        
        # Generate Layer Info
        definition = ""
        forward_expr = ""
        
        # Handle Subgraphs
        if node_type == "graph/subgraph" or node_type == "graph/embedded_subgraph":
            definition, forward_expr = _gen_subgraph(node, ctx)
        # Handle Registered Layers
        elif node_type in LAYER_GENERATORS:
            definition, forward_expr = LAYER_GENERATORS[node_type](node, ctx)
        else:
            # Fallback
            forward_expr = ctx.input_vars[0]
        
        if definition:
            layer_defs.append(f"self.{layer_name} = {definition}")
        
        # Assign output to variable
        out_var = f"out_{node_id_str}"
        forward_lines.append(f"{out_var} = {forward_expr}")
        var_map[node_id_str] = out_var
        
        # Update shapes
        if ctx.out_spatial_size > 0:
            node_shapes[node_id_str] = (ctx.out_channels, ctx.out_spatial_size, ctx.out_spatial_size)
        else:
            node_shapes[node_id_str] = (ctx.out_channels,)
            
        if ctx.helpers:
            needed_helpers.update(ctx.helpers)
    
    # Build class
    layers_init = "\n        ".join(layer_defs) if layer_defs else "pass"
    forward_body = "\n        ".join(forward_lines) if forward_lines else "pass"
    
    # Helper methods
    helper_code = _generate_helper_methods(needed_helpers)
    
    # Return statement
    if not output_ids:
        if forward_lines:
            last_var = forward_lines[-1].split(" = ")[0].strip()
            return_stmt = f"return {last_var}"
        else:
            return_stmt = "return x"
    else:
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

def _gen_subgraph(node: Dict[str, Any], ctx: LayerGenContext) -> Tuple[str, str]:
    subgraph_data = node.get("subgraph", {})
    if not subgraph_data:
        return "", ctx.input_vars[0]
        
    # Parse subgraph
    dag = parse_subgraph_dag(subgraph_data)
    
    # Generate unique class name
    json_str = json.dumps(subgraph_data, sort_keys=True)
    shapes_str = str(ctx.input_shapes)
    cache_key = (json_str, shapes_str)
    
    if cache_key in ctx.subgraph_cache:
        sub_class_name = ctx.subgraph_cache[cache_key]
    else:
        sub_class_name = f"Subgraph_{len(ctx.subgraph_registry)}"
        sub_code, _ = _generate_module_class(
            sub_class_name,
            dag["nodes"],
            dag["execution_order"],
            dag["connections"],
            dag["input_ids"],
            dag["output_ids"],
            ctx.input_shapes,
            ctx.subgraph_registry,
            ctx.subgraph_cache,
            ctx.num_classes
        )
        ctx.subgraph_registry.append((sub_class_name, sub_code))
        ctx.subgraph_cache[cache_key] = sub_class_name
    
    definition = f"{sub_class_name}()"
    args = ", ".join(ctx.input_vars)
    forward_expr = f"self.{ctx.layer_name}({args})"
    
    # Output shape approximation (same as input for now, or we need to run shape inference on subgraph)
    # Ideally _generate_module_class should return output shapes.
    # For now, keep existing behavior (Identity shape)
    return definition, forward_expr

def _generate_helper_methods(needed_helpers: Set[str]) -> str:
    methods = []
    
    # Handle Custom Layers
    for h in needed_helpers:
        if h.startswith("CUSTOM:"):
            code = h[7:]
            indented_code = "\n".join(["    " + line for line in code.split("\n")])
            methods.append("\n" + indented_code)

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
