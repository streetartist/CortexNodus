import json
import copy

class LiteGraphGenerator:
    def __init__(self):
        self.node_id_counter = 1
        self.link_id_counter = 1
        self.nodes = []
        self.links = []
        
    def get_new_id(self):
        nid = self.node_id_counter
        self.node_id_counter += 1
        return nid
        
    def get_new_link_id(self):
        lid = self.link_id_counter
        self.link_id_counter += 1
        return lid

    def add_node(self, type_name, pos, properties=None, inputs=None, outputs=None, title=None):
        nid = self.get_new_id()
        node = {
            "id": nid,
            "type": type_name,
            "pos": pos,
            "size": [140, 30],
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "properties": properties or {}
        }
        if title:
            node["title"] = title
            
        node["inputs"] = []
        if inputs:
            for i, inp in enumerate(inputs):
                # inp can be (node_id, slot_index) or just node_id (default slot 0)
                link_id = self.get_new_link_id()
                source_id = inp[0] if isinstance(inp, (list, tuple)) else inp
                source_slot = inp[1] if isinstance(inp, (list, tuple)) else 0
                
                node["inputs"].append({
                    "name": "in" if i == 0 else f"in{i+1}",
                    "type": "image",
                    "link": link_id
                })
                
                # Add link
                self.links.append([link_id, source_id, source_slot, nid, i, "image"])
                
                # Update source node outputs
                # We need to find the source node in self.nodes and update its outputs
                # This is a bit inefficient but works for generation
                for n in self.nodes:
                    if n["id"] == source_id:
                        if "outputs" not in n:
                            n["outputs"] = [{"name": "out", "type": "image", "links": []}]
                        # Ensure enough output slots? Usually just 1 'out'
                        # If source_slot > 0, we might need more.
                        while len(n["outputs"]) <= source_slot:
                             n["outputs"].append({"name": "out", "type": "image", "links": []})
                        n["outputs"][source_slot]["links"].append(link_id)
                        break
        
        if outputs:
            node["outputs"] = outputs
        else:
            # Default output if not specified, but links will be added when used as input
            node["outputs"] = [{"name": "out", "type": "image", "links": []}]
            
        self.nodes.append(node)
        return nid

    def to_json(self):
        return {
            "version": 0.4,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {}
        }

def create_mbconv_subgraph(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
    gen = LiteGraphGenerator()
    
    # Layout constants
    X_START = 50
    Y_START = 50
    X_STEP = 200
    
    current_x = X_START
    
    # 1. Input
    input_node = gen.add_node("graph/input", [current_x, Y_START], {"name": "in", "type": "image"})
    current_x += X_STEP
    last_node = input_node
    
    expanded_channels = in_channels * expand_ratio
    
    # 2. Expansion (if needed)
    if expand_ratio != 1:
        # Conv1x1
        conv_exp = gen.add_node("Conv2D", [current_x, Y_START], {
            "out_channels": expanded_channels,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0
        }, inputs=[last_node])
        current_x += X_STEP
        
        # BN
        bn_exp = gen.add_node("BatchNorm2d", [current_x, Y_START], {}, inputs=[conv_exp])
        current_x += X_STEP
        
        # SiLU
        silu_exp = gen.add_node("SiLU", [current_x, Y_START], {}, inputs=[bn_exp])
        current_x += X_STEP
        last_node = silu_exp
        
    # 3. Depthwise Conv
    # groups = expanded_channels
    # padding = kernel_size // 2
    dw_conv = gen.add_node("Conv2D", [current_x, Y_START], {
        "out_channels": expanded_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "groups": expanded_channels,
        "padding": kernel_size // 2
    }, inputs=[last_node])
    current_x += X_STEP
    
    bn_dw = gen.add_node("BatchNorm2d", [current_x, Y_START], {}, inputs=[dw_conv])
    current_x += X_STEP
    
    silu_dw = gen.add_node("SiLU", [current_x, Y_START], {}, inputs=[bn_dw])
    current_x += X_STEP
    last_node = silu_dw
    
    # 4. SE Block
    # Squeeze: GlobalAvgPool -> Reduce -> SiLU -> Expand -> Sigmoid
    # Excitation: Multiply
    
    # Branch off for SE
    se_input = last_node
    
    # Global Avg Pool
    gap = gen.add_node("AdaptiveAvgPool", [current_x, Y_START + 100], {"output_size": 1}, inputs=[se_input])
    current_x += X_STEP
    
    num_squeezed_channels = max(1, int(in_channels * se_ratio))
    
    # Reduce
    se_reduce = gen.add_node("Conv2D", [current_x, Y_START + 100], {
        "out_channels": num_squeezed_channels,
        "kernel_size": 1
    }, inputs=[gap])
    current_x += X_STEP
    
    se_silu = gen.add_node("SiLU", [current_x, Y_START + 100], {}, inputs=[se_reduce])
    current_x += X_STEP
    
    # Expand
    se_expand = gen.add_node("Conv2D", [current_x, Y_START + 100], {
        "out_channels": expanded_channels,
        "kernel_size": 1
    }, inputs=[se_silu])
    current_x += X_STEP
    
    se_sigmoid = gen.add_node("Sigmoid", [current_x, Y_START + 100], {}, inputs=[se_expand])
    current_x += X_STEP
    
    # Multiply (Scale)
    # Inputs: [last_node (Feature Map), se_sigmoid (Weights)]
    multiply = gen.add_node("Multiply", [current_x, Y_START], {}, inputs=[last_node, se_sigmoid])
    current_x += X_STEP
    last_node = multiply
    
    # 5. Pointwise Conv
    pw_conv = gen.add_node("Conv2D", [current_x, Y_START], {
        "out_channels": out_channels,
        "kernel_size": 1,
        "stride": 1
    }, inputs=[last_node])
    current_x += X_STEP
    
    bn_pw = gen.add_node("BatchNorm2d", [current_x, Y_START], {}, inputs=[pw_conv])
    current_x += X_STEP
    last_node = bn_pw
    
    # 6. Skip Connection
    if stride == 1 and in_channels == out_channels:
        # Add [input_node, last_node]
        # Wait, input_node is the very first input.
        # But if we had expansion, input_node has different channels than last_node (which is out_channels).
        # The skip connection connects the block's input to the block's output.
        # Condition: in_channels == out_channels and stride == 1.
        # So shapes match.
        
        # We need to route the original input to here.
        # But input_node is far back.
        # In LiteGraph, we can just link it.
        
        add_node = gen.add_node("Add", [current_x, Y_START], {}, inputs=[input_node, last_node])
        current_x += X_STEP
        last_node = add_node
        
    # 7. Output
    gen.add_node("graph/output", [current_x, Y_START], {"name": "out", "type": "image"}, inputs=[last_node])
    
    return gen.to_json()

def generate_efficientnet_b0():
    main_gen = LiteGraphGenerator()
    
    X_START = 50
    Y_START = 200
    X_STEP = 200
    current_x = X_START
    
    # 1. Data
    data_node = main_gen.add_node("MNIST", [current_x, Y_START], {"batch_size": 32}) # Using MNIST as placeholder for ImageNet/Custom
    current_x += X_STEP
    last_node = data_node
    
    # 2. Stem
    # Conv 3x3, s2, 32
    stem_conv = main_gen.add_node("Conv2D", [current_x, Y_START], {
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    }, inputs=[last_node])
    current_x += X_STEP
    
    stem_bn = main_gen.add_node("BatchNorm2d", [current_x, Y_START], {}, inputs=[stem_conv])
    current_x += X_STEP
    
    stem_silu = main_gen.add_node("SiLU", [current_x, Y_START], {}, inputs=[stem_bn])
    current_x += X_STEP
    last_node = stem_silu
    
    # 3. Blocks
    # Definition: (operator, kernel, stride, in_ch, out_ch, expand_ratio, num_layers)
    # EfficientNet-B0 config
    blocks_args = [
        # Stage 2
        {"k": 3, "s": 1, "in": 32, "out": 16, "e": 1, "n": 1},
        # Stage 3
        {"k": 3, "s": 2, "in": 16, "out": 24, "e": 6, "n": 2},
        # Stage 4
        {"k": 5, "s": 2, "in": 24, "out": 40, "e": 6, "n": 2},
        # Stage 5
        {"k": 3, "s": 2, "in": 40, "out": 80, "e": 6, "n": 3},
        # Stage 6
        {"k": 5, "s": 1, "in": 80, "out": 112, "e": 6, "n": 3}, # Original B0 has 3 layers here? Paper says 3.
        # Stage 7
        {"k": 5, "s": 2, "in": 112, "out": 192, "e": 6, "n": 4},
        # Stage 8
        {"k": 3, "s": 1, "in": 192, "out": 320, "e": 6, "n": 1},
    ]
    
    # For the sake of the example JSON size, I will reduce the number of repeats (n) to 1 for most stages,
    # but keep the structure.
    # Or I can generate all of them. Let's try generating all, but maybe compact layout.
    
    for i, args in enumerate(blocks_args):
        kernel = args["k"]
        stride = args["s"]
        in_c = args["in"]
        out_c = args["out"]
        expand = args["e"]
        num = args["n"]
        
        for j in range(num):
            # First layer in block handles stride and channel change
            current_stride = stride if j == 0 else 1
            current_in = in_c if j == 0 else out_c
            
            subgraph_json = create_mbconv_subgraph(
                in_channels=current_in,
                out_channels=out_c,
                kernel_size=kernel,
                stride=current_stride,
                expand_ratio=expand
            )
            
            block_node = main_gen.add_node("graph/subgraph", [current_x, Y_START], {
                "subgraph": subgraph_json
            }, inputs=[last_node], title=f"MBConv{expand} k{kernel} ({current_in}->{out_c})")
            
            current_x += X_STEP
            last_node = block_node
            
    # 4. Head
    # Conv 1x1, 1280
    head_conv = main_gen.add_node("Conv2D", [current_x, Y_START], {
        "out_channels": 1280,
        "kernel_size": 1,
        "stride": 1
    }, inputs=[last_node])
    current_x += X_STEP
    
    head_bn = main_gen.add_node("BatchNorm2d", [current_x, Y_START], {}, inputs=[head_conv])
    current_x += X_STEP
    
    head_silu = main_gen.add_node("SiLU", [current_x, Y_START], {}, inputs=[head_bn])
    current_x += X_STEP
    
    # Pooling
    pool = main_gen.add_node("AdaptiveAvgPool", [current_x, Y_START], {"output_size": 1}, inputs=[head_silu])
    current_x += X_STEP
    
    # Flatten
    flat = main_gen.add_node("Flatten", [current_x, Y_START], {}, inputs=[pool])
    current_x += X_STEP
    
    # Dropout
    drop = main_gen.add_node("Dropout", [current_x, Y_START], {"p": 0.2}, inputs=[flat])
    current_x += X_STEP
    
    # FC
    fc = main_gen.add_node("Dense", [current_x, Y_START], {"out_features": 10}, inputs=[drop]) # 10 classes for MNIST/CIFAR
    current_x += X_STEP
    
    # Loss
    loss = main_gen.add_node("Loss", [current_x, Y_START], {
        "kind": "CrossEntropy",
        "optimizer": "Adam",
        "lr": 0.001,
        "epochs": 5
    }, inputs=[fc])
    
    return main_gen.to_json()

if __name__ == "__main__":
    graph = generate_efficientnet_b0()
    with open("example/efficientnet_b0_subgraph.json", "w") as f:
        json.dump(graph, f, indent=2)
    print("Generated example/efficientnet_b0_subgraph.json")
