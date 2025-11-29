// 简易 LiteGraph 集成与属性面板、保存/加载、运行
const graph = new LGraph();
const canvas = new LGraphCanvas("#graphcanvas", graph);
const graphStack = []; // Stack of {graph: LGraph, node: SubgraphNode}

// SocketIO for real-time logging
const socket = io();
let logHistory = [];

// File management
let currentFileName = null;
let isSaved = true;

// Listen for graph changes
graph.onNodeAdded = markAsUnsaved;
graph.onNodeRemoved = markAsUnsaved;
graph.onConnectionChange = markAsUnsaved;

// Listen for graph changes
graph.onChange = () => {
    if (isSaved) {
        isSaved = false;
        updateFileStatus();
    }
};


// SocketIO event handlers
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
});

socket.on('log', function(logEntry) {
    addLogEntry(logEntry);
});

socket.on('log_history', function(logs) {
    logHistory = logs;
    updateLogDisplay();
});

function addLogEntry(entry) {
    logHistory.push(entry);
    // Keep only last 1000 entries
    if (logHistory.length > 1000) {
        logHistory = logHistory.slice(-1000);
    }
    updateLogDisplay();
}

function updateLogDisplay() {
    const modalLog = document.getElementById("modal-log");
    const miniLog = document.getElementById("mini-log");
    const miniLogContainer = document.getElementById("mini-log-container");
    
    const logHtml = logHistory.map(entry => {
        const levelClass = getLogLevelClass(entry.level);
        return `<div class="log-entry ${levelClass}">
            <span class="log-timestamp">[${entry.timestamp}]</span>
            <span class="log-level">${entry.level}</span>
            <span class="log-message">${entry.message}</span>
        </div>`;
    }).join('');
    
    if (modalLog) {
        const isAtBottom = modalLog.scrollTop + modalLog.clientHeight >= modalLog.scrollHeight - 10;
        modalLog.innerHTML = logHtml;
        if (isAtBottom) {
            modalLog.scrollTop = modalLog.scrollHeight;
        }
    }
    
    if (miniLog) {
        const isAtBottom = miniLog.scrollTop + miniLog.clientHeight >= miniLog.scrollHeight - 10;
        miniLog.innerHTML = logHtml;
        if (isAtBottom) {
            miniLog.scrollTop = miniLog.scrollHeight;
        }
        // Always show container if we have logs
        if (miniLogContainer && logHistory.length > 0) {
            miniLogContainer.style.display = "block";
        }
    }
}

function getLogLevelClass(level) {
    switch(level) {
        case 'INFO': return 'log-info';
        case 'WARNING': return 'log-warning';
        case 'ERROR': return 'log-error';
        case 'DEBUG': return 'log-debug';
        default: return 'log-default';
    }
}

function updateFileStatus() {
    const fileInfo = document.getElementById("current-file");
    if (currentFileName) {
        fileInfo.textContent = currentFileName + (isSaved ? "" : " *");
    } else {
        fileInfo.textContent = "未保存的文件" + (isSaved ? "" : " *");
    }
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function markAsUnsaved() {
    if (isSaved) {
        isSaved = false;
        updateFileStatus();
    }
}
function EmbeddedSubgraphNode() {
    this.properties = {
        template_name: "",
        is_editing: false
    };
    this.size = [180, 80];
    this.serialize_widgets = true;
    this.skip_subgraph_button = true;
    
    // Create embedded subgraph
    this.subgraph = new LGraph();
    
    var that = this;
    this.subgraph.onNodeAdded = function(node) {
        if (!that.properties.is_editing) {
            that.updateSlots();
        }
    };
    this.subgraph.onNodeRemoved = function(node) {
        if (!that.properties.is_editing) {
            that.updateSlots();
        }
    };
    
    // Add default input/output nodes
    this.addDefaultNodes();
    
    // Set initial title
    this.updateTitle();
}

EmbeddedSubgraphNode.title = "Inline Subgraph";
EmbeddedSubgraphNode.desc = "Reusable subgraph (inline)";

EmbeddedSubgraphNode.prototype.addDefaultNodes = function() {
    var inputNode = LiteGraph.createNode("graph/input");
    inputNode.pos = [100, 100];
    this.subgraph.add(inputNode);
    
    var outputNode = LiteGraph.createNode("graph/output");
    outputNode.pos = [400, 100];
    this.subgraph.add(outputNode);
}

EmbeddedSubgraphNode.prototype.updateSlots = function() {
    var inputNodes = this.subgraph.findNodesByType("graph/input");
    var outputNodes = this.subgraph.findNodesByType("graph/output");
    
    // Sort by Y position
    inputNodes.sort(function(a,b) { return a.pos[1] - b.pos[1]; });
    outputNodes.sort(function(a,b) { return a.pos[1] - b.pos[1]; });
    
    // Sync Inputs
    var currentInputs = this.inputs ? this.inputs.length : 0;
    var neededInputs = inputNodes.length;
    
    for (var i = currentInputs; i < neededInputs; i++) {
        this.addInput("in", "any");
    }
    for (var i = currentInputs; i > neededInputs; i--) {
        this.removeInput(i-1);
    }
    
    for (var i = 0; i < neededInputs; i++) {
        var name = inputNodes[i].properties.name || ("in_" + (i+1));
        if (this.inputs[i]) this.inputs[i].name = name;
    }
    
    // Sync Outputs
    var currentOutputs = this.outputs ? this.outputs.length : 0;
    var neededOutputs = outputNodes.length;
    
    for (var i = currentOutputs; i < neededOutputs; i++) {
        this.addOutput("out", "any");
    }
    for (var i = currentOutputs; i > neededOutputs; i--) {
        this.removeOutput(i-1);
    }
    
    for (var i = 0; i < neededOutputs; i++) {
        var name = outputNodes[i].properties.name || ("out_" + (i+1));
        if (this.outputs[i]) this.outputs[i].name = name;
    }
}

EmbeddedSubgraphNode.prototype.onDblClick = function(e, pos, graphCanvas) {
    // Enter editing mode
    this.properties.is_editing = true;
    
    // Push current graph to stack
    graphStack.push({ graph: graphCanvas.graph, node: this });
    
    // Switch to subgraph
    graphCanvas.setGraph(this.subgraph);
    updateBreadcrumbs();
}

EmbeddedSubgraphNode.prototype.onSerialize = function(o) {
    o.subgraph = this.subgraph.serialize();
    o.template_name = this.properties.template_name;
}

EmbeddedSubgraphNode.prototype.onConfigure = function(o) {
    this.properties.template_name = o.template_name || (o.properties && o.properties.template_name) || "";
    
    var sub = o.subgraph || (o.properties && o.properties.subgraph);
    if (sub) {
        try {
            // Temporarily disable events to prevent slot destruction during clear()
            var tmp_added = this.subgraph.onNodeAdded;
            var tmp_removed = this.subgraph.onNodeRemoved;
            this.subgraph.onNodeAdded = null;
            this.subgraph.onNodeRemoved = null;
            
            this.subgraph.clear();
            this.subgraph.configure(sub);
            
            this.subgraph.onNodeAdded = tmp_added;
            this.subgraph.onNodeRemoved = tmp_removed;
            
            this.updateSlots();
        } catch (e) {
            console.error("Error configuring inline subgraph:", e);
        }
    }
    
    this.updateTitle();
}

EmbeddedSubgraphNode.prototype.getExtraMenuOptions = function(canvas, options) {
    var that = this;
    return [
        {
            content: "保存为模板",
            callback: function() {
                var name = prompt("输入模板名称:");
                if (name && name.trim()) {
                    if (embeddedSubgraphTemplates[name] && name !== that.properties.template_name) {
                        alert("模板名称已存在，请选择其他名称。");
                        return;
                    }
                    that.properties.template_name = name;
                    that.updateTitle();
                    that.saveTemplate();
                }
            }
        },
        {
            content: "重置为默认",
            callback: function() {
                if (confirm("重置为默认状态？")) {
                    that.resetToDefault();
                }
            }
        }
    ];
}

EmbeddedSubgraphNode.prototype.updateTitle = function() {
    if (this.properties.template_name) {
        this.title = this.properties.template_name;
    } else {
        this.title = "Inline Subgraph";
    }
}

EmbeddedSubgraphNode.prototype.resetToDefault = function() {
    this.properties.template_name = "";
    
    // Clear and reset subgraph
    this.subgraph.clear();
    this.addDefaultNodes();
    this.updateSlots();
    this.updateTitle();
}

// Register the Embedded Subgraph node
LiteGraph.registerNodeType("graph/embedded_subgraph", EmbeddedSubgraphNode);

// Legacy SubgraphNode for backward compatibility
function SubgraphNode() {
    this.properties = {};
    this.subgraph = new LGraph();
    this.skip_subgraph_button = true;
    
    var that = this;
    this.subgraph.onNodeAdded = function(node) {
        that.updateSlots();
    };
    this.subgraph.onNodeRemoved = function(node) {
        that.updateSlots();
    };
    
    // Default nodes - only if empty (newly created)
    var inputNode = LiteGraph.createNode("graph/input");
    inputNode.pos = [100, 100];
    this.subgraph.add(inputNode);
    
    var outputNode = LiteGraph.createNode("graph/output");
    outputNode.pos = [400, 100];
    this.subgraph.add(outputNode);
}

SubgraphNode.prototype.updateSlots = function() {
    var inputNodes = this.subgraph.findNodesByType("graph/input");
    var outputNodes = this.subgraph.findNodesByType("graph/output");
    
    // Sort by Y position
    inputNodes.sort(function(a,b) { return a.pos[1] - b.pos[1]; });
    outputNodes.sort(function(a,b) { return a.pos[1] - b.pos[1]; });
    
    // Sync Inputs
    var currentInputs = this.inputs ? this.inputs.length : 0;
    var neededInputs = inputNodes.length;
    
    // console.log("Syncing inputs: current", currentInputs, "needed", neededInputs);

    for (var i = currentInputs; i < neededInputs; i++) {
        this.addInput("in", "any");
    }
    for (var i = currentInputs; i > neededInputs; i--) {
        this.removeInput(i-1);
    }
    
    for (var i = 0; i < neededInputs; i++) {
        var name = inputNodes[i].properties.name || ("in_" + (i+1));
        if (this.inputs[i]) this.inputs[i].name = name;
    }
    
    // Sync Outputs
    var currentOutputs = this.outputs ? this.outputs.length : 0;
    var neededOutputs = outputNodes.length;
    
    // console.log("Syncing outputs: current", currentOutputs, "needed", neededOutputs);

    for (var i = currentOutputs; i < neededOutputs; i++) {
        this.addOutput("out", "any");
    }
    for (var i = currentOutputs; i > neededOutputs; i--) {
        this.removeOutput(i-1);
    }
    
    for (var i = 0; i < neededOutputs; i++) {
        var name = outputNodes[i].properties.name || ("out_" + (i+1));
        if (this.outputs[i]) this.outputs[i].name = name;
    }
}

SubgraphNode.title = "Subgraph";
SubgraphNode.prototype.onDblClick = function(e, pos, graphCanvas) {
    // Push current graph to stack
    graphStack.push({ graph: graphCanvas.graph, node: this });
    
    // Switch to subgraph
    graphCanvas.setGraph(this.subgraph);
    updateBreadcrumbs();
}

SubgraphNode.prototype.onSerialize = function(o) {
    o.subgraph = this.subgraph.serialize();
}

SubgraphNode.prototype.onConfigure = function(o) {
    var sub = o.subgraph || (o.properties && o.properties.subgraph);
    if (sub) {
        try {
            // Temporarily disable events to prevent slot destruction during clear()
            var tmp_added = this.subgraph.onNodeAdded;
            var tmp_removed = this.subgraph.onNodeRemoved;
            this.subgraph.onNodeAdded = null;
            this.subgraph.onNodeRemoved = null;
            
            this.subgraph.clear();
            this.subgraph.configure(sub);
            
            this.subgraph.onNodeAdded = tmp_added;
            this.subgraph.onNodeRemoved = tmp_removed;
            
            this.updateSlots();
        } catch (e) {
            console.error("Error configuring subgraph:", e);
        }
    }
}

LiteGraph.registerNodeType("graph/subgraph", SubgraphNode);

function updateBreadcrumbs() {
    const el = document.getElementById("breadcrumbs");
    const pathEl = document.getElementById("breadcrumb-path");
    
    if (graphStack.length === 0) {
        el.style.display = "none";
        return;
    }
    
    el.style.display = "flex";
    let html = `<span class="crumb" onclick="gotoRoot()">Root</span>`;
    
    graphStack.forEach((item, index) => {
        html += `<span class="separator">&gt;</span>`;
        // If it's the last item (current view), make it non-clickable or styled differently if desired
        // But here we are inside a subgraph, so the stack contains the PARENTS.
        // Wait, graphStack contains the path TO the current graph?
        // Let's check logic:
        // On dblclick: push {graph: current, node: this}, setGraph(this.subgraph)
        // So stack has [RootGraph, Level1Graph...]
        // The current view is NOT in the stack, it's the active graph.
        // So the stack represents the path.
        
        html += `<span class="crumb" onclick="gotoLevel(${index})">${item.node.title || "Subgraph"}</span>`;
    });
    
    // Add "Current" indicator? 
    // Actually, the stack represents the parents. The current view is "inside" the last node in stack?
    // No, wait.
    // Root -> SubgraphNode (dblclick) -> Pushes Root to stack. Current is Subgraph.
    // So stack has [Root]. Breadcrumb should show: Root > Subgraph (Current)
    
    // But we don't know the name of the current subgraph easily unless we store it.
    // The `item.node` in stack is the node that *contains* the subgraph we are currently in?
    // No, `item.node` is the node in the `item.graph`.
    // When we double click node A in Root:
    // Stack pushes {graph: Root, node: A}.
    // We switch to A.subgraph.
    // So stack[0].node.title is the name of the current context.
    
    // So: Root > NodeA > NodeB
    // Stack: 
    // 0: {graph: Root, node: NodeA}
    // 1: {graph: NodeA.subgraph, node: NodeB}
    
    // The breadcrumb should list: Root, then NodeA, then NodeB...
    // And finally... "Current"? 
    // Actually, the last item in the breadcrumb IS the current context.
    // If stack has 1 item (NodeA), we are INSIDE NodeA.
    // So the path is Root > NodeA.
    
    // Let's rebuild:
    html = `<span class="crumb" onclick="gotoRoot()">Root</span>`;
    
    graphStack.forEach((item, index) => {
        html += `<span class="separator">/</span>`;
        // The last one is the current context, maybe disable click?
        // But user might want to click to "refresh" or just see name.
        // Let's keep it simple.
        const isLast = index === graphStack.length - 1;
        const title = item.node.title || "Subgraph";
        if (isLast) {
             html += `<span class="current">${title}</span>`;
        } else {
             html += `<span class="crumb" onclick="gotoLevel(${index})">${title}</span>`;
        }
    });
    
    pathEl.innerHTML = html;

    // Toggle subgraph-only nodes in palette
    const subgraphNodes = document.querySelectorAll(".subgraph-only");
    subgraphNodes.forEach(btn => {
        btn.style.display = (graphStack.length > 0) ? "block" : "none";
    });
}

window.gotoRoot = function() {
    if (graphStack.length === 0) return;
    
    // Save template if we're exiting an inline subgraph
    if (graphStack.length > 0) {
        const currentNode = graphStack[graphStack.length - 1].node;
        if (currentNode && currentNode.constructor.name === 'EmbeddedSubgraphNode') {
            currentNode.saveTemplate();
        }
    }
    
    const root = graphStack[0].graph;
    graphStack.length = 0;
    canvas.setGraph(root);
    updateBreadcrumbs();
}

window.gotoLevel = function(index) {
    if (index < 0 || index >= graphStack.length) return;
    
    // Save template if we're exiting an inline subgraph
    if (index < graphStack.length - 1) {
        const currentNode = graphStack[graphStack.length - 1].node;
        if (currentNode && currentNode.constructor.name === 'EmbeddedSubgraphNode') {
            currentNode.saveTemplate();
        }
    }
    
    // We want to go TO the subgraph of the node at index.
    // So we keep 0..index in the stack.
    const targetItem = graphStack[index];
    graphStack.length = index + 1;
    canvas.setGraph(targetItem.node.subgraph);
    updateBreadcrumbs();
}

// Auto resize
function resize() {
  const wrap = document.querySelector(".canvas-wrap");
  if (wrap) {
    canvas.resize(wrap.clientWidth, wrap.clientHeight);
  }
}
window.addEventListener("resize", resize);
// Initial resize
setTimeout(resize, 50);

// 注册基础节点
function addBasicNode(type, title, properties = {}, inputs = [], outputs = []) {
  function NodeCtor() {
    this.title = title;
    this.properties = JSON.parse(JSON.stringify(properties));
    if (inputs) inputs.forEach(i => this.addInput(i, "any"));
    if (outputs) outputs.forEach(o => this.addOutput(o, "any"));
  }
  NodeCtor.title = title;
  NodeCtor.prototype.onExecute = function() {};
  NodeCtor.prototype.onPropertyChanged = function(name, value){ this.properties[name]=value; refreshInspector(this); };
  NodeCtor.prototype.onSelected = function(){ refreshInspector(this); };
  NodeCtor.prototype.onDeselected = function(){ refreshInspector(null); };
  // Draw type in corner
  NodeCtor.prototype.onDrawForeground = function(ctx) {
      if (this.flags.collapsed) return;
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.font = "10px Arial";
      ctx.textAlign = "right";
      ctx.fillText(type, this.size[0] - 6, this.size[1] - 6);
      ctx.restore();
  };
  LiteGraph.registerNodeType(type, NodeCtor);
}

// 节点定义配置表
const NODE_DEFINITIONS = [
  // Data
  { type: "MNIST", title: "MNIST", props: { batch_size: 64 }, out: ["data"] },
  { type: "Fashion-MNIST", title: "Fashion-MNIST", props: { batch_size: 64 }, out: ["data"] },
  { type: "CIFAR-10", title: "CIFAR-10", props: { batch_size: 64 }, out: ["data"] },
  { type: "WikiText-2", title: "WikiText-2", props: { batch_size: 20 }, out: ["data"] },
  { type: "WikiText-103", title: "WikiText-103", props: { batch_size: 20 }, out: ["data"] },
  { type: "PennTreebank", title: "Penn Treebank", props: { batch_size: 20 }, out: ["data"] },
  { type: "CustomData", title: "Custom Data", props: { batch_size: 32, path: "", type: "ImageFolder", input_shape: "3,128,128" }, out: ["data"] },
  
  // Layers - Conv
  { type: "Conv2D", title: "Conv2D", props: { out_channels: 8, kernel_size: 3 }, in: ["in"], out: ["out"] },
  { type: "MaxPool", title: "MaxPool", props: { kernel_size: 2 }, in: ["in"], out: ["out"] },
  { type: "AvgPool", title: "AvgPool", props: { kernel_size: 2 }, in: ["in"], out: ["out"] },
  { type: "AdaptiveAvgPool", title: "AdaptiveAvgPool", props: { output_size: 1 }, in: ["in"], out: ["out"] },
  { type: "ConvTranspose2d", title: "ConvTranspose2d", props: { out_channels: 8, kernel_size: 3, stride: 1, padding: 0, output_padding: 0 }, in: ["in"], out: ["out"] },
  { type: "Upsample", title: "Upsample", props: { scale_factor: 2, mode: "nearest" }, in: ["in"], out: ["out"] },
  { type: "PixelShuffle", title: "PixelShuffle", props: { upscale_factor: 2 }, in: ["in"], out: ["out"] },
  { type: "ZeroPad2d", title: "ZeroPad2d", props: { padding: 1 }, in: ["in"], out: ["out"] },
  
  // Layers - Activation
  { type: "ReLU", title: "ReLU", props: {}, in: ["in"], out: ["out"] },
  { type: "LeakyReLU", title: "LeakyReLU", props: { negative_slope: 0.01 }, in: ["in"], out: ["out"] },
  { type: "PReLU", title: "PReLU", props: {}, in: ["in"], out: ["out"] },
  { type: "Sigmoid", title: "Sigmoid", props: {}, in: ["in"], out: ["out"] },
  { type: "Tanh", title: "Tanh", props: {}, in: ["in"], out: ["out"] },
  { type: "ELU", title: "ELU", props: { alpha: 1.0 }, in: ["in"], out: ["out"] },
  { type: "GELU", title: "GELU", props: {}, in: ["in"], out: ["out"] },
  { type: "SiLU", title: "SiLU", props: {}, in: ["in"], out: ["out"] },
  { type: "Hardswish", title: "Hardswish", props: {}, in: ["in"], out: ["out"] },
  { type: "Hardsigmoid", title: "Hardsigmoid", props: {}, in: ["in"], out: ["out"] },
  { type: "Softplus", title: "Softplus", props: {}, in: ["in"], out: ["out"] },
  { type: "Softmax", title: "Softmax", props: { dim: 1 }, in: ["in"], out: ["out"] },
  { type: "LogSoftmax", title: "LogSoftmax", props: { dim: 1 }, in: ["in"], out: ["out"] },

  // Layers - Regularization & Core
  { type: "BatchNorm2d", title: "BatchNorm2d", props: {}, in: ["in"], out: ["out"] },
  { type: "Dropout", title: "Dropout", props: { p: 0.5 }, in: ["in"], out: ["out"] },
  { type: "Flatten", title: "Flatten", props: {}, in: ["in"], out: ["out"] },
  { type: "Dense", title: "Dense", props: { out_features: 10 }, in: ["in"], out: ["out"] },
  { type: "Identity", title: "Identity", props: {}, in: ["in"], out: ["out"] },
  { type: "Add", title: "Add", props: {}, in: ["in1", "in2"], out: ["out"] },
  { type: "Multiply", title: "Multiply", props: {}, in: ["in1", "in2"], out: ["out"] },
  { type: "Concat", title: "Concat", props: { dim: 1 }, in: ["in1", "in2"], out: ["out"] },
  { type: "BatchNorm1d", title: "BatchNorm1d", props: {}, in: ["in"], out: ["out"] },
  { type: "InstanceNorm2d", title: "InstanceNorm2d", props: {}, in: ["in"], out: ["out"] },
  { type: "GroupNorm", title: "GroupNorm", props: { num_groups: 4 }, in: ["in"], out: ["out"] },
  { type: "LayerNorm", title: "LayerNorm", props: {}, in: ["in"], out: ["out"] },
  { type: "Dropout2d", title: "Dropout2d", props: { p: 0.5 }, in: ["in"], out: ["out"] },
  { type: "AlphaDropout", title: "AlphaDropout", props: { p: 0.5 }, in: ["in"], out: ["out"] },
  
  // Custom
  { type: "CustomLayer", title: "Custom Layer", props: { 
      code: "class MyLayer(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return x", 
      class_name: "MyLayer", 
      args: "{}" 
    }, in: ["in"], out: ["out"] },
  
  // Transformer & NLP
  { type: "Embedding", title: "Embedding", props: { num_embeddings: 1000, embedding_dim: 128 }, in: ["in"], out: ["out"] },
  { type: "PositionalEncoding", title: "Positional Encoding", props: { max_len: 512, d_model: 128, dropout: 0.1 }, in: ["in"], out: ["out"] },
  { type: "TransformerEncoderLayer", title: "Transformer Encoder Layer", props: { d_model: 128, nhead: 8, dim_feedforward: 512, dropout: 0.1 }, in: ["in"], out: ["out"] },
  { type: "TransformerEncoder", title: "Transformer Encoder", props: { d_model: 128, nhead: 8, num_layers: 6, dim_feedforward: 512, dropout: 0.1 }, in: ["in"], out: ["out"] },
  { type: "TransformerDecoderLayer", title: "Transformer Decoder Layer", props: { d_model: 128, nhead: 8, dim_feedforward: 512, dropout: 0.1 }, in: ["in"], out: ["out"] },
  { type: "TransformerDecoder", title: "Transformer Decoder", props: { d_model: 128, nhead: 8, num_layers: 6, dim_feedforward: 512, dropout: 0.1 }, in: ["in"], out: ["out"] },
  { type: "MultiheadAttention", title: "Multihead Attention", props: { embed_dim: 128, num_heads: 8, dropout: 0.0 }, in: ["in"], out: ["out"] },
  { type: "Linear", title: "Linear", props: { out_features: 10 }, in: ["in"], out: ["out"] },
  { type: "GPTBlock", title: "GPT Block", props: { d_model: 128, nhead: 8, dim_feedforward: 512, dropout: 0.1 }, in: ["in"], out: ["out"] },
  
  // Subgraph & Structure
  { type: "graph/input", title: "Graph Input", props: { name: "in", type: "any" }, out: ["out"] },
  { type: "graph/output", title: "Graph Output", props: { name: "out", type: "any" }, in: ["in"] },

  // Training
  { type: "Loss", title: "Training Goal", props: { kind: "CrossEntropy", optimizer: "Adam", lr: 0.001, epochs: 10, target: "Label", weight: 1.0 }, in: ["in"] },
];

// 批量注册
NODE_DEFINITIONS.forEach(def => {
  addBasicNode(def.type, def.title, def.props, def.in, def.out);
});

// Shape Inference & Validation
function validateGraph() {
  const nodes = graph.computeExecutionOrder(false);
  // Map node_id -> output_shape [C, H, W] (or [C] if flat)
  const shapes = {};
  
  // Reset colors
  graph._nodes.forEach(n => {
    n.bgcolor = null; 
    n.boxcolor = null;
  });

  // Find data node
  const dataNode = graph._nodes.find(n => ["MNIST", "Fashion-MNIST", "CIFAR-10", "WikiText-2", "WikiText-103", "PennTreebank", "CustomData"].includes(n.type));
  if (!dataNode) return;

  // Init shape based on dataset
  let initShape = [1, 28, 28];
  if (dataNode.type === "CIFAR-10") initShape = [3, 32, 32];
  if (["WikiText-2", "WikiText-103", "PennTreebank"].includes(dataNode.type)) initShape = [35]; // sequence length for language modeling
  if (dataNode.type === "CustomData") {
      try {
          initShape = dataNode.properties.input_shape.split(",").map(Number);
      } catch(e) { initShape = [3, 128, 128]; }
  }
  shapes[dataNode.id] = initShape;

  // Traverse
  nodes.forEach(node => {
    if (["MNIST", "Fashion-MNIST", "CIFAR-10", "WikiText-2", "WikiText-103", "PennTreebank", "CustomData"].includes(node.type)) return;
    
    // Get input shape from first connected link
    if (!node.inputs || !node.inputs[0] || !node.inputs[0].link) return;
    const linkId = node.inputs[0].link;
    const link = graph.links[linkId];
    if (!link) return;
    const inShape = shapes[link.origin_id];
    
    if (!inShape) return; // Upstream error

    try {
      const outShape = computeNodeOutputShape(node, inShape);
      shapes[node.id] = outShape;
      // Optional: Show shape in title or tooltip?
      // node.title = node.type + " " + JSON.stringify(outShape);
    } catch (e) {
      console.warn("Shape error at node " + node.id, e);
      node.bgcolor = "#550000"; // Dark Red
      node.boxcolor = "#ff0000"; // Red border
    }
  });
  
  graph.setDirtyCanvas(true, true);
}

function computeNodeOutputShape(node, inShape) {
  const p = node.properties;
  const type = node.type;
  
  // Helper: Check dims
  const is2D = inShape.length === 3;
  const C = is2D ? inShape[0] : inShape[0];
  const H = is2D ? inShape[1] : 1;
  const W = is2D ? inShape[2] : 1;

  if (["Conv2D"].includes(type)) {
    if (!is2D) throw "Input must be 3D (C,H,W)";
    const k = parseInt(p.kernel_size||3);
    const s = parseInt(p.stride||1);
    const pad = parseInt(p.padding||0); // Default padding logic in backend is k//2 if not specified, but here we use explicit or default
    // Backend logic: if padding not in props, it uses k//2. 
    // But wait, in backend: pad = k // 2. 
    // Let's match backend exactly.
    const padding = (p.padding !== undefined) ? parseInt(p.padding) : Math.floor(k/2);
    
    const outC = parseInt(p.out_channels||8);
    const outH = Math.floor((H + 2*padding - k)/s + 1);
    const outW = Math.floor((W + 2*padding - k)/s + 1);
    if (outH <= 0 || outW <= 0) throw "Spatial dim became <= 0";
    return [outC, outH, outW];
  }
  
  if (["ConvTranspose2d"].includes(type)) {
    if (!is2D) throw "Input must be 3D";
    const k = parseInt(p.kernel_size||3);
    const s = parseInt(p.stride||1);
    const pad = parseInt(p.padding||0);
    const outPad = parseInt(p.output_padding||0);
    const outC = parseInt(p.out_channels||8);
    
    const outH = (H - 1)*s - 2*pad + k + outPad;
    const outW = (W - 1)*s - 2*pad + k + outPad;
    return [outC, outH, outW];
  }

  if (["MaxPool", "AvgPool"].includes(type)) {
    if (!is2D) throw "Input must be 3D";
    const k = parseInt(p.kernel_size||2);
    // Backend uses default stride = kernel_size
    const s = k; 
    const outH = Math.floor(H/s);
    const outW = Math.floor(W/s);
    if (outH < 1 || outW < 1) throw "Spatial dim too small";
    return [C, outH, outW];
  }
  
  if (["AdaptiveAvgPool"].includes(type)) {
      if (!is2D) throw "Input must be 3D";
      const os = parseInt(p.output_size||1);
      return [C, os, os];
  }

  if (["Flatten"].includes(type)) {
    if (is2D) return [C*H*W];
    return inShape;
  }

  if (["Dense"].includes(type)) {
    // Dense can take 2D (auto-flatten) or 1D
    const outF = parseInt(p.out_features||10);
    return [outF];
  }
  
  if (["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "GELU", "SiLU", "Softmax", "LogSoftmax", "Identity", "Dropout", "Dropout2d", "AlphaDropout", "BatchNorm2d", "BatchNorm1d", "InstanceNorm2d", "GroupNorm", "LayerNorm"].includes(type)) {
      return inShape; // Shape unchanged
  }
  
  if (["Loss", "Optimizer", "TrainRunner"].includes(type)) {
      return inShape;
  }

  if (["Upsample"].includes(type)) {
      if (!is2D) throw "Input must be 3D";
      const sf = parseFloat(p.scale_factor||2.0);
      const outH = Math.floor(H * sf);
      const outW = Math.floor(W * sf);
      return [C, outH, outW];
  }

  if (["PixelShuffle"].includes(type)) {
      if (!is2D) throw "Input must be 3D";
      const uf = parseInt(p.upscale_factor||2);
      const outC = Math.floor(C / (uf*uf));
      if (outC < 1) throw "Channels too small for PixelShuffle";
      const outH = H * uf;
      const outW = W * uf;
      return [outC, outH, outW];
  }

  if (["ZeroPad2d"].includes(type)) {
      if (!is2D) throw "Input must be 3D";
      const pad = parseInt(p.padding||1);
      return [C, H + 2*pad, W + 2*pad];
  }
  
  if (["PReLU", "Softplus", "Hardswish", "Hardsigmoid"].includes(type)) {
      return inShape;
  }

  return inShape;
}

// Hook into graph events
graph.onAfterExecute = validateGraph;
// Also validate on connection change
const origConnect = LGraphNode.prototype.connect;
LGraphNode.prototype.connect = function() {
    const r = origConnect.apply(this, arguments);
    validateGraph();
    return r;
}
// Validate periodically
setInterval(validateGraph, 1000);

// Inspector
const inspectorEl = document.getElementById("inspector");
function refreshInspector(node){
  inspectorEl.innerHTML = "";
  if(!node){ return; }

  // Type Display (Read-only)
  const typeWrap = document.createElement("div");
  typeWrap.style.marginBottom = "10px";
  typeWrap.style.paddingBottom = "5px";
  typeWrap.style.borderBottom = "1px dashed #444";
  typeWrap.style.color = "#aaa";
  typeWrap.style.fontSize = "12px";
  typeWrap.innerHTML = `Type: <span style="color:#fff; font-weight:bold;">${node.type}</span>`;
  inspectorEl.appendChild(typeWrap);

  // Title Editor
  const titleWrap = document.createElement("div");
  titleWrap.style.marginBottom = "10px";
  titleWrap.style.borderBottom = "1px solid #444";
  titleWrap.style.paddingBottom = "10px";
  
  const titleLabel = document.createElement("label"); 
  titleLabel.textContent = "Node Name";
  titleLabel.style.fontWeight = "bold";
  titleLabel.style.display = "block";
  
  const titleInput = document.createElement("input");
  titleInput.value = node.title;
  titleInput.style.width = "100%";
  titleInput.addEventListener("change", () => {
      node.title = titleInput.value;
  });
  
  titleWrap.appendChild(titleLabel);
  titleWrap.appendChild(titleInput);
  inspectorEl.appendChild(titleWrap);

  const props = node.properties || {};
  Object.keys(props).forEach(k => {
    const wrap = document.createElement("div");
    const label = document.createElement("label"); label.textContent = k;
    
    let input;
    // Special handling for dropdowns
    if (k === "kind" && node.title === "Training Goal") {
        input = document.createElement("select");
        ["CrossEntropy", "MSE", "SmoothL1"].forEach(opt => {
            const o = document.createElement("option"); o.value = opt; o.text = opt;
            if(props[k]===opt) o.selected = true;
            input.appendChild(o);
        });
    } else if (k === "optimizer" && node.title === "Training Goal") {
        input = document.createElement("select");
        ["Adam", "SGD", "AdamW"].forEach(opt => {
            const o = document.createElement("option"); o.value = opt; o.text = opt;
            if(props[k]===opt) o.selected = true;
            input.appendChild(o);
        });
    } else if (k === "target" && node.title === "Training Goal") {
        input = document.createElement("select");
        ["Label", "Input"].forEach(opt => {
            const o = document.createElement("option"); o.value = opt; o.text = opt;
            if(props[k]===opt) o.selected = true;
            input.appendChild(o);
        });
    } else if (k === "type" && node.title === "Custom Data") {
        input = document.createElement("select");
        ["ImageFolder", "CSV", "Numpy"].forEach(opt => {
            const o = document.createElement("option"); o.value = opt; o.text = opt;
            if(props[k]===opt) o.selected = true;
            input.appendChild(o);
        });
    } else if (k === "code") {
        input = document.createElement("textarea");
        input.value = props[k];
        input.style.width = "100%";
        input.style.height = "150px";
        input.style.fontFamily = "monospace";
        input.style.fontSize = "12px";
    } else {
        input = document.createElement("input"); input.value = props[k];
    }
    
    input.addEventListener("change", () => { node.setProperty(k, input.value*1 || input.value); });
    wrap.appendChild(label); wrap.appendChild(input); inspectorEl.appendChild(wrap);
  });
}

// Palette spawner
document.querySelectorAll(".palette .node").forEach(btn => {
  btn.addEventListener("click", () => {
    const type = btn.dataset.type;
    const node = LiteGraph.createNode(type);
    node.pos = [Math.random()*400+100, Math.random()*300+100];
    canvas.graph.add(node);
  });
});

// Node search functionality
document.getElementById("node-search").addEventListener("input", function() {
  const searchTerm = this.value.toLowerCase();
  const nodes = document.querySelectorAll(".palette .node");
  const groups = document.querySelectorAll(".palette .group");
  const embeddedSubgraphList = document.getElementById("embedded-subgraph-list");
  const subgraphList = document.getElementById("subgraph-list");
  const btnNewEmbedded = document.getElementById("btn-new-embedded-subgraph");
  const btnNewSubgraph = document.getElementById("btn-new-subgraph");
  
  nodes.forEach(node => {
    const text = node.textContent.toLowerCase();
    const matches = text.includes(searchTerm);
    // Override hidden class for search results
    if (matches) {
      node.style.display = "";
      node.classList.remove("hidden");
    } else {
      node.style.display = "none";
    }
  });
  
  // Handle embedded subgraphs
  if (embeddedSubgraphList) {
    const embeddedButtons = embeddedSubgraphList.querySelectorAll("button");
    let hasVisibleEmbedded = false;
    embeddedButtons.forEach(btn => {
      const text = btn.textContent.toLowerCase();
      const matches = text.includes(searchTerm);
      btn.style.display = matches ? "" : "none";
      if (matches) hasVisibleEmbedded = true;
    });
    embeddedSubgraphList.style.display = hasVisibleEmbedded || searchTerm === "" ? "" : "none";
    if (btnNewEmbedded) btnNewEmbedded.style.display = searchTerm === "" ? "" : "none";
  }
  
  // Handle file subgraphs
  if (subgraphList) {
    const subgraphButtons = subgraphList.querySelectorAll("button");
    let hasVisibleSubgraph = false;
    subgraphButtons.forEach(btn => {
      const text = btn.textContent.toLowerCase();
      const matches = text.includes(searchTerm);
      btn.style.display = matches ? "" : "none";
      if (matches) hasVisibleSubgraph = true;
    });
    subgraphList.style.display = hasVisibleSubgraph || searchTerm === "" ? "" : "none";
    if (btnNewSubgraph) btnNewSubgraph.style.display = searchTerm === "" ? "" : "none";
  }
  
  // Show/hide groups based on whether they have visible nodes
  groups.forEach(group => {
    let hasVisibleNodes = false;
    let next = group.nextElementSibling;
    while (next && !next.classList.contains("group")) {
      if ((next.classList.contains("node") && next.style.display !== "none") ||
          (next.id === "embedded-subgraph-list" && next.style.display !== "none") ||
          (next.id === "subgraph-list" && next.style.display !== "none")) {
        hasVisibleNodes = true;
        break;
      }
      next = next.nextElementSibling;
    }
    group.style.display = hasVisibleNodes || searchTerm === "" ? "" : "none";
    // Expand groups when searching
    if (searchTerm !== "") {
      group.classList.remove("collapsed");
    }
  });
});

// Auto Layout
document.getElementById("btn-autolayout").onclick = () => {
    const currentGraph = canvas.graph;
    if (!currentGraph || !currentGraph._nodes || currentGraph._nodes.length === 0) return;

    // Tunable options
    const opts = {
        startX: 60,
        startY: 60,
        xStep: 300,
        yStep: 120,
        wrapAtLevel: 8,
        iterations: 6 // number of barycenter passes
    };

    autoLayoutGraph(currentGraph, opts);
};

// Improved layered layout with barycenter-based crossing minimization
function autoLayoutGraph(currentGraph, opts) {
    opts = opts || {};
    const startX = opts.startX || 60;
    const startY = opts.startY || 60;
    const xStep = opts.xStep || 300;
    const yStep = opts.yStep || 120;
    const wrapAtLevel = opts.wrapAtLevel || 8;
    const iterations = opts.iterations || 4;

    const nodes = currentGraph._nodes.slice();
    if (!nodes || nodes.length === 0) return;

    // Helper maps
    const idToNode = new Map();
    nodes.forEach(n => idToNode.set(n.id, n));

    // Build adjacency and reverse adjacency
    const adj = new Map();
    const revAdj = new Map();
    const inDegree = new Map();

    nodes.forEach(n => { adj.set(n.id, []); revAdj.set(n.id, []); inDegree.set(n.id, 0); });
    Object.values(currentGraph.links).forEach(link => {
        if (!link) return;
        const o = link.origin_id; const t = link.target_id;
        if (adj.has(o) && revAdj.has(t)) {
            adj.get(o).push(t);
            revAdj.get(t).push(o);
            inDegree.set(t, (inDegree.get(t) || 0) + 1);
        }
    });

    // Kahn-like layering
    const queue = [];
    const levelOf = new Map();
    nodes.forEach(n => { if ((inDegree.get(n.id) || 0) === 0) { queue.push(n.id); levelOf.set(n.id, 0); } });

    let maxLevel = 0;
    while (queue.length > 0) {
        const u = queue.shift();
        const lvl = levelOf.get(u) || 0;
        maxLevel = Math.max(maxLevel, lvl);
        (adj.get(u) || []).forEach(v => {
            inDegree.set(v, inDegree.get(v) - 1);
            if (inDegree.get(v) === 0) {
                levelOf.set(v, lvl + 1);
                queue.push(v);
            }
        });
    }

    // Nodes still without level (cycles/disconnected) -> push them toward the end
    let extraLevel = maxLevel + 1;
    nodes.forEach(n => {
        if (!levelOf.has(n.id)) {
            levelOf.set(n.id, extraLevel++);
        }
    });

    // Build level buckets
    const buckets = [];
    levelOf.forEach((lvl, id) => {
        buckets[lvl] = buckets[lvl] || [];
        const nd = idToNode.get(id);
        if (nd) buckets[lvl].push(nd);
    });

    const L = buckets.length;

    // Initialize ordering in each level using current y position (stable) or id
    for (let i = 0; i < L; i++) {
        if (!buckets[i]) buckets[i] = [];
        buckets[i].sort((a,b) => (a.pos && b.pos) ? (a.pos[1] - b.pos[1]) : (a.id - b.id));
    }

    // Barycenter crossing-reduction iterations
    for (let it = 0; it < iterations; it++) {
        // top -> bottom
        for (let lvl = 1; lvl < L; lvl++) {
            const arr = buckets[lvl];
            const prev = buckets[lvl - 1] || [];
            const indexOfPrev = new Map(prev.map((n, idx) => [n.id, idx]));

            const bary = arr.map(n => {
                const parents = revAdj.get(n.id) || [];
                let sum = 0, cnt = 0;
                parents.forEach(p => { if (indexOfPrev.has(p)) { sum += indexOfPrev.get(p); cnt++; } });
                return { n, key: cnt ? (sum / cnt) : Infinity };
            });

            bary.sort((a,b) => a.key - b.key);
            buckets[lvl] = bary.map(x => x.n);
        }

        // bottom -> top
        for (let lvl = L - 2; lvl >= 0; lvl--) {
            const arr = buckets[lvl];
            const next = buckets[lvl + 1] || [];
            const indexOfNext = new Map(next.map((n, idx) => [n.id, idx]));

            const bary = arr.map(n => {
                const children = adj.get(n.id) || [];
                let sum = 0, cnt = 0;
                children.forEach(c => { if (indexOfNext.has(c)) { sum += indexOfNext.get(c); cnt++; } });
                return { n, key: cnt ? (sum / cnt) : Infinity };
            });

            bary.sort((a,b) => a.key - b.key);
            buckets[lvl] = bary.map(x => x.n);
        }
    }

    // Prepare block wrapping if graph is deep
    const wrapAt = wrapAtLevel;
    const blocks = {};
    for (let lvl = 0; lvl < buckets.length; lvl++) {
        const blockRow = Math.floor(lvl / wrapAt);
        blocks[blockRow] = blocks[blockRow] || {levels: [], maxCount: 0};
        blocks[blockRow].levels.push(lvl);
        blocks[blockRow].maxCount = Math.max(blocks[blockRow].maxCount, (buckets[lvl] || []).length);
    }

    // Place nodes per level, centering each level within its block row
    Object.keys(blocks).forEach(blockKey => {
        const info = blocks[blockKey];
        const blockRow = parseInt(blockKey);
        const blockHeight = Math.max(1, info.maxCount) * yStep + yStep; // room for padding
        info.levels.forEach(lvl => {
            const col = lvl % wrapAt;
            const baseX = startX + col * xStep;
            const baseY = startY + blockRow * (blockHeight + 80); // gap between block rows

            const list = buckets[lvl] || [];
            // center list vertically inside the block height
            const totalHeight = (list.length - 1) * yStep;
            const offsetY = baseY + Math.max(0, Math.floor((blockHeight - totalHeight) / 2));

            list.forEach((n, idx) => {
                // spread slightly horizontally within level based on index to avoid perfect overlap
                const jitterX = Math.floor((idx - (list.length - 1) / 2) * Math.min(20, xStep * 0.08));
                n.pos = n.pos || [0, 0];
                n.pos[0] = baseX + jitterX;
                n.pos[1] = offsetY + idx * yStep;
            });
        });
    });

    currentGraph.setDirtyCanvas(true, true);
}

// File operations
document.getElementById("btn-new").onclick = () => {
  if (!isSaved && graph._nodes.length > 0) {
    if (!confirm("当前文件未保存，确定要新建吗？")) return;
  }
  
  graph.clear();
  canvas.draw(true,true);
  currentFileName = null;
  isSaved = false;
  updateFileStatus();
};

document.getElementById("btn-open").onclick = () => {
  document.getElementById("open-file-input").click();
};

document.getElementById("open-file-input").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const json = JSON.parse(event.target.result);
      graph.configure(json);
      extractEmbeddedSubgraphs(json);
      // Reset view to root
      graphStack.length = 0;
      canvas.setGraph(graph);
      updateBreadcrumbs();
      canvas.draw(true,true);
      
      currentFileName = file.name;
      isSaved = true;
      updateFileStatus();
    } catch (err) {
      console.error(err);
      alert("加载失败: " + err);
    }
  };
  reader.readAsText(file);
  // Reset input
  document.getElementById("open-file-input").value = "";
};

document.getElementById("btn-save").onclick = () => {
  if (!currentFileName) {
    // 如果没有文件名，执行另存为
    document.getElementById("btn-save-as").onclick();
    return;
  }
  
  const data = graph.serialize();
  downloadJSON(data, currentFileName);
  isSaved = true;
  updateFileStatus();
};

document.getElementById("btn-save-as").onclick = () => {
  const filename = prompt("请输入文件名:", currentFileName || "my_graph.json");
  if (!filename) return;
  
  if (!filename.endsWith('.json')) {
    filename += '.json';
  }
  
  const data = graph.serialize();
  downloadJSON(data, filename);
  currentFileName = filename;
  isSaved = true;
  updateFileStatus();
};

// Run
const modal = document.getElementById("run-modal");
const modalStatus = document.getElementById("modal-status-text");
const modalLog = document.getElementById("modal-log");
const spanClose = document.getElementsByClassName("close")[0];

spanClose.onclick = function() {
  modal.style.display = "none";
}
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

document.getElementById("btn-run").onclick = async () => {
  if (!isSaved) {
    alert("请先保存文件后再运行训练。");
    return;
  }
  
  const data = {
    filename: currentFileName,
    graph: graph.serialize()
  };
  modal.style.display = "block";
  modalStatus.textContent = "正在启动训练...";
  logHistory = [];
  updateLogDisplay();
  
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const json = await res.json();
    if (json.ok) {
        modalStatus.textContent = "训练已在后台启动，请等待日志更新...";
        socket.emit('request_logs'); // Request log history
    } else {
        modalStatus.textContent = "启动失败: " + (json.error || "未知错误");
        addLogEntry({
            timestamp: new Date().toLocaleString(),
            level: 'ERROR',
            message: json.error || "未知错误",
            module: 'frontend'
        });
    }
  } catch (e) {
    modalStatus.textContent = "启动失败: " + e;
    addLogEntry({
        timestamp: new Date().toLocaleString(),
        level: 'ERROR',
        message: e.toString(),
        module: 'frontend'
    });
  }
};

// Generate script
document.getElementById("btn-gen").onclick = async () => {
  const data = {
    filename: currentFileName,
    graph: graph.serialize()
  };
  const res = await fetch("/api/generate_script", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(data) });
  const j = await res.json();
  if (j.ok) {
    // Download the script
    const blob = new Blob([j.code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = j.filename;
    a.click();
    URL.revokeObjectURL(url);
    alert("脚本已生成并下载: " + j.filename);
  } else {
    alert("生成脚本失败: " + j.error);
  }
};


// Status polling
const trainingStatusBar = document.getElementById("training-status-bar");
const btnStopTraining = document.getElementById("btn-stop-training");
let trainingChart = null;

// Initialize Chart
function initChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx) return;
    
    // Destroy existing chart if any
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Loss',
                    data: [],
                    borderColor: '#ff6384',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    yAxisID: 'y',
                    tension: 0.1
                },
                {
                    label: 'Val Acc',
                    data: [],
                    borderColor: '#36a2eb',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    yAxisID: 'y1',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#ccc' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#888' },
                    grid: { color: '#333' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    ticks: { color: '#ff6384' },
                    grid: { color: '#333' },
                    title: { display: true, text: 'Loss', color: '#ff6384' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: { color: '#36a2eb' },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Accuracy', color: '#36a2eb' },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// Stop training handler
if (btnStopTraining) {
    btnStopTraining.onclick = async () => {
        if (confirm("确定要停止训练吗？")) {
            try {
                const res = await fetch("/api/stop", { method: "POST" });
                const json = await res.json();
                if (json.ok) {
                    addLogEntry({
                        timestamp: new Date().toLocaleString(),
                        level: 'INFO',
                        message: "正在停止训练...",
                        module: 'frontend'
                    });
                }
            } catch (e) {
                console.error("停止训练失败:", e);
            }
        }
    };
}

setInterval(async () => {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    
    // Re-fetch elements to ensure we have the current DOM nodes
    const statusBar = document.getElementById("training-status-bar");
    const stopBtn = document.getElementById("btn-stop-training");
    const chartsContainer = document.getElementById("charts-container");
    
    // Update status bar
    if (statusBar) {
        if (s.running) {
            statusBar.textContent = `Epoch ${s.epoch}/${s.total_epochs} | Loss: ${s.loss.toFixed(4)} | Acc: ${(s.val_acc*100).toFixed(1)}%`;
            statusBar.style.color = "#4CAF50"; // Greenish for running
        } else {
            if (s.epoch > 0) {
                statusBar.textContent = `完成 | Epoch ${s.epoch} | Acc: ${(s.val_acc*100).toFixed(1)}%`;
                statusBar.style.color = "#9fb1ff";
            } else {
                statusBar.textContent = "准备就绪";
                statusBar.style.color = "#888";
            }
        }
    }
    
    // Enable/disable stop button based on running state
    if (stopBtn) {
        if (s.running) {
            stopBtn.disabled = false;
            stopBtn.style.background = "#d32f2f";
            stopBtn.style.color = "white";
            stopBtn.style.cursor = "pointer";
        } else {
            stopBtn.disabled = true;
            stopBtn.style.background = "#444";
            stopBtn.style.color = "#aaa";
            stopBtn.style.cursor = "not-allowed";
        }
    }
    
    // Update charts
    if (s.history && s.history.length > 0) {
        const chartsSection = document.getElementById("charts-section");
        if (chartsSection && !chartsSection.classList.contains('expanded')) {
            // Auto-expand charts section when data becomes available
            chartsSection.classList.add('expanded');
            
            if (trainingChart) {
                setTimeout(() => {
                    trainingChart.resize();
                }, 100);
            }
        }
        
        if (trainingChart) {
            const labels = s.history.map(h => h.epoch);
            const lossData = s.history.map(h => h.loss);
            const accData = s.history.map(h => h.val_acc);
            
            trainingChart.data.labels = labels;
            trainingChart.data.datasets[0].data = lossData;
            trainingChart.data.datasets[1].data = accData;
            trainingChart.update('none'); // Update without animation for performance
        }
        } else if (s.running) {
            const chartsSection = document.getElementById("charts-section");
            if (chartsSection && !chartsSection.classList.contains('expanded')) {
                // Auto-expand charts section when training starts
                chartsSection.classList.add('expanded');
                
                if (trainingChart) {
                    setTimeout(() => {
                    trainingChart.resize();
                }, 100);
            }
        }
    }
    
    // Update modal if open
    if (modal.style.display === "block") {
      if (s.running) {
        modalStatus.textContent = `训练中... Epoch ${s.epoch}/${s.total_epochs} | Loss: ${s.loss.toFixed(4)} | Val Acc: ${(s.val_acc*100).toFixed(2)}%`;
      } else if (s.epoch > 0 && s.epoch === s.total_epochs) {
        modalStatus.textContent = `训练完成! 最终验证准确率: ${(s.val_acc*100).toFixed(2)}%`;
      }
    }
  } catch(e) { console.error(e); }
}, 1500);

graph.start();

// --- Subgraph Reference Implementation ---

function SubgraphRefNode() {
    this.addProperty("subgraph_name", "");
    this.addProperty("subgraph_path", "");
    this.size = [140, 60];
    this.serialize_widgets = true;
}

SubgraphRefNode.title = "Subgraph Ref";

SubgraphRefNode.prototype.onPropertyChanged = function(name, value) {
    if (name == "subgraph_name") {
        this.properties.subgraph_path = "subgraphs/" + value;
        this.loadDefinition(value);
    }
}

SubgraphRefNode.prototype.loadDefinition = function(name) {
    var that = this;
    fetch("/api/subgraphs/" + name)
        .then(r => {
            if(!r.ok) throw new Error("Not found");
            return r.json();
        })
        .then(data => {
            that.updateSlotsFromData(data);
        })
        .catch(e => console.error(e));
}

SubgraphRefNode.prototype.updateSlotsFromData = function(data) {
    var nodes = data.nodes || [];
    var inputNodes = nodes.filter(n => n.type == "graph/input");
    var outputNodes = nodes.filter(n => n.type == "graph/output");
    
    // Sort by pos
    inputNodes.sort((a,b) => a.pos[1] - b.pos[1]);
    outputNodes.sort((a,b) => a.pos[1] - b.pos[1]);
    
    // Sync Inputs
    var currentInputs = this.inputs ? this.inputs.length : 0;
    var neededInputs = inputNodes.length;
    
    for (var i = currentInputs; i < neededInputs; i++) {
        this.addInput("in", "any");
    }
    for (var i = currentInputs; i > neededInputs; i--) {
        this.removeInput(i-1);
    }
    
    for (var i = 0; i < neededInputs; i++) {
        var name = inputNodes[i].properties.name || ("in_" + (i+1));
        if (this.inputs[i]) this.inputs[i].name = name;
    }
    
    // Sync Outputs
    var currentOutputs = this.outputs ? this.outputs.length : 0;
    var neededOutputs = outputNodes.length;
    
    for (var i = currentOutputs; i < neededOutputs; i++) {
        this.addOutput("out", "any");
    }
    for (var i = currentOutputs; i > neededOutputs; i--) {
        this.removeOutput(i-1);
    }
    
    for (var i = 0; i < neededOutputs; i++) {
        var name = outputNodes[i].properties.name || ("out_" + (i+1));
        if (this.outputs[i]) this.outputs[i].name = name;
    }
    
    this.title = this.properties.subgraph_name.replace(".json", "");
}

SubgraphRefNode.prototype.onDblClick = function() {
    var name = this.properties.subgraph_name;
    if(name) {
        openSubgraphEditor(name);
    }
}

SubgraphRefNode.prototype.getExtraMenuOptions = function(canvas, options) {
    var that = this;
    return [
        {
            content: "Reload Definition",
            callback: function() {
                that.loadDefinition(that.properties.subgraph_name);
            }
        }
    ];
}

LiteGraph.registerNodeType("graph/subgraph_ref", SubgraphRefNode);

// --- Subgraph Library UI Logic ---

var currentSubgraphName = null;

function fetchSubgraphs() {
    fetch("/api/subgraphs")
        .then(r => r.json())
        .then(data => {
            renderSubgraphList(data.files);
        });
}

function renderSubgraphList(files) {
    var container = document.getElementById("subgraph-list");
    if(!container) return;
    container.innerHTML = "";
    files.forEach(f => {
        var div = document.createElement("div");
        div.className = "subgraph-item";
        div.style.display = "flex";
        div.style.justifyContent = "space-between";
        div.style.alignItems = "center";
        div.style.padding = "4px 8px";
        div.style.borderBottom = "1px solid #444";
        div.style.cursor = "pointer";
        div.style.fontSize = "12px";
        
        var span = document.createElement("span");
        span.innerText = f.replace(".json", "");
        span.style.cursor = "pointer";
        span.style.color = "#4CAF50";
        span.onmouseover = () => {
            span.style.textDecoration = "underline";
        };
        span.onmouseout = () => {
            span.style.textDecoration = "none";
        };
        span.onclick = (e) => {
            e.stopPropagation();
            addSubgraphRefNode(f);
        };
        
        // 右键点击编辑子图
        span.oncontextmenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            openSubgraphEditor(f);
        };
        
        var btnDelete = document.createElement("button");
        btnDelete.innerText = "×";
        btnDelete.style.marginLeft = "5px";
        btnDelete.style.padding = "0px 4px";
        btnDelete.style.backgroundColor = "#ff4444";
        btnDelete.style.color = "white";
        btnDelete.style.border = "none";
        btnDelete.style.borderRadius = "3px";
        btnDelete.style.cursor = "pointer";
        btnDelete.onclick = (e) => {
            e.stopPropagation();
            deleteSubgraph(f);
        };
        
        div.appendChild(span);
        div.appendChild(btnDelete);
        container.appendChild(div);
    });
}

function addSubgraphRefNode(filename) {
    var node = LiteGraph.createNode("graph/subgraph_ref");
    node.pos = [100, 100];
    node.properties.subgraph_name = filename;
    node.properties.subgraph_path = "subgraphs/" + filename;
    node.loadDefinition(filename);
    canvas.graph.add(node);
}

function deleteSubgraph(filename) {
    if (confirm("确定要删除子图 '" + filename.replace(".json", "") + "' 吗？此操作不可撤销。")) {
        fetch("/api/subgraphs/" + filename, {
            method: "DELETE"
        })
        .then(response => {
            if (response.ok) {
                alert("子图已删除: " + filename.replace(".json", ""));
                fetchSubgraphs(); // 刷新列表
            } else {
                alert("删除失败: " + filename);
            }
        })
        .catch(error => {
            console.error("删除子图失败:", error);
            alert("删除失败: " + error);
        });
    }
}

function openSubgraphEditor(filename) {
    fetch("/api/subgraphs/" + filename)
        .then(r => r.json())
        .then(data => {
            currentSubgraphName = filename;
            
            // 创建一个新的图实例来编辑子图，避免影响主图
            const subgraph = new LGraph();
            canvas.setGraph(subgraph);
            subgraph.configure(data);
            
            var bc = document.getElementById("breadcrumbs");
            bc.style.display = "block";
            bc.innerHTML = ""; // Clear existing
            
            var btnBack = document.createElement("button");
            btnBack.innerText = "← Back to Main";
            btnBack.onclick = loadMainGraph;
            bc.appendChild(btnBack);
            
            var span = document.createElement("span");
            span.innerText = " Editing: " + filename;
            span.style.marginLeft = "10px";
            bc.appendChild(span);
            
            document.querySelectorAll(".subgraph-only").forEach(el => el.style.display = "block");
        })
        .catch(error => {
            console.error("Failed to open subgraph editor:", error);
            alert("无法打开子图编辑器: " + error);
        });
}

async function autoSaveCurrentSubgraph() {
    if (currentSubgraphName) {
        try {
            const currentGraph = canvas.graph;
            const data = currentGraph.serialize();
            await fetch("/api/subgraphs/" + currentSubgraphName, {
                method: "POST",
                body: JSON.stringify(data)
            });
            console.log("子图已自动保存: " + currentSubgraphName);
        } catch (error) {
            console.error("自动保存子图失败:", error);
        }
    }
}

async function loadMainGraph() {
    // 先自动保存当前子图
    await autoSaveCurrentSubgraph();
    
    currentSubgraphName = null;
    document.getElementById("breadcrumbs").style.display = "none";
    document.querySelectorAll(".subgraph-only").forEach(el => el.style.display = "none");
    
    fetch("/api/load_graph")
        .then(r => r.json())
        .then(data => {
            // 确保我们回到主图
            canvas.setGraph(graph);
            graphStack.length = 0;
            updateBreadcrumbs();
            
            if(data.graph) {
                graph.configure(data.graph);
                extractEmbeddedSubgraphs(data.graph);
            }
            // 强制重绘
            graph.setDirtyCanvas(true, true);
        })
        .catch(error => {
            console.error("Failed to load main graph:", error);
            // 即使加载失败，也要确保显示主图
            canvas.setGraph(graph);
            graphStack.length = 0;
            updateBreadcrumbs();
        });
}

// Override Save Button
document.getElementById("btn-save").onclick = async () => {
    // 获取当前活动的图
    var currentGraph = canvas.graph;
    var data = currentGraph.serialize();
    
    if (currentSubgraphName) {
        await fetch("/api/subgraphs/" + currentSubgraphName, {
            method: "POST",
            body: JSON.stringify(data)
        });
        alert("子图已保存: " + currentSubgraphName);
        // Refresh list in case it's new (though we only edit existing ones here)
    } else {
        await fetch("/api/save_graph", { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(data) 
        });
        alert("主图已保存");
    }
};

// New Subgraph Button
document.getElementById("btn-new-subgraph").onclick = function() {
    var name = prompt("Enter subgraph name (e.g. my_block):");
    if(!name) return;
    if(!name.endsWith(".json")) name += ".json";
    
    var newGraph = {
        nodes: [
            {id: 1, type: "graph/input", pos: [100,100], properties: {name: "in"}},
            {id: 2, type: "graph/output", pos: [400,100], properties: {name: "out"}}
        ],
        links: [],
        groups: [],
        config: {}
    };
    
    fetch("/api/subgraphs/" + name, {
        method: "POST",
        body: JSON.stringify(newGraph)
    }).then(() => {
        fetchSubgraphs();
        openSubgraphEditor(name);
    });
};

// 页面关闭前自动保存子图
window.addEventListener('beforeunload', async (event) => {
    if (currentSubgraphName) {
        try {
            const currentGraph = canvas.graph;
            const data = currentGraph.serialize();
            
            // 使用 navigator.sendBeacon 来确保在页面关闭时也能发送请求
            const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
            navigator.sendBeacon('/api/subgraphs/' + currentSubgraphName, blob);
            
            console.log("页面关闭前自动保存子图: " + currentSubgraphName);
        } catch (error) {
            console.error("页面关闭前自动保存子图失败:", error);
        }
    }
});



// Initial Load
fetchSubgraphs();

// Embedded Subgraph Template Management
const embeddedSubgraphTemplates = {}; // 从当前文件加载，不再使用 localStorage

function extractEmbeddedSubgraphs(json) {
    // 清空当前模板
    Object.keys(embeddedSubgraphTemplates).forEach(key => delete embeddedSubgraphTemplates[key]);
    
    if (json.nodes) {
        json.nodes.forEach(node => {
            if (node.type === "graph/subgraph" || node.type === "graph/embedded_subgraph") {
                const sub = node.subgraph || (node.properties && node.properties.subgraph);
                if (sub) {
                    let name = node.template_name || (node.properties && node.properties.template_name);
                    if (!name) {
                        name = node.title || ("Subgraph_" + node.id);
                    }
                    embeddedSubgraphTemplates[name] = sub;
                }
            }
        });
    }
    
    // 重新渲染列表
    renderEmbeddedSubgraphList();
}

function renderEmbeddedSubgraphList() {
    const container = document.getElementById("embedded-subgraph-list");
    if (!container) return;
    container.innerHTML = "";
    
    Object.keys(embeddedSubgraphTemplates).forEach(name => {
        const div = document.createElement("div");
        div.className = "embedded-subgraph-item";
        div.style.display = "flex";
        div.style.justifyContent = "space-between";
        div.style.alignItems = "center";
        div.style.padding = "4px 8px";
        div.style.borderBottom = "1px solid #444";
        div.style.cursor = "pointer";
        div.style.fontSize = "12px";
        
        const span = document.createElement("span");
        span.innerText = name;
        span.style.cursor = "pointer";
        span.style.color = "#4CAF50";
        span.onmouseover = () => {
            span.style.textDecoration = "underline";
        };
        span.onmouseout = () => {
            span.style.textDecoration = "none";
        };
        span.onclick = (e) => {
            e.stopPropagation();
            addEmbeddedSubgraphNode(name);
        };
        
        // 右键点击编辑模板
        span.oncontextmenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            editEmbeddedTemplate(name);
        };
        
        const btnDelete = document.createElement("button");
        btnDelete.innerText = "×";
        btnDelete.style.marginLeft = "5px";
        btnDelete.style.padding = "0px 4px";
        btnDelete.style.backgroundColor = "#ff4444";
        btnDelete.style.color = "white";
        btnDelete.style.border = "none";
        btnDelete.style.borderRadius = "3px";
        btnDelete.style.cursor = "pointer";
        btnDelete.onclick = (e) => {
            e.stopPropagation();
            deleteEmbeddedSubgraphTemplate(name);
        };
        
        div.appendChild(span);
        div.appendChild(btnDelete);
        container.appendChild(div);
    });
}

let previousGraphForTemplateEdit = null;
let currentEditingTemplateName = null;

function editEmbeddedTemplate(name) {
    const template = embeddedSubgraphTemplates[name];
    if (!template) return;
    
    // 保存当前图，以便返回
    previousGraphForTemplateEdit = canvas.graph;
    currentEditingTemplateName = name;
    
    // 创建临时图用于编辑模板
    const g = new LGraph();
    g.configure(template);
    canvas.setGraph(g);
    
    // 更新面包屑
    updateBreadcrumbsForTemplateEdit(name);
    
    // 显示仅在子图编辑时可见的元素
    document.querySelectorAll(".subgraph-only").forEach(el => el.style.display = "block");
}

function updateBreadcrumbsForTemplateEdit(name) {
    const el = document.getElementById("breadcrumbs");
    const pathEl = document.getElementById("breadcrumb-path");
    
    el.style.display = "flex";
    pathEl.innerHTML = "";
    
    const btnBack = document.createElement("span");
    btnBack.className = "crumb";
    btnBack.innerText = "Back (Save Template)";
    btnBack.onclick = finishEmbeddedTemplateEdit;
    pathEl.appendChild(btnBack);
    
    const sep = document.createElement("span");
    sep.className = "separator";
    sep.innerText = ">";
    pathEl.appendChild(sep);
    
    const title = document.createElement("span");
    title.className = "current";
    title.innerText = "Editing Template: " + name;
    pathEl.appendChild(title);
}

function finishEmbeddedTemplateEdit() {
    if (!currentEditingTemplateName) return;
    
    // 保存模板
    const data = canvas.graph.serialize();
    embeddedSubgraphTemplates[currentEditingTemplateName] = data;
    renderEmbeddedSubgraphList();
    
    // 恢复之前的图
    if (previousGraphForTemplateEdit) {
        canvas.setGraph(previousGraphForTemplateEdit);
        previousGraphForTemplateEdit = null;
    } else {
        // Fallback to root if something went wrong
        if (graphStack.length > 0) {
            canvas.setGraph(graphStack[graphStack.length-1].graph);
        } else {
            canvas.setGraph(graph); 
        }
    }
    
    currentEditingTemplateName = null;
    updateBreadcrumbs(); // Restore normal breadcrumbs
    
    // 隐藏仅在子图编辑时可见的元素
    document.querySelectorAll(".subgraph-only").forEach(el => el.style.display = "none");
}

function addEmbeddedSubgraphNode(templateName) {
    const node = LiteGraph.createNode("graph/embedded_subgraph");
    node.pos = [100, 100];
    node.properties.template_name = templateName;
    
    // Load template data
    const templateData = embeddedSubgraphTemplates[templateName];
    if (templateData) {
        node.onConfigure({ subgraph: templateData, template_name: templateName });
    }
    
    canvas.graph.add(node);
}

function deleteEmbeddedSubgraphTemplate(name) {
    if (confirm("确定要删除嵌入式子图模板 '" + name + "' 吗？此操作不可撤销。")) {
        delete embeddedSubgraphTemplates[name];
        // 不再保存到 localStorage
        renderEmbeddedSubgraphList();
    }
}

function saveEmbeddedSubgraphTemplate(name, subgraphData) {
    embeddedSubgraphTemplates[name] = subgraphData;
    // 不再保存到 localStorage
    renderEmbeddedSubgraphList();
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    renderEmbeddedSubgraphList();
    
    document.getElementById('btn-new-embedded-subgraph').addEventListener('click', () => {
        const name = prompt("输入嵌入式子图模板名称:");
        if (name && name.trim()) {
            if (embeddedSubgraphTemplates[name]) {
                alert("模板名称已存在，请选择其他名称。");
                return;
            }
            
            // Create a new embedded subgraph node and immediately enter edit mode
            const node = LiteGraph.createNode("graph/embedded_subgraph");
            node.pos = [100, 100];
            node.properties.template_name = name;
            node.updateTitle();
            canvas.graph.add(node);
            
            // Enter editing mode
            node.onDblClick(null, null, canvas);
        }
    });
    
    // Palette folding
    document.querySelectorAll(".palette .group").forEach(group => {
        group.addEventListener("click", () => {
            const searchInput = document.getElementById("node-search");
            const isSearching = searchInput && searchInput.value.trim() !== "";
            
            // Don't allow collapsing during search
            if (isSearching) return;
            
            group.classList.toggle("collapsed");
            let next = group.nextElementSibling;
            while(next && !next.classList.contains("group")) {
                // Hide/show node buttons
                if(next.classList.contains("node")) {
                    next.classList.toggle("hidden");
                }
                // Hide/show embedded subgraph list and button
                if(next.id === "embedded-subgraph-list" || next.id === "btn-new-embedded-subgraph") {
                    next.style.display = next.style.display === "none" ? "" : "none";
                }
                // Hide/show file subgraph list and button
                if(next.id === "subgraph-list" || next.id === "btn-new-subgraph") {
                    next.style.display = next.style.display === "none" ? "" : "none";
                }
                next = next.nextElementSibling;
            }
        });
    });
});

// Modify EmbeddedSubgraphNode to save templates when exiting edit mode
const originalOnDblClick = EmbeddedSubgraphNode.prototype.onDblClick;
EmbeddedSubgraphNode.prototype.onDblClick = function(e, pos, graphCanvas) {
    // If we have a template name, save current state before entering edit mode
    if (this.properties.template_name) {
        const currentData = this.subgraph.serialize();
        saveEmbeddedSubgraphTemplate(this.properties.template_name, currentData);
    }
    
    // Call original method
    originalOnDblClick.call(this, e, pos, graphCanvas);
};

// Override the exit editing functionality to save template
// We need to find where editing is exited and add save functionality
// This might be in the breadcrumb navigation or escape key handling

// Let's add a method to save template
EmbeddedSubgraphNode.prototype.saveTemplate = function() {
    if (this.properties.template_name) {
        const data = this.subgraph.serialize();
        saveEmbeddedSubgraphTemplate(this.properties.template_name, data);
    }
};

// Handle ESC key to exit subgraph editing
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && graphStack.length > 0) {
        // Save template if we're exiting an embedded subgraph
        const currentNode = graphStack[graphStack.length - 1].node;
        if (currentNode && currentNode.constructor.name === 'EmbeddedSubgraphNode') {
            currentNode.saveTemplate();
        }
        
        // Exit to parent graph
        if (graphStack.length === 1) {
            gotoRoot();
        } else {
            gotoLevel(graphStack.length - 2);
        }
    }
});

// Enhanced Panel toggle functionality
function setupPanelToggles() {
    const headers = document.querySelectorAll('.panel-header');
    const inspectorSection = document.getElementById('inspector-section');
    const chartsSection = document.getElementById('charts-section');
    const logsSection = document.getElementById('logs-section');
    
    // Set initial states
    inspectorSection.classList.add('expanded');
    chartsSection.classList.remove('expanded');
    logsSection.classList.remove('expanded');
    
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const section = header.parentElement;
            const icon = header.querySelector('.toggle-icon');
            const wasExpanded = section.classList.contains('expanded');
            
            // Toggle current panel
            section.classList.toggle('expanded');
            
            // Update icon
            if (section.classList.contains('expanded')) {
                header.classList.add('active');
            } else {
                header.classList.remove('active');
            }
            
            // Handle chart resize when charts panel is expanded
            if (section.id === 'charts-section' && section.classList.contains('expanded')) {
                if (trainingChart) {
                    setTimeout(() => {
                        trainingChart.resize();
                    }, 150); // Slightly longer delay for smooth animation
                }
            }
            
        });
    });
}

// Smooth panel collapse animation
function collapsePanel(section) {
    const icon = section.querySelector('.toggle-icon');
    const header = section.querySelector('.panel-header');
    
    section.classList.remove('expanded');
    header.classList.remove('active');
}

// Panel priority indicators and tooltips
function addPanelTooltips() {
    const headers = document.querySelectorAll('.panel-header');
    
    headers.forEach(header => {
        const section = header.parentElement;
        let tooltip = '';
        
        switch(section.id) {
            case 'inspector-section':
                tooltip = '属性面板 - 编辑选中节点的属性';
                break;
            case 'charts-section':
                tooltip = '训练图表 - 查看训练过程中的损失和准确率';
                break;
            case 'logs-section':
                tooltip = '训练日志 - 实时查看训练日志和状态';
                break;
        }
        
        header.title = tooltip;
    });
}

// Keyboard shortcuts for panel toggles
function setupPanelKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + 1: Toggle Inspector
        if ((e.ctrlKey || e.metaKey) && e.key === '1') {
            e.preventDefault();
            togglePanel('inspector-section');
        }
        // Ctrl/Cmd + 2: Toggle Charts
        else if ((e.ctrlKey || e.metaKey) && e.key === '2') {
            e.preventDefault();
            togglePanel('charts-section');
        }
        // Ctrl/Cmd + 3: Toggle Logs
        else if ((e.ctrlKey || e.metaKey) && e.key === '3') {
            e.preventDefault();
            togglePanel('logs-section');
        }
        // Ctrl/Cmd + 0: Reset to default (only inspector expanded)
        else if ((e.ctrlKey || e.metaKey) && e.key === '0') {
            e.preventDefault();
            resetPanelsToDefault();
        }
    });
}

function togglePanel(panelId) {
    const section = document.getElementById(panelId);
    if (section) {
        const header = section.querySelector('.panel-header');
        header.click();
    }
}

function resetPanelsToDefault() {
    const inspectorSection = document.getElementById('inspector-section');
    const chartsSection = document.getElementById('charts-section');
    const logsSection = document.getElementById('logs-section');
    
    // Expand only inspector
    [inspectorSection, chartsSection, logsSection].forEach(section => {
        const icon = section.querySelector('.toggle-icon');
        const header = section.querySelector('.panel-header');
        
        if (section.id === 'inspector-section') {
            section.classList.add('expanded');
            header.classList.add('active');
        } else {
            section.classList.remove('expanded');
            header.classList.remove('active');
        }
    });
}

// Initialize panel system on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupPanelToggles();
    addPanelTooltips();
    setupPanelKeyboardShortcuts();
    
    // Add visual feedback for panel interactions
    const headers = document.querySelectorAll('.panel-header');
    headers.forEach(header => {
        header.addEventListener('mouseenter', () => {
            if (!header.parentElement.classList.contains('expanded')) {
                header.style.transform = 'translateX(2px)';
            }
        });
        
        header.addEventListener('mouseleave', () => {
            header.style.transform = 'translateX(0)';
        });
    });
});

// Initialize file status
updateFileStatus();
