import os
import json
import threading
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

# 延迟导入以加快初次启动
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import collections

from ml.designer import parse_graph_to_plan, build_model_from_plan, build_optim_loss_from_plan
from ml.code_generator import generate_pytorch_script

def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """Simple progress bar for console output"""
    percent = current / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1%} {suffix}', end='', flush=True)
    if current == total:
        print()  # New line when complete

def load_wikitext_dataset(data_dir="./data/wikitext-2", seq_length=35):
    """
    Load WikiText-2 dataset for language modeling.
    Returns train_ds, val_ds, test_ds, vocab_size
    """
    def read_tokens(file_path):
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
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    train_seq, train_tgt = create_sequences(train_ids, seq_length)
    val_seq, val_tgt = create_sequences(val_ids, seq_length)
    test_seq, test_tgt = create_sequences(test_ids, seq_length)
    
    train_ds = TensorDataset(train_seq, train_tgt)
    val_ds = TensorDataset(val_seq, val_tgt)
    test_ds = TensorDataset(test_seq, test_tgt)
    
    return train_ds, val_ds, test_ds, vocab_size


app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# 日志队列和处理器
log_queue = Queue()

class SocketIOLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': self.format(record),
            'module': record.name
        }
        log_queue.put(log_entry)
        socketio.emit('log', log_entry)

# 设置日志
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)
handler = SocketIOLogHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.addHandler(console_handler)


# 简易内存存储
GRAPH_STORE_PATH = os.path.join(os.getcwd(), "graph.json")


@dataclass
class TrainState:
    running: bool = False
    stop_requested: bool = False
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    val_acc: float = 0.0
    best_val_acc: float = 0.0
    best_val_loss: float = 1e9
    best_model_path: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


state = TrainState()
state_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/save_graph", methods=["POST"])
def save_graph():
    # 客户端操作，不再需要服务器保存
    return jsonify({"ok": True})


@app.route("/api/load_graph", methods=["GET"]) 
def load_graph():
    # 客户端操作，不再需要服务器加载
    return jsonify({"graph": None})


def get_model_filename_from_graph(filename=None):
    """从graph.json文件生成模型文件名"""
    try:
        if filename:
            # 如果提供了文件名，使用它
            base_name = filename.replace('.json', '')
        else:
            # 默认使用graph.json
            if os.path.exists(GRAPH_STORE_PATH):
                with open(GRAPH_STORE_PATH, "r", encoding="utf-8") as f:
                    graph = json.load(f)
                
                # 尝试从graph中获取名称，如果没有则使用默认名称
                graph_name = graph.get("name", "model")
                # 清理文件名，移除不合法字符
                graph_name = "".join(c for c in graph_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                if not graph_name:
                    graph_name = "model"
                base_name = graph_name
            else:
                base_name = "model"
        
        return f"{base_name}.pt"
    except Exception:
        return "model.pt"

def run_training_thread(plan: Dict[str, Any], filename: str = None):
    global state
    
    # 立即设置运行状态
    with state_lock:
        state.running = True
        state.stop_requested = False
        state.epoch = 0
        # total_epochs 稍后更新
        state.best_val_acc = 0.0
        state.best_val_loss = 1e9
        state.best_model_path = None
        state.history = []
        
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 确保training_results文件夹存在
        training_results_dir = os.path.join(os.getcwd(), "training_results")
        if not os.path.exists(training_results_dir):
            os.makedirs(training_results_dir)

        # Data
        batch_size = plan.get("data", {}).get("batch_size", 64)
        dataset_name = plan.get("data", {}).get("dataset", "MNIST")

        # Check if it's a node-based dataset
        data_nodes = [n for n in plan.get("nodes", []) if n.get("type") in ["MNIST", "Fashion-MNIST", "CIFAR-10", "WikiText-2", "WikiText-103", "PennTreebank", "CustomData"]]
        if data_nodes:
            dataset_name = data_nodes[0]["type"]
            batch_size = data_nodes[0]["properties"].get("batch_size", batch_size)

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
            # Load WikiText-2 dataset
            seq_length = data_nodes[0]["properties"].get("seq_length", 35) if data_nodes else 35
            train_ds, val_ds, test_ds, vocab_size = load_wikitext_dataset(data_dir="./data/wikitext-2", seq_length=seq_length)
            in_channels = seq_length  # sequence length
            num_classes = vocab_size  # vocabulary size
        elif dataset_name == "CustomData":
            data_props = plan.get("data", {})
            data_path = data_props.get("path", "")
            data_type = data_props.get("type", "ImageFolder")

            if data_type in ["WikiText-2", "WikiText-103", "PennTreebank"]:
                # For demo / local runs we avoid depending on torchtext usage differences across versions.
                # Use a small built-in sample text to generate sequences instead of calling torchtext APIs.
                sample_text = """The quick brown fox jumps over the lazy dog. This is a sample text for language modeling. We need sufficient text to create meaningful sequences for training a GPT model. Language models learn patterns in text by predicting the next word given previous words. Transformer architectures have revolutionized natural language processing."""

                tokens = sample_text.lower().split()
                vocab = list(set(tokens))
                vocab.sort()
                word_to_idx = {word: i for i, word in enumerate(vocab)}

                data = [word_to_idx[token] for token in tokens]

                seq_length = int(data_props.get("input_shape", 35))
                sequences = []
                targets = []
                for i in range(len(data) - seq_length):
                    sequences.append(data[i:i+seq_length])
                    targets.append(data[i+1:i+seq_length+1])

                sequences = torch.tensor(sequences, dtype=torch.long)
                targets = torch.tensor(targets, dtype=torch.long)

                full_ds = TensorDataset(sequences, targets)
                in_channels = seq_length
                num_classes = len(vocab)

                total_len = len(full_ds)
                train_len = int(0.8 * total_len)
                val_len = total_len - train_len
                train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

            elif not data_path or not os.path.exists(data_path):
                raise ValueError(f"Custom data path not found: {data_path}")
                
            if data_type == "ImageFolder":
                transform = transforms.Compose([
                    transforms.Resize((128, 128)), 
                    transforms.ToTensor(),
                ])
                full_ds = datasets.ImageFolder(root=data_path, transform=transform)
                in_channels = 3
                num_classes = len(full_ds.classes)
                
            elif data_type == "CSV":
                df = pd.read_csv(data_path)
                # Assume last column is label
                x = df.iloc[:, :-1].values.astype('float32')
                y = df.iloc[:, -1].values.astype('int64')
                
                full_ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
                in_channels = x.shape[1]
                num_classes = len(np.unique(y))
                
            elif data_type == "Numpy":
                data = np.load(data_path)
                if 'x' in data and 'y' in data:
                    x = torch.from_numpy(data['x'].astype('float32'))
                    y = torch.from_numpy(data['y'].astype('int64'))
                    full_ds = TensorDataset(x, y)
                    if len(x.shape) == 4:
                        in_channels = x.shape[1]
                    else:
                        in_channels = x.shape[1]
                    num_classes = len(torch.unique(y))
                else:
                    raise ValueError("Numpy file must contain 'x' and 'y'")
                    
            elif data_type == "Text":
                # Simple text file processing
                with open(data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Simple tokenization (split by spaces)
                tokens = text.split()
                vocab = list(set(tokens))
                vocab.sort()
                word_to_idx = {word: i for i, word in enumerate(vocab)}
                
                # Convert to indices
                data = [word_to_idx[token] for token in tokens]
                data = torch.tensor(data, dtype=torch.long)
                
                seq_length = int(data_props.get("input_shape", "50"))
                
                # Create sequences
                sequences = []
                targets = []
                for i in range(len(data) - seq_length):
                    sequences.append(data[i:i+seq_length])
                    targets.append(data[i+1:i+seq_length+1])
                
                sequences = torch.stack(sequences)
                targets = torch.stack(targets)
                
                full_ds = TensorDataset(sequences, targets)
                in_channels = seq_length
                num_classes = len(vocab)
                
                total_len = len(full_ds)
                train_len = int(0.8 * total_len)
                val_len = total_len - train_len
                train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])
                
            else:
                raise ValueError(f"Unsupported custom data type: {data_type}")
            
            total_len = len(full_ds)
            train_len = int(0.8 * total_len)
            val_len = total_len - train_len
            train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])
        else:
            raise ValueError("Unsupported dataset")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Model
        model = build_model_from_plan(plan, in_channels=in_channels, num_classes=num_classes).to(device)

        # Optimizer & Loss
        # criterion is now a dict: {node_id: loss_fn}
        optimizer, criterions = build_optim_loss_from_plan(plan, model)
        
        # Train Configs
        train_configs = plan.get("train", [])
        if not train_configs:
            raise ValueError("No training configuration found")
            
        # Use the first config for global settings like epochs
        main_config = train_configs[0]
        total_epochs = int(main_config.get("epochs", 3))

        with state_lock:
            state.total_epochs = total_epochs

        logger.info(f"🚀 开始训练 {total_epochs} 个周期，使用 {dataset_name} 数据集")
        for epoch in range(1, total_epochs + 1):
            if state.stop_requested:
                logger.info("🛑 训练已停止")
                break
                
            model.train()
            epoch_loss = 0.0
            total_batches = len(train_loader)
            for batch_idx, (x, y) in enumerate(train_loader):
                if state.stop_requested:
                    break
                    
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                # Forward pass returns dict of outputs
                outputs = model(x)
                
                total_batch_loss = 0.0
                head_losses = {}
                
                # Calculate loss for each head
                for cfg in train_configs:
                    nid = cfg["node_id"]
                    target_type = cfg.get("target", "Label")
                    weight = cfg.get("weight", 1.0)
                    crit = criterions[nid]
                    
                    loss_node_inputs = plan['model']['connections'].get(str(nid), [])
                    if not loss_node_inputs:
                        continue
                        
                    src_id = str(loss_node_inputs[0])
                    pred = outputs.get(src_id)
                    
                    if pred is None:
                        continue
                        
                    if target_type == "Input":
                        target = x
                    else:
                        target = y
                    
                    # Handle sequence outputs (for language modeling)
                    if len(pred.shape) == 3:  # (batch, seq_len, vocab_size)
                        # Reshape for CrossEntropyLoss
                        pred = pred.reshape(-1, pred.size(-1))  # (batch*seq_len, vocab_size)
                        if len(target.shape) == 2:  # (batch, seq_len)
                            target = target.reshape(-1)  # (batch*seq_len,)
                    
                    l = crit(pred, target)
                    total_batch_loss += l * weight
                    head_losses[nid] = l.item()
                
                if isinstance(total_batch_loss, torch.Tensor):
                    total_batch_loss.backward()
                    optimizer.step()
                    epoch_loss += total_batch_loss.item()
                else:
                    # No loss computed (maybe all heads skipped?)
                    pass

                # Update progress bar every 10 batches or at the end
                current_loss = total_batch_loss.item() if isinstance(total_batch_loss, torch.Tensor) else 0.0
                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches
                    filled_length = int(50 * progress)
                    bar = '█' * filled_length + '-' * (50 - filled_length)
                    print(f'\rEpoch {epoch} |{bar}| {progress:.1%} Loss: {current_loss:.4f}', end='', flush=True)
                    if batch_idx == total_batches - 1:
                        print()  # New line when epoch complete

                if batch_idx % 50 == 0:
                    current_loss = total_batch_loss.item() if isinstance(total_batch_loss, torch.Tensor) else 0.0
                    logger.info(f"📊 Epoch {epoch}, Batch {batch_idx}/{total_batches}, Loss: {current_loss:.4f}")
                    
                    # If multi-head, add details
                    if len(train_configs) > 1:
                        details = []
                        for i, cfg in enumerate(train_configs):
                            nid = cfg["node_id"]
                            loss_val = head_losses.get(nid, 0.0)
                            details.append(f"Head{i+1}: {loss_val:.4f}")
                        logger.info(f"   多头损失详情: {', '.join(details)}")

            # Eval (Simplified: just use the first head for accuracy/metric)
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            # Main head config
            main_head_cfg = train_configs[0]
            main_nid = str(main_head_cfg["node_id"])
            main_target_type = main_head_cfg.get("target", "Label")
            main_crit = criterions[main_head_cfg["node_id"]]
            
            main_src_id = str(plan['model']['connections'].get(main_nid, [])[0])
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    pred = outputs.get(main_src_id)
                    
                    if pred is None: continue
                    
                    if main_target_type == "Input":
                        # Handle sequence outputs
                        if len(pred.shape) == 3:
                            pred_flat = pred.reshape(-1, pred.size(-1))
                            x_flat = x.reshape(-1) if len(x.shape) == 2 else x
                            l = main_crit(pred_flat, x_flat)
                        else:
                            l = main_crit(pred, x)
                        val_loss += l.item()
                        total += 1
                    else:
                        # Handle sequence outputs for classification
                        if len(pred.shape) == 3:  # (batch, seq_len, vocab_size)
                            # For language modeling: compute accuracy on all positions
                            pred_flat = pred.reshape(-1, pred.size(-1))  # (batch*seq_len, vocab_size)
                            if len(y.shape) == 2:  # (batch, seq_len)
                                y_flat = y.reshape(-1)  # (batch*seq_len,)
                                p = pred_flat.argmax(dim=1)
                                correct += (p == y_flat).sum().item()
                                total += y_flat.size(0)
                            else:
                                # Fallback for unexpected shapes
                                p = pred.argmax(dim=-1)
                                correct += (p == y).sum().item()
                                total += y.numel()
                        else:
                            p = pred.argmax(dim=1)
                            correct += (p == y).sum().item()
                            total += y.size(0)
            
            if main_target_type == "Input":
                val_metric = val_loss / max(len(test_loader), 1)
                metric_name = "val_loss"
            else:
                val_metric = correct / max(total, 1)
                metric_name = "val_acc"

            with state_lock:
                state.epoch = epoch
                state.loss = float(epoch_loss / max(len(train_loader), 1))
                state.val_acc = float(val_metric)
                state.history.append({
                    "epoch": epoch,
                    "loss": state.loss,
                    "val_acc": state.val_acc
                })
                
                if main_target_type == "Input":
                    logger.info(f"✅ Epoch {epoch} 完成, 验证损失: {val_metric:.4f}")
                else:
                    logger.info(f"✅ Epoch {epoch} 完成, 验证准确率: {val_metric:.4f}")
                
                if main_target_type == "Input":
                    if val_metric < state.best_val_loss:
                        state.best_val_loss = float(val_metric)
                        model_filename = get_model_filename_from_graph(filename)
                        best_path = os.path.join(training_results_dir, model_filename)
                        torch.save(model.state_dict(), best_path)
                        state.best_model_path = best_path
                        logger.info(f"💾 保存最佳模型到 training_results/{model_filename} (验证损失: {val_metric:.4f})")
                elif val_metric > state.best_val_acc:
                    state.best_val_acc = float(val_metric)
                    model_filename = get_model_filename_from_graph(filename)
                    best_path = os.path.join(training_results_dir, model_filename)
                    torch.save(model.state_dict(), best_path)
                    state.best_model_path = best_path
                    logger.info(f"💾 保存最佳模型到 training_results/{model_filename} (验证准确率: {val_metric:.4f})")

        logger.info("🎉 训练完成！")
        
    except Exception as e:
        logger.error(f"❌ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        with state_lock:
            state.running = False


@app.route("/api/stop", methods=["POST"])
def stop_training():
    with state_lock:
        if state.running:
            state.stop_requested = True
            logger.info("🛑 收到停止训练请求...")
    return jsonify({"ok": True})


@app.route("/api/run", methods=["POST"]) 
def run_training():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        graph = data.get("graph", data)  # 兼容旧格式
        plan = parse_graph_to_plan(graph)
        t = threading.Thread(target=run_training_thread, args=(plan, filename,), daemon=True)
        t.start()
        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Internal Error: {str(e)}"}), 500


@socketio.on('connect')
def handle_connect():
    logger.info("📡 前端已连接")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("📡 前端已断开连接")

@socketio.on('request_logs')
def handle_request_logs():
    # 发送最近的日志历史
    logs = []
    while not log_queue.empty() and len(logs) < 100:  # 最多发送100条最近日志
        logs.append(log_queue.get())
    emit('log_history', logs)

@app.route("/api/status", methods=["GET"]) 
def status():
    with state_lock:
        return jsonify(asdict(state))


@app.route("/api/generate_script", methods=["POST"]) 
def generate_script():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        graph = data.get("graph", data)  # 兼容旧格式
        plan = parse_graph_to_plan(graph)
        
        # 生成基于graph名称的脚本文件名
        script_filename = get_model_filename_from_graph(filename).replace('.pt', '.py')

        # Generate standalone PyTorch script
        code = generate_pytorch_script(plan)

        return jsonify({"ok": True, "code": code, "filename": script_filename})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/subgraphs", methods=["GET"])
def list_subgraphs():
    subgraphs_dir = os.path.join(os.getcwd(), "subgraphs")
    if not os.path.exists(subgraphs_dir):
        os.makedirs(subgraphs_dir)
    files = [f for f in os.listdir(subgraphs_dir) if f.endswith(".json")]
    return jsonify({"files": files})

@app.route("/api/subgraphs/<name>", methods=["GET"])
def get_subgraph(name):
    subgraphs_dir = os.path.join(os.getcwd(), "subgraphs")
    path = os.path.join(subgraphs_dir, name)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/api/subgraphs/<name>", methods=["POST"])
def save_subgraph(name):
    subgraphs_dir = os.path.join(os.getcwd(), "subgraphs")
    if not os.path.exists(subgraphs_dir):
        os.makedirs(subgraphs_dir)
    path = os.path.join(subgraphs_dir, name)
    data = request.get_json(force=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True})

@app.route("/api/subgraphs/<name>", methods=["DELETE"])
def delete_subgraph(name):
    subgraphs_dir = os.path.join(os.getcwd(), "subgraphs")
    path = os.path.join(subgraphs_dir, name)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    try:
        os.remove(path)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # 在训练过程中禁用调试模式以避免文件变化导致的自动重启
    # 可以通过环境变量 FLASK_DEBUG=1 来启用调试模式
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    
    if debug_mode:
        # 在调试模式下，忽略training_results文件夹的变化
        import werkzeug.serving
        original_is_changed = werkzeug.serving._is_changed
        
        def patched_is_changed(filename):
            # 忽略training_results文件夹中的文件变化
            if 'training_results' in str(filename):
                return False
            return original_is_changed(filename)
        
        werkzeug.serving._is_changed = patched_is_changed
    
    socketio.run(app, host="0.0.0.0", port=port, debug=debug_mode)
