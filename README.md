# CortexNodus

CortexNodus 是一个基于 Flask + LiteGraph.js 的可视化 AI 训练工作台，可通过拖拽方式构建完整的深度学习流水线（数据 → 模型 → 训练 → 评估），并自动在后端生成 PyTorch 代码并执行训练。目前已内置 MNIST、Fashion-MNIST、CIFAR-10 等常见图片数据集，能够图形化训练一个数字识别模型，也可以扩展为更多同等复杂度的网络。

## 目录结构

- `app.py`：Flask 服务，负责解析画布、构建 PyTorch 模型与训练循环、生成脚本。
- `templates/index.html`：主界面骨架。
- `static/style.css`：暗色主题 UI 样式。
- `static/designer.js`：LiteGraph 画布逻辑、节点注册、Inspector、状态轮询。
- `ml/designer.py`：画布 JSON 解析器，负责将节点拓扑转换为 `DesignerPlan`。
- `docs/UX_PLAN.md`：LabVIEW 风格界面设计方案。

## 快速开始（Windows PowerShell）

```powershell
cd "d:\新建文件夹 (2)"
python -m venv .venv
".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
python app.py
```

启动后访问 `http://localhost:5000` 即可打开画布。

## 使用步骤

1. **拖拽节点**：在左侧库中选择数据源、卷积层、池化层、全连接层、损失函数、优化器以及训练控制器，拖入画布后用连线连接形成顺序拓扑。
2. **配置属性**：选中任意节点，右侧 Inspector 会展示可编辑属性（如卷积通道数、学习率等），修改后实时保存。
3. **运行训练**：点击顶部“运行训练”按钮，服务端会解析当前画布、构建 PyTorch `nn.Sequential` 模型，随后后台线程执行训练。状态卡片、日志面板会实时显示 epoch、损失、验证准确率。
4. **测试/部署**：训练完成后可点击“测试集评估”以及“下载模型”，或选择“生成 PyTorch 脚本”导出独立训练脚本。

## 功能特性

- LiteGraph 画布支持拖拽、连线、属性面板以及本地/服务端双重保存。
- 支持多数据集、动态模型构建、Adam/SGD/AdamW 优化器、CrossEntropy/MSE/SmoothL1 等损失。
- 训练过程在后台线程运行，提供实时状态、日志、最佳权重保存、停止控制。
- **一键生成纯 PyTorch 训练脚本**：生成独立的、可读的标准 PyTorch 代码，无需额外依赖即可运行。
- **全面支持 Transformer 和 GPT 架构**：包括 Embedding、位置编码、多头注意力、Transformer 编码器/解码器、GPT Block 等。

## 支持的模型类型

### 计算机视觉
- CNN（卷积神经网络）：支持各种卷积层、池化层、归一化层
- ResNet、DenseNet 等通过子图支持
- 图像分类、目标检测基础架构

### 自然语言处理
- **Transformer 编码器**：用于文本分类、命名实体识别等
- **Transformer 解码器**：用于文本生成
- **GPT 架构**：通过 GPT Block 构建语言模型
- 支持词嵌入（Embedding）和位置编码（Positional Encoding）

### 数据集支持
- **图像数据集**：MNIST、Fashion-MNIST、CIFAR-10
- **文本数据集**：WikiText-2、WikiText-103、Penn Treebank
- **自定义数据集**：支持 ImageFolder、CSV、Numpy、文本文件

## 示例文件

- `example_graph.json`：基础 MNIST 卷积神经网络示例
- `gpt_example.json`：GPT 语言模型训练示例，包含：
  - Token Embedding 层
  - 位置编码
  - 3 层 GPT Block（带自注意力机制）
  - Layer Normalization
  - 输出投影层
  
## 使用 GPT 示例

```bash
# 导入 gpt_example.json 到界面
# 或手动构建以下结构：
# WikiText-2 → Embedding → Positional Encoding → GPT Block × 3 → LayerNorm → Linear → Loss
```

GPT 模型配置说明：
- **d_model**: 模型维度（默认 128）
- **nhead**: 注意力头数（默认 4-8）
- **dim_feedforward**: 前馈网络维度（默认 512）
- **dropout**: Dropout 比例（默认 0.1）
- **num_embeddings**: 词汇表大小
- **embedding_dim**: 词嵌入维度

## 代码生成

项目现在支持生成**纯 PyTorch 代码**，而不是依赖 JSON 解析的程序：

```bash
# 使用示例生成代码
python test_code_gen.py

# 运行生成的代码
python train_generated.py
```

生成的代码特点：
- ✅ 标准 PyTorch `nn.Module` 类定义
- ✅ 完整的训练和验证循环
- ✅ 无需额外依赖，可独立运行
- ✅ 自动形状推断和维度计算
- ✅ 可读性强，易于理解和修改

详细文档请参考：[CODE_GENERATION.md](docs/CODE_GENERATION.md)

## 后续可扩展点

- 增加更多预训练模型支持（BERT、T5 等）
- 集成 WebSocket 推送，实时绘制训练曲线图
- 支持多人协作/版本管理、超参搜索等高级功能
- 添加模型导出为 ONNX 格式功能
