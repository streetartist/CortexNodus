# GPT 模型训练指南

本指南介绍如何在 CortexNodus 中构建和训练 GPT 风格的语言模型。

## 快速开始

### 1. 加载示例

最简单的方式是导入预制的 GPT 示例：

1. 启动应用：`python app.py`
2. 访问 http://localhost:5000
3. 点击"上传 JSON"按钮
4. 选择 `gpt_example.json`
5. 点击"运行训练"

### 2. 手动构建 GPT 模型

#### 基本架构

一个典型的 GPT 模型包含以下组件：

```
数据源 → Embedding → 位置编码 → GPT Block × N → LayerNorm → Linear → Loss
```

#### 详细步骤

**步骤 1：选择数据集**
- 从节点面板拖入 `WikiText-2`、`WikiText-103` 或 `Penn Treebank` 节点
- 配置 `batch_size`（建议 16-32）

**步骤 2：添加词嵌入层**
- 拖入 `Embedding` 节点并连接到数据源
- 配置参数：
  - `num_embeddings`: 词汇表大小（自动从数据集推断，或手动设置）
  - `embedding_dim`: 词向量维度（建议 128-768）

**步骤 3：添加位置编码**
- 拖入 `Positional Encoding` 节点
- 配置参数：
  - `max_len`: 最大序列长度（512 或更大）
  - `d_model`: 应与 embedding_dim 相同
  - `dropout`: 0.1（默认）

**步骤 4：堆叠 GPT Block**
- 添加多个 `GPT Block` 节点（建议 3-12 层）
- 每个 GPT Block 包含：
  - 多头自注意力机制（带因果掩码）
  - 前馈神经网络
  - Layer Normalization
  - 残差连接

配置参数：
- `d_model`: 模型维度（128/256/512/768）
- `nhead`: 注意力头数（4/8/12/16，必须能整除 d_model）
- `dim_feedforward`: FFN 维度（通常是 d_model 的 4 倍）
- `dropout`: Dropout 比例（0.1）

**步骤 5：添加最终归一化**
- 拖入 `LayerNorm` 节点

**步骤 6：输出投影**
- 添加 `Linear` 节点
- 配置 `out_features` 为词汇表大小（与 num_embeddings 相同）

**步骤 7：设置训练目标**
- 添加 `Loss` 节点（Training Goal）
- 配置：
  - `kind`: CrossEntropy
  - `optimizer`: Adam
  - `lr`: 0.001（可根据模型大小调整）
  - `epochs`: 20-100
  - `target`: Label（预测下一个词）

## 模型配置建议

### 小型模型（快速实验）
```
embedding_dim: 128
num_layers: 3
nhead: 4
dim_feedforward: 512
batch_size: 20
lr: 0.001
```

### 中型模型（平衡性能）
```
embedding_dim: 256
num_layers: 6
nhead: 8
dim_feedforward: 1024
batch_size: 16
lr: 0.0005
```

### 大型模型（更好性能，需要更多资源）
```
embedding_dim: 512
num_layers: 12
nhead: 8
dim_feedforward: 2048
batch_size: 8
lr: 0.0003
```

## Transformer 组件说明

### GPT Block

GPT Block 是专门为自回归语言建模设计的 Transformer 解码器层：

**特点：**
- 使用因果（causal）自注意力掩码，确保只能看到之前的词
- Pre-Layer Normalization 架构（类似 GPT-2）
- 包含残差连接

**内部结构：**
```python
x = x + MultiheadAttention(LayerNorm(x), causal_mask=True)
x = x + FFN(LayerNorm(x))
```

### 其他 Transformer 组件

**TransformerEncoder**
- 用于双向编码（如 BERT）
- 不使用因果掩码
- 适合文本分类、NER 等任务

**TransformerDecoder**
- 标准 Transformer 解码器
- 需要编码器输出作为输入
- 适合序列到序列任务

**MultiheadAttention**
- 独立的多头注意力层
- 可用于构建自定义架构

**PositionalEncoding**
- 添加位置信息
- 使用正弦/余弦编码
- 对序列模型至关重要

## 训练技巧

### 1. 学习率调整

- 小模型：0.001
- 中型模型：0.0005
- 大模型：0.0001-0.0003

建议使用学习率预热（warmup）和衰减，但当前版本需要手动实现。

### 2. 批次大小

根据 GPU 内存调整：
- 16GB GPU: batch_size = 16-32
- 8GB GPU: batch_size = 8-16
- CPU: batch_size = 4-8

### 3. 序列长度

- 较短序列（32-64）：更快训练，适合快速实验
- 中等序列（128-256）：平衡性能
- 长序列（512-1024）：更好上下文，但需要更多内存

### 4. 正则化

- Dropout: 0.1-0.2（防止过拟合）
- 梯度裁剪：建议在 `app.py` 中手动添加

### 5. 监控指标

观察训练日志中的：
- 训练损失（应该逐渐下降）
- 困惑度（Perplexity = exp(loss)）
- 验证损失（检查过拟合）

## 常见问题

### Q: 训练很慢怎么办？

A: 
- 减少 `batch_size`
- 减少模型层数
- 减少 `d_model` 和 `dim_feedforward`
- 使用 GPU（如果可用）

### Q: 模型不收敛？

A:
- 降低学习率
- 增加训练轮数
- 检查数据集是否正确加载
- 确保 `num_embeddings` 与词汇表大小匹配

### Q: 内存不足？

A:
- 减少 `batch_size`
- 减少模型大小
- 减少序列长度
- 使用梯度累积（需要修改代码）

### Q: 如何使用自己的文本数据？

A:
1. 使用 `CustomData` 节点
2. 设置 `type` 为 "Text"
3. 提供文本文件路径
4. 配置 `input_shape` 为序列长度

## 高级用法

### 1. 创建 GPT Block 子图

可以将多个 GPT Block 封装为可重用的子图：

1. 创建新的嵌入式子图
2. 添加多个 GPT Block
3. 保存为模板
4. 在主图中重复使用

### 2. 混合架构

尝试组合不同的组件：
- GPT Block + TransformerEncoder
- 自定义注意力模式
- 添加卷积层进行特征提取

### 3. 多任务学习

添加多个 Loss 节点实现多任务学习（需要不同的输出头）。

## 参考资源

- **GPT 论文**: "Improving Language Understanding by Generative Pre-Training"
- **GPT-2**: "Language Models are Unsupervised Multitask Learners"
- **Transformer**: "Attention Is All You Need"

## 示例代码片段

生成的训练脚本将类似于：

```python
# 数据加载
dataset = WikiText2(...)
dataloader = DataLoader(dataset, batch_size=20)

# 模型构建
model = nn.Sequential(
    nn.Embedding(vocab_size, 128),
    PositionalEncoding(128, 512),
    GPTBlock(128, 4, 512),
    GPTBlock(128, 4, 512),
    GPTBlock(128, 4, 512),
    nn.LayerNorm(128),
    nn.Linear(128, vocab_size)
)

# 训练循环
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

for epoch in range(20):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 下一步

- 尝试不同的超参数配置
- 在更大的数据集上训练
- 实现文本生成功能
- 探索其他 NLP 任务（分类、问答等）

---

如有问题，请参考项目 README 或提交 Issue。
