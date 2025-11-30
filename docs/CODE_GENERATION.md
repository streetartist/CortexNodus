# 代码生成功能说明

## 概述

项目可以生成**纯 PyTorch 训练脚本**，而不是依赖 JSON 解析的程序。生成的代码是独立的、可读的标准 PyTorch 代码。

## 主要改进

### 之前的方式
生成的代码依赖于：
- `ml.designer` 模块的 `build_model_from_plan` 函数
- 嵌入的 JSON plan 数据
- 运行时解析 JSON 来构建模型

```python
# 旧的生成方式（简化示例）
from ml.designer import build_model_from_plan
plan = {...}  # 大段 JSON 数据
model = build_model_from_plan(plan, in_channels=1, num_classes=10)
```

### 现在的方式
生成标准的 PyTorch 代码：
- 直接定义 `nn.Module` 类
- 使用标准 PyTorch 层（`nn.Conv2d`, `nn.Linear` 等）
- 无需额外依赖，可以独立运行

```python
# 新的生成方式
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_2 = nn.Linear(784, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_6 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_4(x)
        x = F.relu(x)
        x = self.layer_6(x)
        return x
```

## 使用方法

### 1. 通过 Web 界面生成

1. 在 Web 界面中设计你的神经网络
2. 点击 "生成代码" 按钮
3. 代码将保存到 `train_generated.py`

### 3. 编程方式生成

```python
from ml.designer import parse_graph_to_plan
from ml.code_generator import generate_pytorch_script
import json

# 加载图定义
with open('example/dnn_example.json', 'r') as f:
    graph = json.load(f)

# 解析为 plan
plan = parse_graph_to_plan(graph)

# 生成 PyTorch 代码
code = generate_pytorch_script(plan)

# 保存
with open('my_model.py', 'w') as f:
    f.write(code)
```

## 生成的代码结构

生成的脚本包含以下部分：

1. **导入语句**：标准 PyTorch 和 torchvision 导入
2. **模型类**：定义神经网络架构
3. **数据加载函数**：设置数据集和数据加载器
4. **训练函数**：单个 epoch 的训练逻辑
5. **验证函数**：模型评估逻辑
6. **主函数**：组织训练循环

## 支持的层类型

当前支持以下层类型的代码生成：

### 卷积层
- `Conv2D` - 2D 卷积层
- `MaxPool` - 最大池化
- `AvgPool` - 平均池化
- `AdaptiveAvgPool` - 自适应平均池化

### 全连接层
- `Linear` / `Dense` - 全连接层
- `Flatten` - 展平层

### 激活函数
- `ReLU` - ReLU 激活
- `LeakyReLU` - Leaky ReLU
- `Sigmoid` - Sigmoid 激活
- `Tanh` - Tanh 激活
- `Softmax` - Softmax 激活

### 正则化
- `BatchNorm2d` - 批归一化
- `Dropout` - Dropout
- `Dropout2d` - 2D Dropout

### 其他
- `Identity` - 恒等层

## 特性

### 自动形状推断
代码生成器会自动追踪张量形状：
- 卷积层后的空间维度变化
- 池化层的输出大小
- Flatten 后的特征数量
- 全连接层的输入/输出维度

### 智能 Flatten 处理
当遇到第一个全连接层时，如果输入还是 2D/3D 特征图，默认会自动插入 flatten 操作。

### 优化器和损失函数支持
- 优化器：Adam, SGD, AdamW
- 损失函数：CrossEntropy, MSE, SmoothL1

## 示例

### DNN 示例（MNIST）

使用 `example/dnn_example.json` 生成的代码可以实现：
- 输入：28×28 MNIST 图像
- 架构：Flatten → Dense(784→128) → ReLU → Dense(128→64) → ReLU → Dense(64→10)
- 训练 5 epochs，达到 ~97.8% 验证准确率

```bash
python test_code_gen.py
python train_generated.py
```

### CNN 示例（CIFAR-10）

可以加载 CNN 示例图并生成卷积网络代码：
- 自动处理 3 通道输入（RGB）
- 正确计算卷积和池化后的特征图大小
- 在全连接层前自动 flatten

## 技术细节

### 代码生成器架构

`ml/code_generator.py` 包含：
- `generate_pytorch_script()` - 主入口函数
- `_generate_imports()` - 生成导入语句
- `_generate_model_class()` - 生成模型类定义
- `_generate_layer_code()` - 生成单个层的代码
- `_generate_dataset_code()` - 生成数据加载代码
- `_generate_training_code()` - 生成训练循环代码

### 形状追踪

代码生成器维护以下状态：
- `current_channels` - 当前特征通道数
- `current_spatial_size` - 当前空间维度（H/W）
- `is_flattened` - 是否已经 flatten

每个层会更新这些状态，确保下一层能正确连接。

## 与旧系统的兼容性

- Web 界面的训练功能仍然使用 `ml.designer` 模块（基于 JSON 的动态构建）
- 代码生成功能是独立的，不影响现有训练流程
- 两种方式生成的模型架构相同，只是实现方式不同

## 未来改进

计划支持的功能：
- [ ] 更多层类型（Transformer、注意力机制等）
- [ ] 自定义层的代码生成
- [ ] 子图（Subgraph）的代码生成
- [ ] 多输入/多输出模型
- [ ] 学习率调度器
- [ ] 数据增强代码生成
- [ ] 模型可视化代码

## 故障排除

### 生成的代码无法运行

1. 检查是否所有依赖已安装：
   ```bash
   pip install torch torchvision
   ```

2. 确保数据集路径正确（默认：`./data`）

3. 检查模型架构是否合理（例如，通道数匹配）

### 维度不匹配错误

如果遇到维度不匹配，可能是：
- 卷积/池化参数设置不当
- 全连接层输入特征数计算错误

可以手动检查生成的代码并调整层定义。

## 贡献

如果你想为代码生成器添加新的层类型支持，请：
1. 在 `ml/code_generator.py` 的 `_generate_layer_code()` 函数中添加新的 elif 分支
2. 正确更新形状追踪状态
3. 添加测试示例
4. 提交 Pull Request
