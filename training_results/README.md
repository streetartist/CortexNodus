# Training Results Folder

这个文件夹用于存放训练过程中生成的所有文件，包括：

- **模型文件 (.pt)**: 训练过程中保存的最佳模型权重
- **训练脚本 (.py)**: 自动生成的独立PyTorch训练脚本

## 文件命名规则

文件名基于graph.json中的模型名称生成，格式为：
```
{模型名称}.pt
{模型名称}.py
```

例如：
- `my_cnn_model.pt`
- `my_cnn_model.py`

注意：由于只保存最佳模型，每次训练会覆盖同名的模型文件。

## Flask调试模式配置

为了防止训练过程中文件变化导致Flask自动重启，应用已配置为：

1. **默认情况下**: 禁用调试模式 (`FLASK_DEBUG=0`)
2. **需要调试时**: 设置环境变量 `FLASK_DEBUG=1` 启用调试模式
3. **文件监控**: 在调试模式下，自动忽略 `training_results` 文件夹中的文件变化

## 启动方式

### 生产模式（推荐）
```bash
python app.py
```

### 调试模式
```bash
set FLASK_DEBUG=1
python app.py
```

或者在PowerShell中：
```powershell
$env:FLASK_DEBUG="1"
python app.py
```

## 注意事项

- 训练结果会自动保存到此文件夹
- 每次训练会覆盖同名的模型文件（只保存最佳模型）
- 此文件夹的变化不会触发Flask重启
- 可以安全地删除或重命名此文件夹中的文件
