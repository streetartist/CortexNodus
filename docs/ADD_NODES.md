# 如何添加新节点

CortexNodus 采用注册表模式，允许研究人员轻松扩展新的层和功能。添加新节点分为两步：前端定义与后端实现。

## 1. 前端定义 (`static/designer.js`)

在 `static/designer.js` 文件中，找到 `NODE_DEFINITIONS` 数组。这是一个包含所有节点元数据的列表。

要添加新节点，只需在数组中追加一个新的对象：

```javascript
{
  type: "MyNewLayer",       // 唯一标识符，需与后端匹配
  title: "My New Layer",    // 显示在节点标题栏的名称
  props: {                  // 默认属性
    my_param: 1.0,
    active: true
  },
  in: ["in"],               // 输入端口列表
  out: ["out"]              // 输出端口列表
}
```

添加后，还需要在 `templates/index.html` 的侧边栏（Palette）中添加对应的按钮，以便用户拖拽：

```html
<button class="node" data-type="MyNewLayer">My New Layer</button>
```

## 2. 后端实现 (`ml/designer.py`)

在 `ml/designer.py` 中，使用 `@register_layer` 装饰器注册该节点的构建逻辑。

你需要定义一个函数，接收 `props` (属性字典) 和 `ctx` (上下文对象)，并返回一个 `nn.Module` 实例。

```python
@register_layer("MyNewLayer")
def build_my_new_layer(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    # 1. 获取属性
    param = float(props.get("my_param", 1.0))
    
    # 2. (可选) 使用 ctx 获取输入维度信息
    # ctx.in_channels: 当前输入通道数
    # ctx.spatial_size: 当前特征图空间尺寸 (H/W)
    
    # 3. 创建 PyTorch 模块
    layer = MyCustomModule(param)
    
    # 4. (重要) 更新上下文，以便下一层知道输出维度
    # ctx.out_channels = ...
    # ctx.out_spatial_size = ...
    
    return layer
```

### 上下文对象 (`LayerContext`)

`ctx` 对象用于在层之间传递形状信息，帮助自动推断维度（例如 `Dense` 层自动计算输入特征数）。

- `ctx.in_channels`: 输入通道数。
- `ctx.spatial_size`: 输入空间尺寸。
- `ctx.out_channels`: **必须更新**，输出通道数。
- `ctx.out_spatial_size`: **必须更新**，输出空间尺寸。

## 示例：添加 Dropout 层

**前端 (`static/designer.js`)**:
```javascript
{ type: "Dropout", title: "Dropout", props: { p: 0.5 }, in: ["in"], out: ["out"] }
```

**后端 (`ml/designer.py`)**:
```python
@register_layer("Dropout")
def build_dropout(props: Dict[str, Any], ctx: LayerContext) -> nn.Module:
    p = float(props.get("p", 0.5))
    # Dropout 不改变维度，无需更新 ctx
    return nn.Dropout(p=p)
```
