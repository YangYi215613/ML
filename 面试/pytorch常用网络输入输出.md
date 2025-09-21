下面是 PyTorch 中常用层的输入输出举例说明：

---

#### 1. `nn.Conv2d`
二维卷积层，常用于图像。

- **输入**：`(batch_size, in_channels, height, width)`
- **输出**：`(batch_size, out_channels, out_height, out_width)`

```python
x = torch.randn(8, 3, 224, 224)
conv = nn.Conv2d(3, 16, 3, 1, 1)
y = conv(x)
print(y.shape)  # torch.Size([8, 16, 224, 224])
```

---

#### 2. `nn.Linear`
全连接层，常用于特征向量。

- **输入**：`(batch_size, in_features)`
- **输出**：`(batch_size, out_features)`

```python
x = torch.randn(8, 128)
fc = nn.Linear(128, 64)
y = fc(x)
print(y.shape)  # torch.Size([8, 64])
```

---

#### 3. `nn.MaxPool2d`
二维最大池化，常用于图像降采样。

- **输入**：`(batch_size, channels, height, width)`
- **输出**：`(batch_size, channels, out_height, out_width)`

```python
x = torch.randn(8, 16, 32, 32)
pool = nn.MaxPool2d(2, 2)
y = pool(x)
print(y.shape)  # torch.Size([8, 16, 16, 16])
```

---

#### 4. `nn.BatchNorm2d`
二维批归一化，常用于卷积输出。

- **输入**：`(batch_size, channels, height, width)`
- **输出**：同输入形状

```python
x = torch.randn(8, 16, 32, 32)
bn = nn.BatchNorm2d(16)
y = bn(x)
print(y.shape)  # torch.Size([8, 16, 32, 32])
```

---

#### 5. `nn.ReLU`
激活函数，无参数。

- **输入/输出**：形状不变

```python
x = torch.randn(8, 16, 32, 32)
relu = nn.ReLU()
y = relu(x)
print(y.shape)  # torch.Size([8, 16, 32, 32])
```

---

#### 6. `nn.Flatten`
展平层，常用于卷积到全连接的过渡。

- **输入**：`(batch_size, ...)`
- **输出**：`(batch_size, -1)`

```python
x = torch.randn(8, 16, 7, 7)
flatten = nn.Flatten()
y = flatten(x)
print(y.shape)  # torch.Size([8, 784])
```

------

#### 7. `nn.Conv1d`

一维卷积层，常用于时序数据或语音。

- **输入**：`(batch_size, in_channels, seq_len)`
- **输出**：`(batch_size, out_channels, out_seq_len)`

```python
x = torch.randn(8, 3, 100)   # batch=8, 3通道, 序列长度=100
conv = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
y = conv(x)
print(y.shape)  # torch.Size([8, 16, 100])
```

------

#### 8. `nn.Conv3d`

三维卷积层，常用于视频或医学图像。

- **输入**：`(batch_size, in_channels, depth, height, width)`
- **输出**：`(batch_size, out_channels, out_depth, out_height, out_width)`

```python
x = torch.randn(4, 1, 16, 64, 64)  # batch=4, 通道=1, 16帧, 64x64
conv = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
y = conv(x)
print(y.shape)  # torch.Size([4, 8, 16, 64, 64])
```

------

#### 9. `nn.AvgPool2d`

二维平均池化。

- **输入**：`(batch_size, channels, height, width)`
- **输出**：降采样后的形状

```python
x = torch.randn(8, 16, 32, 32)
pool = nn.AvgPool2d(2, 2)
y = pool(x)
print(y.shape)  # torch.Size([8, 16, 16, 16])
```

------

#### 10. `nn.Dropout`

随机丢弃，用于防止过拟合。

- **输入/输出**：形状不变

```python
x = torch.randn(8, 128)
drop = nn.Dropout(p=0.5)
y = drop(x)
print(y.shape)  # torch.Size([8, 128])
```

------

#### 11. `nn.Embedding`

词嵌入层，常用于 NLP。

- **输入**：`(batch_size, seq_len)`（索引）
- **输出**：`(batch_size, seq_len, embedding_dim)`

```python
x = torch.randint(0, 1000, (4, 10))  # batch=4, 序列长度=10
embed = nn.Embedding(num_embeddings=1000, embedding_dim=64)
y = embed(x)
print(y.shape)  # torch.Size([4, 10, 64])
```

------

#### 12. `nn.LSTM`

长短期记忆网络，序列建模。

- **输入**：`(seq_len, batch_size, input_size)`
- **输出**：
  - `output`: `(seq_len, batch_size, hidden_size)`
  - `(h_n, c_n)`: `(num_layers * num_directions, batch_size, hidden_size)`

```python
x = torch.randn(5, 3, 10)  # 序列长=5, batch=3, 输入维度=10
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
output, (h_n, c_n) = lstm(x)
print(output.shape)  # torch.Size([5, 3, 20])
print(h_n.shape)     # torch.Size([2, 3, 20])
```

------

#### 13. `nn.TransformerEncoderLayer`

Transformer 编码器层。

- **输入**：`(seq_len, batch_size, embed_dim)`
- **输出**：同输入形状

```python
x = torch.randn(10, 32, 512)  # 序列长=10, batch=32, embed_dim=512
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
y = encoder_layer(x)
print(y.shape)  # torch.Size([10, 32, 512])
```

------

#### 14. `nn.Softmax`

归一化激活函数。

- **输入/输出**：形状不变

```python
x = torch.randn(2, 5)
softmax = nn.Softmax(dim=1)
y = softmax(x)
print(y.shape)  # torch.Size([2, 5])
```

------

#### 15. `nn.Sigmoid`

S 型激活函数。

- **输入/输出**：形状不变

```python
x = torch.randn(4, 6)
sigmoid = nn.Sigmoid()
y = sigmoid(x)
print(y.shape)  # torch.Size([4, 6])
```

------

#### 16. `nn.Tanh`

双曲正切激活函数。

- **输入/输出**：形状不变

```python
x = torch.randn(4, 6)
tanh = nn.Tanh()
y = tanh(x)
print(y.shape)  # torch.Size([4, 6])
```

------

#### 17. `nn.AdaptiveAvgPool2d`

自适应平均池化，常用于 CNN 特征接到全连接层前。

- **输入**：`(batch_size, channels, height, width)`
- **输出**：`(batch_size, channels, target_height, target_width)`

```python
x = torch.randn(8, 64, 10, 10)
pool = nn.AdaptiveAvgPool2d((1, 1))
y = pool(x)
print(y.shape)  # torch.Size([8, 64, 1, 1])
```

------



# PyTorch 常用层输入输出对照表

| 层 (Layer)                 | 输入维度 (Input)              | 输出维度 (Output)                          | 常见用途 (Usage)     |
| -------------------------- | ----------------------------- | ------------------------------------------ | -------------------- |
| nn.Conv1d                  | (batch, in_channels, seq_len) | (batch, out_channels, out_seq_len)         | 时序/语音卷积        |
| nn.Conv2d                  | (batch, in_channels, H, W)    | (batch, out_channels, H_out, W_out)        | 图像卷积             |
| nn.Conv3d                  | (batch, in_channels, D, H, W) | (batch, out_channels, D_out, H_out, W_out) | 视频/医学图像卷积    |
| nn.Linear                  | (batch, in_features)          | (batch, out_features)                      | 全连接层/特征映射    |
| nn.MaxPool2d               | (batch, channels, H, W)       | (batch, channels, H_out, W_out)            | 二维最大池化         |
| nn.AvgPool2d               | (batch, channels, H, W)       | (batch, channels, H_out, W_out)            | 二维平均池化         |
| nn.AdaptiveAvgPool2d       | (batch, channels, H, W)       | (batch, channels, target_H, target_W)      | 自适应平均池化       |
| nn.BatchNorm1d             | (batch, features)             | 同输入                                     | 一维批归一化         |
| nn.BatchNorm2d             | (batch, channels, H, W)       | 同输入                                     | 二维批归一化         |
| nn.ReLU                    | 任意形状                      | 同输入                                     | 激活函数             |
| nn.Sigmoid                 | 任意形状                      | 同输入                                     | 激活函数             |
| nn.Tanh                    | 任意形状                      | 同输入                                     | 激活函数             |
| nn.Dropout                 | 任意形状                      | 同输入                                     | 防止过拟合           |
| nn.Flatten                 | (batch, ...)                  | (batch, -1)                                | 展平层               |
| nn.Embedding               | (batch, seq_len)              | (batch, seq_len, embed_dim)                | 词嵌入 (NLP)         |
| nn.LSTM                    | (seq_len, batch, input_size)  | (seq_len, batch, hidden_size)              | 序列建模             |
| nn.GRU                     | (seq_len, batch, input_size)  | (seq_len, batch, hidden_size)              | 序列建模             |
| nn.TransformerEncoderLayer | (seq_len, batch, embed_dim)   | 同输入                                     | Transformer 编码器层 |
| nn.Softmax                 | 任意形状                      | 同输入                                     | 归一化函数           |



# Yolo v5模型架构

YOLOv5 的主干网络通常分为四部分：**输入层、Backbone（主干）、Neck（颈部）、Head（检测头）**。下面以 YOLOv5s 为例，详细列出每一层的结构及其输入输出形状（以输入图片为 640x640 为例，batch size=1，通道数以默认为准）。

---

### 1. 输入层
- **输入**：`(1, 3, 640, 640)`  # batch, channel, height, width

---

### 2. Backbone（主干特征提取）

| 层名  | 结构/模块         | 输入形状           | 输出形状           |
| ----- | ----------------- | ------------------ | ------------------ |
| Focus | Focus             | (1, 3, 640, 640)   | (1, 32, 320, 320)  |
| Conv  | Conv2d            | (1, 32, 320, 320)  | (1, 64, 320, 320)  |
| C3    | C3                | (1, 64, 320, 320)  | (1, 64, 320, 320)  |
| Conv  | Conv2d (stride=2) | (1, 64, 320, 320)  | (1, 128, 160, 160) |
| C3    | C3                | (1, 128, 160, 160) | (1, 128, 160, 160) |
| Conv  | Conv2d (stride=2) | (1, 128, 160, 160) | (1, 256, 80, 80)   |
| C3    | C3                | (1, 256, 80, 80)   | (1, 256, 80, 80)   |
| Conv  | Conv2d (stride=2) | (1, 256, 80, 80)   | (1, 512, 40, 40)   |
| SPP   | SPP               | (1, 512, 40, 40)   | (1, 512, 40, 40)   |
| C3    | C3                | (1, 512, 40, 40)   | (1, 512, 40, 40)   |

---

### 3. Neck（FPN+PAN）

| 层名     | 结构/模块         | 输入形状                              | 输出形状           |
| -------- | ----------------- | ------------------------------------- | ------------------ |
| Conv     | Conv2d            | (1, 512, 40, 40)                      | (1, 256, 40, 40)   |
| Upsample | Upsample          | (1, 256, 40, 40)                      | (1, 256, 80, 80)   |
| Concat   | Concat            | (1, 256, 80, 80)+(1, 256, 80, 80)     | (1, 512, 80, 80)   |
| C3       | C3                | (1, 512, 80, 80)                      | (1, 256, 80, 80)   |
| Conv     | Conv2d            | (1, 256, 80, 80)                      | (1, 128, 80, 80)   |
| Upsample | Upsample          | (1, 128, 80, 80)                      | (1, 128, 160, 160) |
| Concat   | Concat            | (1, 128, 160, 160)+(1, 128, 160, 160) | (1, 256, 160, 160) |
| C3       | C3                | (1, 256, 160, 160)                    | (1, 128, 160, 160) |
| Conv     | Conv2d (stride=2) | (1, 128, 160, 160)                    | (1, 128, 80, 80)   |
| Concat   | Concat            | (1, 128, 80, 80)+(1, 256, 80, 80)     | (1, 384, 80, 80)   |
| C3       | C3                | (1, 384, 80, 80)                      | (1, 256, 80, 80)   |
| Conv     | Conv2d (stride=2) | (1, 256, 80, 80)                      | (1, 256, 40, 40)   |
| Concat   | Concat            | (1, 256, 40, 40)+(1, 512, 40, 40)     | (1, 768, 40, 40)   |
| C3       | C3                | (1, 768, 40, 40)                      | (1, 512, 40, 40)   |

---

### 4. Head（检测头）

YOLOv5 输出三个尺度的特征图，分别用于检测大、中、小目标：

| 层名         | 结构/模块 | 输入形状           | 输出形状           |
| ------------ | --------- | ------------------ | ------------------ |
| Detect Head1 | Conv2d    | (1, 128, 160, 160) | (1, 255, 160, 160) |
| Detect Head2 | Conv2d    | (1, 256, 80, 80)   | (1, 255, 80, 80)   |
| Detect Head3 | Conv2d    | (1, 512, 40, 40)   | (1, 255, 40, 40)   |

> 其中 255 = 3 × (num_classes + 5)，3 是每个网格的 anchor 数，5 是 [x, y, w, h, obj_conf]。

---

### 总结

每一层的输入输出形状都和上面类似，实际通道数和尺寸会因模型大小（s/m/l/x）和输入图片尺寸不同而变化，但结构类似。  
如果你需要更详细的每一层参数，可以用如下代码打印模型结构：

```python
from models.yolo import Model
import torch

model = Model('models/yolov5s.yaml')
x = torch.randn(1, 3, 640, 640)
y = model(x)
print(model)
```

这样可以看到每一层的详细结构和参数。



# Yolo v5模型架构（ChatGPT5）

# YOLOv5s 模型结构（层级、输入输出与作用）

| 层 (Layer)      | 输入维度 (Input)                      | 输出维度 (Output)                                            | 作用 (Function)                  |
| --------------- | ------------------------------------- | ------------------------------------------------------------ | -------------------------------- |
| **Input**       | (batch, 3, 640, 640)                  | (batch, 3, 640, 640)                                         | 输入图像                         |
| **Focus**       | (batch, 3, 640, 640)                  | (batch, 32, 320, 320)                                        | 将图像切片后拼接，并做卷积降采样 |
| **Conv**        | (batch, 32, 320, 320)                 | (batch, 64, 160, 160)                                        | 卷积 + BN + 激活                 |
| **CSP1_1**      | (batch, 64, 160, 160)                 | (batch, 64, 160, 160)                                        | 残差结构，增强梯度流动           |
| **Conv**        | (batch, 64, 160, 160)                 | (batch, 128, 80, 80)                                         | 降采样，提取更深层特征           |
| **CSP2_3**      | (batch, 128, 80, 80)                  | (batch, 128, 80, 80)                                         | 堆叠残差块，增强表达能力         |
| **Conv**        | (batch, 128, 80, 80)                  | (batch, 256, 40, 40)                                         | 降采样                           |
| **CSP3_3**      | (batch, 256, 40, 40)                  | (batch, 256, 40, 40)                                         | 深层残差块，提取中层特征         |
| **Conv**        | (batch, 256, 40, 40)                  | (batch, 512, 20, 20)                                         | 降采样                           |
| **CSP4_1**      | (batch, 512, 20, 20)                  | (batch, 512, 20, 20)                                         | 残差块，提取高级语义特征         |
| **SPPF**        | (batch, 512, 20, 20)                  | (batch, 512, 20, 20)                                         | 空间金字塔池化，扩大感受野       |
| **Conv**        | (batch, 512, 20, 20)                  | (batch, 256, 20, 20)                                         | 通道压缩，准备上采样             |
| **Upsample**    | (batch, 256, 20, 20)                  | (batch, 256, 40, 40)                                         | 上采样                           |
| **Concat**      | [(256,40,40), (256,40,40)]            | (batch, 512, 40, 40)                                         | 融合浅层和深层特征               |
| **CSP5_1**      | (batch, 512, 40, 40)                  | (batch, 256, 40, 40)                                         | PANet 特征融合                   |
| **Conv**        | (batch, 256, 40, 40)                  | (batch, 128, 40, 40)                                         | 通道压缩                         |
| **Upsample**    | (batch, 128, 40, 40)                  | (batch, 128, 80, 80)                                         | 上采样                           |
| **Concat**      | [(128,80,80), (128,80,80)]            | (batch, 256, 80, 80)                                         | 多尺度特征融合                   |
| **CSP6_1**      | (batch, 256, 80, 80)                  | (batch, 128, 80, 80)                                         | 提取小目标特征                   |
| **Conv**        | (batch, 128, 80, 80)                  | (batch, 128, 40, 40)                                         | 降采样                           |
| **Concat**      | [(128,40,40), (256,40,40)]            | (batch, 384, 40, 40)                                         | 融合中层特征                     |
| **CSP7_1**      | (batch, 384, 40, 40)                  | (batch, 256, 40, 40)                                         | 提取中目标特征                   |
| **Conv**        | (batch, 256, 40, 40)                  | (batch, 256, 20, 20)                                         | 降采样                           |
| **Concat**      | [(256,20,20), (512,20,20)]            | (batch, 768, 20, 20)                                         | 融合深层特征                     |
| **CSP8_1**      | (batch, 768, 20, 20)                  | (batch, 512, 20, 20)                                         | 提取大目标特征                   |
| **Detect Head** | (128,80,80), (256,40,40), (512,20,20) | [(batch,3,80,80,(nc+5)), (batch,3,40,40,(nc+5)), (batch,3,20,20,(nc+5))] | 输出目标检测结果                 |

📌 说明：

- `(nc+5)` = `[x, y, w, h, obj_conf, class_scores...]`
- 三个输出尺度分别对应 **小目标 (80×80)**、**中目标 (40×40)**、**大目标 (20×20)**
- Backbone（CSPDarknet）：负责特征提取
- Neck（FPN+PAN）：负责多尺度特征融合
- Head（Detect）：输出最终预测

# Pytorch 代码举例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# 1) 数据与预处理：Fashion-MNIST 单通道 28x28
transform = transforms.Compose([
    transforms.ToTensor(),                     # -> (C,H,W) in [0,1]
    transforms.Normalize((0.5,), (0.5,))      # 标准化
])

train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# 2) 模型：用你熟悉的 nn 层堆起来
class SimpleCNN(nn.Module):
    """
    输入:  (N, 1, 28, 28)
    输出:  10 类 logits (N, 10)
    结构说明:
    - Conv2d + BN + ReLU + MaxPool2d  × 2  抽取特征并降采样
    - AdaptiveAvgPool2d(1)             做全局池化以适配任意输入尺寸
    - Flatten + Dropout + Linear       分类头
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # block1: (1,28,28) -> (32,14,14)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (N,32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # (N,32,14,14)

            # block2: (32,14,14) -> (64,7,7)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (N,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # (N,64,7,7)

            # 自适应池化到 1x1，方便后面接全连接
            nn.AdaptiveAvgPool2d((1, 1))                            # (N,64,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # (N,64)
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)                              # (N,10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
print(model)


# 3) 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4) 训练与验证循环
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                     # (N,10)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate():
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(test_loader, desc="Evaluate"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

# 5) 训练若干轮
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(epoch)
    te_loss, te_acc = evaluate()
    print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={te_loss:.4f} acc={te_acc:.4f}")

# 6) 推理示例
model.eval()
sample, label = next(iter(test_loader))
sample = sample.to(device)
logits = model(sample[:8])
preds = logits.argmax(dim=1).cpu().tolist()
print("Pred labels of first 8 samples:", preds)
```

# Transformer

你好！下面详细列出**Transformer**（以标准Encoder-Decoder结构为例，每一层的输入输出），并举例说明每个模块的作用和数据形状。假设输入为批量文本序列，batch size=2，序列长度=10，词向量维度=512，注意力头数=8。

---

## 1. 输入层
- **输入**：`(batch_size, seq_len)`，如 `(2, 10)`，每个元素是词ID
- **Embedding**：`nn.Embedding`
  - 输入：`(2, 10)`
  - 输出：`(2, 10, 512)`（每个词变成512维向量）

---

## 2. Positional Encoding
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`（加上位置信息）

---

## 3. Encoder部分（以1层为例，实际可堆叠多层）

### 3.1 多头自注意力（MultiHeadAttention）
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 3.2 残差+LayerNorm
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 3.3 前馈全连接（FeedForward）
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 3.4 残差+LayerNorm
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

---

## 4. Decoder部分（以1层为例）

### 4.1 Masked MultiHeadAttention（对目标序列自注意力）
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 4.2 残差+LayerNorm
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 4.3 Encoder-Decoder MultiHeadAttention
- **输入**：Decoder输出`(2, 10, 512)`，Encoder输出`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 4.4 残差+LayerNorm
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 4.5 前馈全连接（FeedForward）
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

### 4.6 残差+LayerNorm
- **输入**：`(2, 10, 512)`
- **输出**：`(2, 10, 512)`

---

## 5. 输出层
- **Linear投影到词表大小**：`nn.Linear(512, vocab_size)`
  - 输入：`(2, 10, 512)`
  - 输出：`(2, 10, vocab_size)`（每个位置是一个词的概率分布）

---

## 总结表

| 层名                         | 输入形状     | 输出形状       | 说明             |
| ---------------------------- | ------------ | -------------- | ---------------- |
| Embedding                    | (2, 10)      | (2, 10, 512)   | 词ID转向量       |
| Positional Encoding          | (2, 10, 512) | (2, 10, 512)   | 加位置信息       |
| Encoder Self-Attention       | (2, 10, 512) | (2, 10, 512)   | 多头注意力       |
| Encoder FeedForward          | (2, 10, 512) | (2, 10, 512)   | 前馈网络         |
| Decoder Masked Self-Attn     | (2, 10, 512) | (2, 10, 512)   | 目标序列自注意力 |
| Decoder Encoder-Decoder Attn | (2, 10, 512) | (2, 10, 512)   | 融合编码器信息   |
| Decoder FeedForward          | (2, 10, 512) | (2, 10, 512)   | 前馈网络         |
| Linear                       | (2, 10, 512) | (2, 10, vocab) | 输出词概率       |

---

### PyTorch官方实现（nn.Transformer）

```python
import torch
import torch.nn as nn

model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
src = torch.randn(10, 2, 512)  # (seq_len, batch, d_model)
tgt = torch.randn(10, 2, 512)
out = model(src, tgt)
print(out.shape)  # torch.Size([10, 2, 512])
```
- 输入输出都是 (seq_len, batch, d_model)

```
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
    )
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
)
```

# Transformer每一层作用

下面详细列出**Transformer**每一层的作用（以PyTorch nn.Transformer为例）：

---

### 1. 输入嵌入（Embedding + Positional Encoding）
- **作用**：将离散的词ID或特征转为连续向量，并加入位置信息，使模型能感知顺序。
- **输入**：词ID序列 (batch, seq_len) 或特征 (seq_len, batch, d_model)
- **输出**：嵌入向量 (seq_len, batch, d_model)

---

### 2. Encoder部分（每层都重复以下结构）

#### 2.1 多头自注意力（Multi-Head Self-Attention）
- **作用**：让序列中每个位置都能关注到其它所有位置的信息，实现全局依赖建模。
- **输入/输出**： (seq_len, batch, d_model)

#### 2.2 残差连接 + LayerNorm
- **作用**：缓解梯度消失/爆炸，提升训练稳定性。
- **输入/输出**： (seq_len, batch, d_model)

#### 2.3 前馈全连接网络（Feed Forward）
- **作用**：对每个位置独立地进行非线性变换，提升表达能力。
- **结构**：Linear(d_model, dim_feedforward) → 激活 → Linear(dim_feedforward, d_model)
- **输入/输出**： (seq_len, batch, d_model)

#### 2.4 残差连接 + LayerNorm
- **作用**：同上，稳定训练。
- **输入/输出**： (seq_len, batch, d_model)

---

### 3. Decoder部分（每层都重复以下结构）

#### 3.1 Masked 多头自注意力（Masked Multi-Head Self-Attention）
- **作用**：只允许每个位置看到它之前的内容，保证自回归生成。
- **输入/输出**： (tgt_seq_len, batch, d_model)

#### 3.2 残差连接 + LayerNorm
- **作用**：同上。
- **输入/输出**： (tgt_seq_len, batch, d_model)

#### 3.3 Encoder-Decoder 多头注意力（Cross-Attention）
- **作用**：让解码器每个位置能关注编码器所有位置，实现条件生成。
- **输入**：解码器输出 (tgt_seq_len, batch, d_model)，编码器输出 (src_seq_len, batch, d_model)
- **输出**： (tgt_seq_len, batch, d_model)

#### 3.4 残差连接 + LayerNorm
- **作用**：同上。
- **输入/输出**： (tgt_seq_len, batch, d_model)

#### 3.5 前馈全连接网络（Feed Forward）
- **作用**：同Encoder。
- **输入/输出**： (tgt_seq_len, batch, d_model)

#### 3.6 残差连接 + LayerNorm
- **作用**：同上。
- **输入/输出**： (tgt_seq_len, batch, d_model)

---

### 4. 输出层（通常外部添加）
- **作用**：将解码器输出映射到词表或标签空间，得到最终预测。
- **结构**：Linear(d_model, vocab_size)
- **输入**： (tgt_seq_len, batch, d_model)
- **输出**： (tgt_seq_len, batch, vocab_size)

---

**总结：**
- Encoder和Decoder都由多层堆叠，每层都包含注意力、前馈、残差和归一化。
- Decoder比Encoder多了Masked Self-Attention和Cross-Attention。
- 每一层的输入输出形状通常保持不变（d_model维度）。

# Yolo v5模型架构

下面是一个**最小YOLOv5模型实现示例**，包含主干结构（Focus、Conv、C3、SPP）、Neck（FPN/PAN）、Head（Detect），并详细标注每一层的输入输出维度。  
假设输入图片为`(1, 3, 640, 640)`（batch=1，3通道，640x640），以YOLOv5s为例：

```python
import torch
import torch.nn as nn

# Focus层
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1)
    def forward(self, x):
        # 切片拼接
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)

# C3层（简化版）
class C3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

# SPP层（简化版）
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
    def forward(self, x):
        return self.conv(x)

# 主干
class YOLOv5Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 32)           # (1, 3, 640, 640) -> (1, 32, 320, 320)
        self.conv1 = nn.Conv2d(32, 64, 3, 2, 1)  # (1, 32, 320, 320) -> (1, 64, 160, 160)
        self.c3_1  = C3(64, 64)             # (1, 64, 160, 160) -> (1, 64, 160, 160)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1) # (1, 64, 160, 160) -> (1, 128, 80, 80)
        self.c3_2  = C3(128, 128)           # (1, 128, 80, 80) -> (1, 128, 80, 80)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1) # (1, 128, 80, 80) -> (1, 256, 40, 40)
        self.c3_3  = C3(256, 256)           # (1, 256, 40, 40) -> (1, 256, 40, 40)
        self.spp   = SPP(256, 256)          # (1, 256, 40, 40) -> (1, 256, 40, 40)
        # 检测头（简化为1个输出）
        self.head  = nn.Conv2d(256, 3 * (5 + 80), 1) # (1, 256, 40, 40) -> (1, 255, 40, 40)

    def forward(self, x):
        print('Input:', x.shape)             # (1, 3, 640, 640)
        x = self.focus(x)
        print('After Focus:', x.shape)       # (1, 32, 320, 320)
        x = self.conv1(x)
        print('After Conv1:', x.shape)       # (1, 64, 160, 160)
        x = self.c3_1(x)
        print('After C3_1:', x.shape)        # (1, 64, 160, 160)
        x = self.conv2(x)
        print('After Conv2:', x.shape)       # (1, 128, 80, 80)
        x = self.c3_2(x)
        print('After C3_2:', x.shape)        # (1, 128, 80, 80)
        x = self.conv3(x)
        print('After Conv3:', x.shape)       # (1, 256, 40, 40)
        x = self.c3_3(x)
        print('After C3_3:', x.shape)        # (1, 256, 40, 40)
        x = self.spp(x)
        print('After SPP:', x.shape)         # (1, 256, 40, 40)
        x = self.head(x)
        print('Output:', x.shape)            # (1, 255, 40, 40)
        return x

# 测试
if __name__ == "__main__":
    model = YOLOv5Tiny()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
```

---

### 每一层输入输出维度说明
| 层名  | 输入维度          | 输出维度          |
| ----- | ----------------- | ----------------- |
| Input | (1, 3, 640, 640)  | (1, 3, 640, 640)  |
| Focus | (1, 3, 640, 640)  | (1, 32, 320, 320) |
| Conv1 | (1, 32, 320, 320) | (1, 64, 160, 160) |
| C3_1  | (1, 64, 160, 160) | (1, 64, 160, 160) |
| Conv2 | (1, 64, 160, 160) | (1, 128, 80, 80)  |
| C3_2  | (1, 128, 80, 80)  | (1, 128, 80, 80)  |
| Conv3 | (1, 128, 80, 80)  | (1, 256, 40, 40)  |
| C3_3  | (1, 256, 40, 40)  | (1, 256, 40, 40)  |
| SPP   | (1, 256, 40, 40)  | (1, 256, 40, 40)  |
| Head  | (1, 256, 40, 40)  | (1, 255, 40, 40)  |

---

如需添加Neck、PAN、更多检测头或多尺度输出，可继续扩展。这个例子已涵盖YOLOv5主干的最小实现和每层输入输出。

# Yolo v5每层作用

下面是**YOLOv5主干结构中每一层的作用**，以常见的 YOLOv5s 为例：

---

### 1. **Focus**
- **作用**：将输入图片的空间信息（每2x2像素）拼接到通道维度，实现下采样和信息融合。
- **输入**：`(batch, 3, 640, 640)`
- **输出**：`(batch, 32, 320, 320)`

---

### 2. **Conv（卷积层）**
- **作用**：提取局部特征，增加通道数，减小空间分辨率（通常stride=2）。
- **输入**：`(batch, 32, 320, 320)`
- **输出**：`(batch, 64, 160, 160)`

---

### 3. **C3（CSP Bottleneck）**
- **作用**：跨阶段部分结构（CSP），包含多个Bottleneck模块，提升特征表达能力，减少计算量。
- **输入**：`(batch, 64, 160, 160)`
- **输出**：`(batch, 64, 160, 160)`

---

### 4. **Conv（卷积层）**
- **作用**：继续提取特征并下采样。
- **输入**：`(batch, 64, 160, 160)`
- **输出**：`(batch, 128, 80, 80)`

---

### 5. **C3**
- **作用**：同上，进一步提取特征。
- **输入**：`(batch, 128, 80, 80)`
- **输出**：`(batch, 128, 80, 80)`

---

### 6. **Conv（卷积层）**
- **作用**：继续下采样，提取更深层次特征。
- **输入**：`(batch, 128, 80, 80)`
- **输出**：`(batch, 256, 40, 40)`

---

### 7. **C3**
- **作用**：同上，提取更深层特征。
- **输入**：`(batch, 256, 40, 40)`
- **输出**：`(batch, 256, 40, 40)`

---

### 8. **SPP（空间金字塔池化）**
- **作用**：通过多尺度池化增强感受野，融合不同尺度的上下文信息。
- **输入**：`(batch, 256, 40, 40)`
- **输出**：`(batch, 256, 40, 40)`

---

### 9. **C3**
- **作用**：进一步特征融合。
- **输入**：`(batch, 256, 40, 40)`
- **输出**：`(batch, 256, 40, 40)`

---

### 10. **Neck（FPN+PAN）**
- **作用**：特征融合与传递，将不同尺度的特征进行上采样、下采样和融合，便于检测不同大小目标。
- **输入**：多个尺度的特征图
- **输出**：多个尺度的特征图（如`(batch, 128, 80, 80)`、`(batch, 256, 40, 40)`、`(batch, 512, 20, 20)`）

---

### 11. **Detect Head（检测头）**
- **作用**：对每个尺度的特征图进行预测，输出每个网格的类别、置信度和边界框参数。
- **输入**：融合后的特征图
- **输出**：预测结果（如`(batch, anchors*(num_classes+5), H, W)`）

---

**总结：**
- **Focus**：空间到通道，初步下采样
- **Conv/C3/SPP**：特征提取与融合
- **Neck**：多尺度特征融合
- **Detect Head**：输出检测结果
