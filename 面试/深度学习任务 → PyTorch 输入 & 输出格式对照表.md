------

# 深度学习任务 → PyTorch 输入 & 输出格式对照表

| 任务类别                                    | 输入数据格式                                                 | 输出数据格式                                                 | 说明与示例                                                   |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **序列分类 (Sequence Classification)**      | `(batch_size, seq_len)` → token ids                          | `(batch_size, num_labels)`                                   | 文本分类任务。例：情感分析，输出正负情绪的概率分布。         |
| **Token 分类 (Token Classification / NER)** | `(batch_size, seq_len)`                                      | `(batch_size, seq_len, num_labels)`                          | 每个 token 一个标签。例：句子 `"John lives in Paris"` → 每个词对应标签。 |
| **因果语言建模 (Causal LM)**                | `(batch_size, seq_len)`                                      | `(batch_size, seq_len, vocab_size)`                          | 输入序列，预测下一个 token。例：`"The cat sat on the"` → 预测 `"mat"`。 |
| **掩码语言建模 (Masked LM)**                | `(batch_size, seq_len)`，含 `[MASK]`                         | `(batch_size, seq_len, vocab_size)`                          | 预测被 mask 的 token。例：`"The cat sat on the [MASK]"` → 输出 `"mat"` 概率最高。 |
| **序列到序列 (Seq2Seq LM / 翻译/摘要)**     | Encoder 输入 `(batch_size, src_seq_len)`；Decoder 输入 `(batch_size, tgt_seq_len)` | `(batch_size, tgt_seq_len, vocab_size)`                      | 翻译、摘要等。例：英文 `"Hello"` → 输出法文 `"Bonjour"`。    |
| **问答 (QA - Extractive)**                  | `(batch_size, seq_len)` (context+question 拼接)              | `(batch_size, seq_len)` start_logits + `(batch_size, seq_len)` end_logits | 模型预测答案 span 的起止位置。例：答案 `"Paris"`。           |
| **多选任务 (Multiple Choice)**              | `(batch_size, num_choices, seq_len)`                         | `(batch_size, num_choices)`                                  | 每个候选一个得分。例：问题 `"Where is John?"`，选项 ["London", "Paris"] → Paris 分数高。 |
| **图像分类**                                | `(batch_size, channels, height, width)`                      | `(batch_size, num_classes)`                                  | 图像分类任务。例：CIFAR-10 图像 → 输出 10 类概率分布。       |
| **目标检测 (Detection)**                    | 图像 `(B, C, H, W)`；标签 `List[Dict]`                       | 训练：Loss dict；推理：List[Dict]，每个包含 `boxes (N,4)`, `labels (N,)`, `scores (N,)` | 每张图像输出若干检测框和类别。                               |
| **图像分割 (Segmentation)**                 | 图像 `(B, C, H, W)`；mask `(B, H, W)`                        | `(B, num_classes, H, W)`                                     | 每个像素一个类别概率。                                       |
| **语音分类 (Audio Classification)**         | `(batch_size, time_steps)` 或 `(batch_size, channels, time_steps)` | `(batch_size, num_classes)`                                  | 输入波形 → 输出类别分布。                                    |
| **语音识别 (ASR - CTC)**                    | `(batch_size, time_steps, features)`                         | `(batch_size, time_steps, vocab_size)`                       | CTC 输出每帧 token 分布，经过解码得到文字。                  |
| **表格问答 (Table QA)**                     | `(batch_size, seq_len)` (表格转序列)                         | `(batch_size, seq_len)` 或 `(batch_size, num_labels)`        | 输出可能是答案 span 或分类结果。                             |
| **视觉问答 (VQA)**                          | 图像 `(B, C, H, W)` + 问题 `(B, seq_len)`                    | `(batch_size, num_answers)` 或 `(batch_size, vocab_size)`    | 多模态输入 → 输出答案类别或生成答案文本。                    |

------

这样你就能一眼看到：

- 输入数据张量怎么组织
- 输出一般是什么形状

👉 在训练时，输出通常是 **logits**（未经过 softmax），方便用 `CrossEntropyLoss` 等 loss function。



# 举几个代码例子

### 1. 文本分类 (SequenceClassification)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer(["I love this movie!", "This film is terrible..."],
                   padding=True, truncation=True, return_tensors="pt")

outputs = model(**inputs)  # inputs['input_ids'] shape: (batch_size, seq_len)
```

### 2. 图像分类

```python
import torch
from torchvision import models

model = models.resnet18(pretrained=True)
x = torch.randn(16, 3, 224, 224)  # batch=16
y = model(x)  # 输出 shape: (16, num_classes)
```

### 3. 目标检测 (FasterRCNN)

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

model = fasterrcnn_resnet50_fpn(pretrained=True)
images = [torch.randn(3, 224, 224), torch.randn(3, 300, 300)]  # list of images
targets = [
    {"boxes": torch.tensor([[50, 50, 150, 150]]), "labels": torch.tensor([1])},
    {"boxes": torch.tensor([[30, 40, 100, 200]]), "labels": torch.tensor([2])}
]
output = model(images, targets)
```

------





