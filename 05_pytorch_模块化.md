## 05. PyTorch 模块化

这一部分解答了“如何将我的笔记本代码转换为 Python 脚本？”的问题。

为此，我们将把 [notebook 04. PyTorch 自定义数据集](https://www.learnpytorch.io/04_pytorch_custom_datasets/) 中最有用的代码单元转换为一系列 Python 脚本，并将它们保存在名为 [`going_modular`](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular) 的目录中。

## 什么是模块化？

模块化是指将笔记本中的代码（来自 Jupyter Notebook 或 Google Colab 笔记本）转换为一系列具有类似功能的 Python 脚本。

例如，我们可以将笔记本中的代码从一系列单元格转换为以下 Python 文件：

- `data_setup.py` - 一个用于准备和下载数据的文件（如果需要）。
- `engine.py` - 一个包含各种训练函数的文件。
- `model_builder.py` 或 `model.py` - 一个用于创建 PyTorch 模型的文件。
- `train.py` - 一个用来调用其他文件并训练目标 PyTorch 模型的文件。
- `utils.py` - 一个包含有用工具函数的文件。

> **注意：** 上述文件的命名和布局将取决于你的使用情况和代码需求。Python 脚本和单个笔记本单元一样通用，这意味着你几乎可以为任何功能创建一个脚本。

## 为什么要模块化？

笔记本非常适合快速地进行实验和探索。

然而，对于较大规模的项目，你可能会发现 Python 脚本更具可重现性并且更容易运行。

尽管这是一个有争议的话题，正如 [Netflix](https://netflixtechblog.com/notebook-innovation-591ee3221233) 这样的公司展示了他们如何将笔记本用于生产代码。

**生产代码** 是指为某人或某物提供服务的代码。

例如，如果你有一个在线运行的应用程序，其他人可以访问和使用，那么运行该应用程序的代码被视为 **生产代码**。

像 fast.ai 的 [`nb-dev`](https://github.com/fastai/nbdev)（笔记本开发的缩写）这样的库使你能够通过 Jupyter 笔记本编写完整的 Python 库（包括文档）。

### 笔记本与 Python 脚本的优缺点

这两者都有其优势。

但下面的列表总结了一些主要话题。

|            | **优点**                                       | **缺点**                       |
| ---------- | ---------------------------------------------- | ------------------------------ |
| **笔记本** | 易于实验/快速入门                              | 版本控制可能比较难             |
|            | 易于共享（例如分享 Google Colab 笔记本的链接） | 难以只使用其中的某些部分       |
|            | 非常直观                                       | 文本和图形可能会妨碍代码的阅读 |

|                 | **优点**                                                   | **缺点**                                                     |
| --------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| **Python 脚本** | 可以将代码打包在一起（避免在不同笔记本中重复编写相似代码） | 实验过程不如笔记本直观（通常需要运行整个脚本，而不是单独运行某个单元格） |
|                 | 可以使用 git 进行版本控制                                  |                                                              |
|                 | 许多开源项目使用脚本                                       |                                                              |
|                 | 更大的项目可以在云服务提供商上运行（而笔记本支持较少）     |                                                              |

### 我的工作流程

我通常在 Jupyter/Google Colab 笔记本中开始机器学习项目，以进行快速实验和可视化。

然后，当我有了有效的代码后，我会将最有用的代码片段移到 Python 脚本中。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-my-workflow-for-experimenting.png" alt="开始机器学习代码的可能工作流程，从 Jupyter 或 Google Colab 笔记本开始，然后在代码有效时转到 Python 脚本。" width="1000/>

*编写机器学习代码的有多种工作流程。有些人喜欢从脚本开始，而有些人（像我）则喜欢从笔记本开始，之后再转到脚本。*

### PyTorch 在实际应用中的使用

在你旅程中，你会发现很多基于 PyTorch 的机器学习项目代码库会有如何以 Python 脚本的形式运行 PyTorch 代码的说明。

例如，你可能会被要求在终端/命令行中运行如下代码来训练一个模型：

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-python-train-command-line-annotated.png" alt="命令行调用用于训练 PyTorch 模型的代码，带有不同的超参数设置" width="1000/>

*在命令行上运行 PyTorch `train.py` 脚本，并使用不同的超参数设置进行训练。*

在这种情况下，`train.py` 是目标 Python 脚本，它可能包含用于训练 PyTorch 模型的函数。

而 `--model`、`--batch_size`、`--lr` 和 `--num_epochs` 被称为 **参数标志**。

你可以设置它们为任何你想要的值，如果它们与 `train.py` 兼容，它们就会工作，否则会报错。

例如，假设我们想训练我们在笔记本 04 中构建的 TinyVGG 模型，训练 10 个 epoch，批量大小为 32，学习率为 0.001：

```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

你可以在 `train.py` 文件中设置任意数量的这些参数标志，以适应你的需求。

用于训练最先进计算机视觉模型的 PyTorch 博客文章采用了这种风格。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-training-sota-recipe.png" alt="训练最先进计算机视觉模型的 PyTorch 命令行训练脚本配方，使用 8 个 GPU。" width="800/>

*用于训练最先进计算机视觉模型的 PyTorch 命令行训练脚本配方，使用 8 个 GPU。来源： [PyTorch 博客](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#the-training-recipe)。*

## 我们将要做什么

本节的主要概念是：**将有用的笔记本代码单元转换为可重用的 Python 文件。**

这样做将节省我们重复编写相同代码的时间。

这一部分有两个笔记本：

1. [**05. Going Modular: Part 1（单元模式）**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - 这个笔记本作为传统的 Jupyter 笔记本/Google Colab 笔记本运行，是 [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/) 的精简版本。
2. [**05. Going Modular: Part 2（脚本模式）**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - 这个笔记本与第 1 部分相同，但增加了将每个主要部分转换为 Python 脚本的功能，例如 `data_setup.py` 和 `train.py`。

文本内容侧重于脚本模式笔记本中的代码单元，即顶部有 `%%writefile ...` 的单元。

### 为什么分为两部分？

因为有时候最好的学习方式是看看它与其他东西的 **不同之处**。

如果你并排运行这两个笔记本，你会看到它们的不同，这正是关键的学习内容。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-notebook-cell-mode-vs-script-mode.png" alt="运行单元模式笔记本与脚本模式笔记本的对比" width="1000/>

*并排运行第 05 部分的两个笔记本。你会注意到 **脚本模式笔记本包含额外的代码单元**，将单元模式笔记本中的代码转换为 Python 脚本。*

### 我们要做的事情

到本节结束时，我们希望实现以下两点：

1. 通过命令行用一行代码训练我们在笔记本 04 中构建的模型（Food Vision Mini）：`python train.py`。
2. 一个可重用 Python 脚本的目录结构，例如：

```
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

### 需要注意的事项

- **文档字符串** - 编写可重现和易于理解的代码非常重要。考虑到这一点，我们将把每个脚本中的函数/类都按照 Google 的 [Python 文档字符串风格](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) 来创建。
- **脚本顶部的导入** - 由于我们将要创建的所有 Python 脚本都可以视为独立的小程序，所以所有脚本都需要在脚本开始时导入它们的输入模块，例如：

```python
# 导入 train.py 所需的模块
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
```

## 0. 单元模式与脚本模式

一个单元模式笔记本，比如 [05. Going Modular Part 1 (单元模式)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb)，是一个正常运行的笔记本，每个单元格要么是代码，要么是 markdown。

而脚本模式笔记本，比如 [05. Going Modular Part 2 (脚本模式)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb)，与单元模式笔记本非常相似，但是很多代码单元可能已经转化为 Python 脚本。

> **注意：** 你不 *需要* 通过笔记本创建 Python 脚本，你可以直接通过 IDE（集成开发环境）如 [VS Code](https://code.visualstudio.com/) 来创建它们。将脚本模式笔记本作为这一部分的一部分，目的是展示从笔记本到 Python 脚本的转换方式。

## 1. 获取数据

在每个 05 笔记本中获取数据的方式与 [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#1-get-data) 中的方式相同。

我们通过 Python 的 `requests` 模块调用 GitHub 来下载一个 `.zip` 文件并解压。

```python
import os
import requests
import zipfile
from pathlib import Path

# 设置数据文件夹路径
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 如果图像文件夹不存在，则下载并准备数据...
if image_path.is_dir():
    print(f"{image_path} 目录已存在。")
else:
    print(f"没有找到 {image_path} 目录，正在创建...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# 下载披萨、牛排、寿司数据
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("正在下载披萨、牛排、寿司数据...")
    f.write(request.content)

# 解压披萨、牛排、寿司数据
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("正在解压披萨、牛排、寿司数据...") 
    zip_ref.extractall(image_path)

# 删除 zip 文件
os.remove(data_path / "pizza_steak_sushi.zip")
```

这将生成一个名为 `data` 的文件夹，其中包含另一个名为 `pizza_steak_sushi` 的文件夹，里面有标准图像分类格式的披萨、牛排和寿司图像。

```
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   │   ├── image01.jpeg
    │   │   └── ...
    │   ├── steak/
    │   └── sushi/
    └── test/
        ├── pizza/
        ├── steak/
        └── sushi/
```

## 2. 创建数据集和 DataLoader (`data_setup.py`)

一旦我们有了数据，就可以将其转换为 PyTorch 的 `Dataset` 和 `DataLoader`（一个用于训练数据，一个用于测试数据）。

我们将有用的 `Dataset` 和 `DataLoader` 创建代码转换为一个名为 `create_dataloaders()` 的函数。

然后我们将它写入文件，使用 `%%writefile going_modular/data_setup.py`：

```python
%%writefile going_modular/data_setup.py
"""
包含用于创建 PyTorch DataLoader 的功能，适用于图像分类数据。
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """创建训练和测试的 DataLoader。

  输入训练和测试目录路径，将它们转换为 PyTorch Dataset，然后转换为 PyTorch DataLoader。

  参数：
    train_dir: 训练数据目录路径。
    test_dir: 测试数据目录路径。
    transform: 对训练和测试数据执行的 torchvision 转换。
    batch_size: 每个 DataLoader 中每批次的数据量。
    num_workers: 每个 DataLoader 的工作线程数。

  返回：
    返回一个元组，包含 (train_dataloader, test_dataloader, class_names)。
    其中 class_names 是目标类别的列表。
    示例用法：
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # 使用 ImageFolder 创建数据集
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # 获取类别名称
  class_names = train_data.classes

  # 将图像转换为数据加载器
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # 不需要对测试数据进行洗牌
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

现在，如果我们想创建 `DataLoader`，可以使用 `data_setup.py` 中的函数，如下所示：

```python
# 导入 data_setup.py
from going_modular import data_setup

# 创建训练和测试的 DataLoader，并获取类名列表
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
```

## 3. 创建模型 (`model_builder.py`)

在过去的几个笔记本（notebook 03 和 notebook 04）中，我们已经构建了几次 TinyVGG 模型。

因此，将模型放入一个脚本中，以便我们可以反复使用它是有意义的。

让我们将 `TinyVGG()` 模型类放入一个脚本中，并使用 `%%writefile going_modular/model_builder.py` 进行保存：

```python
%%writefile going_modular/model_builder.py
"""
包含用于实例化 TinyVGG 模型的 PyTorch 代码。
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """创建 TinyVGG 架构。

  在 PyTorch 中复制 CNN explainer 网站上的 TinyVGG 架构。
  查看原始架构： https://poloclub.github.io/cnn-explainer/
  
  参数：
    input_shape: 一个整数，表示输入通道数。
    hidden_units: 一个整数，表示层之间的隐藏单元数。
    output_shape: 一个整数，表示输出单元数。
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # 这个 in_features 的形状是从哪里来的？
          # 这是因为我们网络中的每一层都在压缩并改变输入数据的形状。
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- 利用操作符融合的好处
```

现在，我们可以通过导入模型，而不是每次都从头开始编码 TinyVGG 模型：

```python
import torch
# 导入 model_builder.py
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# 从 "model_builder.py" 脚本中实例化模型
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
```

## 4. 创建 `train_step()` 和 `test_step()` 函数，以及 `train()` 来将它们组合起来

我们在 [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions) 中编写了几个训练函数：

1. `train_step()` - 接收一个模型、`DataLoader`、损失函数和优化器，并在 `DataLoader` 上训练模型。
2. `test_step()` - 接收一个模型、`DataLoader` 和损失函数，并在 `DataLoader` 上评估模型。
3. `train()` - 在给定的 epoch 数量内同时执行 1 和 2，并返回结果字典。

由于这些将是我们模型训练的 *引擎*，我们可以将它们都放入一个名为 `engine.py` 的 Python 脚本中，并使用 `%%writefile going_modular/engine.py`：

```python
%%writefile going_modular/engine.py
"""
包含训练和测试 PyTorch 模型的函数。
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """训练 PyTorch 模型的单个 epoch。

  将目标 PyTorch 模型转为训练模式，然后执行所有必要的训练步骤（前向传播、
  损失计算、优化器步骤）。

  参数：
    model: 要训练的 PyTorch 模型。
    dataloader: 用于训练的 DataLoader 实例。
    loss_fn: 要最小化的 PyTorch 损失函数。
    optimizer: 用于最小化损失函数的 PyTorch 优化器。
    device: 用于计算的目标设备（例如 "cuda" 或 "cpu"）。

  返回：
    训练损失和训练准确率的元组。
    形式为 (train_loss, train_accuracy)，例如：
    
    (0.1112, 0.8743)
  """
  # 将模型设置为训练模式
  model.train()
  
  # 设置训练损失和训练准确率的初始值
  train_loss, train_acc = 0, 0
  
  # 遍历数据加载器中的数据批次
  for batch, (X, y) in enumerate(dataloader):
      # 将数据发送到目标设备
      X, y = X.to(device), y.to(device)

      # 1. 前向传播
      y_pred = model(X)

      # 2. 计算并累加损失
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. 优化器零梯度
      optimizer.zero_grad()

      # 4. 损失反向传播
      loss.backward()

      # 5. 优化器步骤
      optimizer.step()

      # 计算并累加准确率
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # 调整度量值以获取每个批次的平均损失和准确率 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc
```

现在，我们已经完成了 `engine.py` 脚本，我们可以通过导入该脚本来使用它：

```python
# 导入 engine.py
from going_modular import engine

# 使用 engine.py 中的 train() 函数进行训练
engine.train(...)
```

## 5. 创建保存模型的函数 (`utils.py`)

在训练过程中或训练结束后，通常你会希望保存模型。

因为我们之前已经多次编写了保存模型的代码，所以我们将它转化为一个函数，并保存到文件中。

将辅助函数保存到名为 `utils.py` 的文件中是常见的做法（`utils` 是 `utilities` 的缩写）。

让我们将 `save_model()` 函数保存到 `utils.py` 文件中，使用 `%%writefile going_modular/utils.py`：

```python
%%writefile going_modular/utils.py
"""
包含用于 PyTorch 模型训练和保存的各种实用功能。
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """将 PyTorch 模型保存到目标目录。

  参数：
    model: 要保存的 PyTorch 模型。
    target_dir: 保存模型的目录。
    model_name: 保存模型的文件名，应该包括 `.pth` 或 `.pt` 文件扩展名。
  
  示例用法：
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tinyvgg_model.pth")
  """
  # 创建目标目录
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # 创建模型保存路径
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name 应以 '.pt' 或 '.pth' 结尾"
  model_save_path = target_dir_path / model_name

  # 保存模型的 state_dict()
  print(f"[INFO] 正在保存模型到: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
```

现在，如果我们想使用 `save_model()` 函数，而不是每次都写它，我们可以导入并使用它：

```python
# 导入 utils.py
from going_modular import utils

# 保存模型到文件
save_model(model=...
           target_dir=...,
           model_name=...)
```

## 6. 训练、评估并保存模型 (`train.py`)

如前所述，你会经常遇到将所有功能结合在 `train.py` 文件中的 PyTorch 项目。

这个文件本质上是在说“使用任何可用的数据训练模型”。

在我们的 `train.py` 文件中，我们将结合我们创建的其他 Python 脚本的所有功能，用于训练模型。

这样，我们就可以通过命令行用一行代码训练 PyTorch 模型：

```
python train.py
```

为了创建 `train.py`，我们将按照以下步骤进行：

1. 导入各种依赖项，特别是 `torch`、`os`、`torchvision.transforms` 以及来自 `going_modular` 目录的所有脚本（`data_setup`、`engine`、`model_builder`、`utils`）。

- **注意：** 由于 `train.py` 将位于 `going_modular` 目录中，我们可以通过 `import ...` 导入其他模块，而不是 `from going_modular import ...`。

1. 设置各种超参数，如批量大小、epoch 数量、学习率和隐藏单元数（这些将来可以通过 [Python 的 `argparse`](https://docs.python.org/3/library/argparse.html) 设置）。
2. 设置训练和测试目录。
3. 设置设备无关的代码。
4. 创建必要的数据转换。
5. 使用 `data_setup.py` 创建 DataLoader。
6. 使用 `model_builder.py` 创建模型。
7. 设置损失函数和优化器。
8. 使用 `engine.py` 训练模型。
9. 使用 `utils.py` 保存模型。

我们可以通过在笔记本单元格中写入 `%%writefile going_modular/train.py` 来创建 `train.py`：

```python
%%writefile going_modular/train.py
"""
使用设备无关代码训练 PyTorch 图像分类模型。
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# 设置超参数
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# 设置目录
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# 设置目标设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建转换
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# 使用 data_setup.py 创建 DataLoader
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# 使用 model_builder.py 创建模型
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# 设置损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
```

现在，我们可以通过运行以下命令行来训练 PyTorch 模型：

```
python train.py
```

这样，我们就可以利用我们创建的所有其他代码脚本来训练模型。

如果我们想要的话，我们还可以调整 `train.py` 文件，以使用 Python 的 `argparse` 模块来接受命令行参数，这样就能提供不同的超参数设置，如之前讨论的那样：

```
python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20
```

这使得我们能够灵活地调整训练过程中的各个参数。

## 练习

**资源：**

- [05 练习模板笔记本](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb)
- [05 示例解决方案笔记本](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb)
  - [YouTube 上的 05 解决方案笔记本实时编码演示](https://youtu.be/ijgFhMK3pp4)

**练习：**

1. 将获取数据的代码（来自上述第 1 节 获取数据部分）转换为 Python 脚本，例如 `get_data.py`。
   - 当你运行 `python get_data.py` 时，它应该检查数据是否已存在，如果存在则跳过下载。
   - 如果数据下载成功，你应该能够从 `data` 目录访问 `pizza_steak_sushi` 图像。
2. 使用 [Python 的 `argparse` 模块](https://docs.python.org/3/library/argparse.html)，使 `train.py` 能够接受自定义的超参数值。
   - 为以下参数添加命令行标志：
     - 训练/测试目录
     - 学习率
     - 批量大小
     - 训练的 epoch 数量
     - TinyVGG 模型中的隐藏单元数
   - 保持这些参数的默认值与当前设置一致（如在笔记本 05 中）。
   - 例如，你应该能够运行类似以下命令来训练 TinyVGG 模型： `python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20`。
   - **注意：** 由于 `train.py` 依赖于我们在第 05 部分创建的其他脚本，如 `model_builder.py`、`utils.py` 和 `engine.py`，因此你还需要确保这些脚本可用。你可以在 [课程 GitHub 上的 `going_modular` 文件夹](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular) 中找到它们。
3. 创建一个脚本（例如 `predict.py`），可以根据目标图像的文件路径进行预测。
   - 例如，你应该能够运行命令 `python predict.py some_image.jpeg`，并让训练好的 PyTorch 模型对该图像进行预测并返回其预测结果。
   - 要查看预测代码示例，请查看 [notebook 04 中的自定义图像预测部分](https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function)。
   - 你可能还需要编写代码来加载训练好的模型。

## 课外扩展

- 要了解更多关于 Python 项目结构的信息，请查看 Real Python 的 [Python 应用程序布局指南](https://realpython.com/python-application-layouts/)。
- 想要为你的 PyTorch 代码提供样式灵感，请查看 [Igor Susmelj 的 PyTorch 样式指南](https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model)（本章中的许多样式基于此指南以及其他类似的 PyTorch 项目）。
- 如果你想要一个 `train.py` 脚本和 PyTorch 团队编写的其他 PyTorch 脚本，用于训练最先进的图像分类模型，请查看他们在 [GitHub 上的 `classification` 仓库](https://github.com/pytorch/vision/tree/main/references/classification)。
