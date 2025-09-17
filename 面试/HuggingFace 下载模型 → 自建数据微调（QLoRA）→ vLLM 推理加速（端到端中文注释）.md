# HuggingFace 下载模型 → 自建数据微调（QLoRA）→ vLLM 推理加速（端到端中文注释）

> 里面包括：
>
> - 环境安装与依赖版本建议
> - 从 HuggingFace 拉模型到本地的脚本
> - 构造 JSONL 指令数据与 `DataCollator`（带标签遮盖逻辑）
> - QLoRA 微调脚本（4bit 量化 + LoRA）
> - 合并 LoRA 到基座（便于单权重部署）的脚本
> - 用 vLLM 启 OpenAI 兼容 API 的命令（含“基座+适配器”或“合并后模型”两种方式）
> - Python/HTTP 客户端调用示例
> - 常见报错与调参建议、目录结构示例



> 适用场景：资源有限（单卡 24–48GB 或更少）进行指令微调，并用 vLLM 高吞吐部署。
>  依赖栈：`transformers` + `datasets` + `peft` + `bitsandbytes` + `accelerate` + `vllm`

------

## 0. 环境准备（建议 Conda 新环境）

```bash
# 创建与激活虚拟环境（可选）
conda create -n llm python=3.10 -y
conda activate llm

# 安装核心依赖
pip install -U pip
pip install -U "transformers>=4.42.0" datasets peft accelerate
pip install -U bitsandbytes  # 量化与QLoRA依赖（NVIDIA GPU）

# vLLM（推理加速）
pip install -U "vllm>=0.5.0"  # 如果 CUDA/驱动较新，可用更高版本

# 可选：评测与服务端
pip install -U fastapi uvicorn openai tiktoken
```

> **注意**：需要 NVIDIA 驱动 + CUDA。vLLM 版本与 CUDA/驱动匹配性请参考其官方说明；如遇到编译/依赖冲突，可退回兼容版本。

------

## 1. 下载 HuggingFace 基座模型

> 例子选用轻量中文/多语模型（你也可替换为任意HF模型，如 `Qwen2-1.5B-Instruct`、`Llama-3.1-8B` 等）。

```python
# scripts/00_download_model.py
# 中文注释：使用 from_pretrained 直接拉取权重与分词器到本地缓存。
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"  # 可替换
LOCAL_DIR = "./models/qwen2-1_5b_instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",      # 自动映射到可用 GPU
    torch_dtype="auto"
)
# 将快照保存到本地目录，便于离线/版本固化
model.save_pretrained(LOCAL_DIR)
tokenizer.save_pretrained(LOCAL_DIR)
print("Model & tokenizer saved to", LOCAL_DIR)
```

运行：

```bash
python scripts/00_download_model.py
```

------

## 2. 构建自有指令数据集（JSONL）

> 采用 Alpaca 风格：`{"instruction":..., "input":..., "output":...}`。**面向生成**任务。

```python
# data/build_toy_dataset.py
# 中文注释：生成一个最小可跑通的数据集，实际请替换为你的真实标注。
import json, os
os.makedirs("data", exist_ok=True)
train_path = "data/train.jsonl"
valid_path = "data/valid.jsonl"

train_samples = [
    {"instruction": "解释Transformer中的自注意力", "input": "简述计算流程", "output": "使用Q、K、V计算相似度，缩放点积经softmax得到权重，对V加权求和。"},
    {"instruction": "将这句话翻译为英文", "input": "人工智能正在改变世界。", "output": "Artificial intelligence is changing the world."},
]
valid_samples = [
    {"instruction": "什么是门控循环单元GRU？", "input": "", "output": "GRU通过更新门与重置门控制信息流动，缓解梯度消失。"}
]

with open(train_path, "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(valid_path, "w", encoding="utf-8") as f:
    for s in valid_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print("Wrote", train_path, valid_path)
```

运行：

```bash
python data/build_toy_dataset.py
```

------

## 3. Prompt 模板与数据整理

> 将 `instruction`、`input`、`output` 拼接为模型可学习的**上下文→答案**格式。

**简单模板（支持无 input 情况）**：

```text
### 指令:
{instruction}

### 输入:
{input}

### 回答:
{output}
```

实现为 `DataCollator`：

```python
# scripts/01_data_collator.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

PROMPT = (
    "### 指令:\n{instruction}\n\n"
    "### 输入:\n{input}\n\n"
    "### 回答:\n"
)

@dataclass
class SFTDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8  # 便于 Tensor Core

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 将每条样本拼接为 prompt + answer，并做label遮盖（只计算回答部分loss）。
        input_texts = []
        labels = []
        for ex in features:
            inst = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip() or "无"
            out = ex.get("output", "").strip()

            prompt = PROMPT.format(instruction=inst, input=inp)
            full = prompt + out

            # 编码：注意右侧padding，便于自回归
            ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # labels：prompt部分设为 -100，忽略其loss；答案部分为真实 token id
            label = [-100] * len(prompt_ids) + ids[len(prompt_ids):]

            input_texts.append(ids)
            labels.append(label)

        batch = self.tokenizer.pad(
            {"input_ids": input_texts, "labels": labels},
            padding=True,
            max_length=None,
            return_tensors="pt"
        )
        # 右侧 padding
        batch["attention_mask"] = torch.ones_like(batch["input_ids"]).masked_fill(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0
        )
        return batch
```

------

## 4. 使用 QLoRA 微调（PEFT + bitsandbytes 4bit）

> **QLoRA**：将基座以 4bit 量化加载，仅训练 LoRA 适配器，显著节省显存。

```python
# scripts/02_train_qlora.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from 01_data_collator import SFTDataCollator  # 同目录下导入

MODEL_DIR = "./models/qwen2-1_5b_instruct"   # 第一步下载到的目录
OUTPUT_DIR = "./outputs/qwen2_qlora_sft"     # 微调输出（含adapter）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Tokenizer：确保有 pad_token
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"  # 自回归训练常用右侧padding

# 2) 数据集（JSONL）
print("Loading dataset...")
train_ds = load_dataset("json", data_files="data/train.jsonl")["train"]
valid_ds = load_dataset("json", data_files="data/valid.jsonl")["train"]

# 3) DataCollator：构造 SFT 输入/标签
collator = SFTDataCollator(tokenizer)

# 4) 以 4bit 量化加载模型
print("Loading 4bit model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    load_in_4bit=True,
    torch_dtype="auto",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=None,
    device_map="auto",
    trust_remote_code=True,
)

# 5) QLoRA：准备与配置 LoRA 适配器
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=16,                     # 低秩
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=None,      # None=自动匹配常见线性层；如需精确指定可传列表
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 6) 训练参数（可根据显存/数据规模调整）
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,              # demo；实际可增大
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # 叠加梯度以等效大 batch
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=200,
    save_total_limit=2,
    bf16=True,                       # A100/Hopper 可用；否则用 fp16=True
    fp16=False,
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collator,
)

# 7) 启动训练
trainer.train()

# 8) 保存 adapter（LoRA 权重）与 tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved to", OUTPUT_DIR)
```

运行：

```bash
python scripts/02_train_qlora.py
```

> **提示**：如果你多卡训练，先运行 `accelerate config` 并改用 `accelerate launch scripts/02_train_qlora.py`。

------

## 5. 合并 LoRA 适配器到基座（可选，用于打包单模型部署）

> vLLM 0.4+ 已支持 `--adapter` 直接加载 LoRA；但**合并后**单权重部署更简单，也避免接口不统一。

```python
# scripts/03_merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

BASE = "./models/qwen2-1_5b_instruct"
ADAPTER = "./outputs/qwen2_qlora_sft"
MERGED = "./outputs/qwen2_merged"
os.makedirs(MERGED, exist_ok=True)

# 1) 加载基座（bf16/fp16/float32 按需）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)

# 2) 附加 LoRA，并执行权重合并
peft_model = PeftModel.from_pretrained(base_model, ADAPTER)
peft_model = peft_model.merge_and_unload()  # 将 LoRA 权重合并回线性层

# 3) 保存合并后模型与分词器
peft_model.save_pretrained(MERGED)
AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
print("Merged model saved to", MERGED)
```

运行：

```bash
python scripts/03_merge_lora.py
```

------

## 6. 用 vLLM 启动高吞吐推理服务

> 两种方式：直接加载**合并后模型**，或加载**基座 + 适配器**。

### 方式 A：加载合并后模型

```bash
# 启动 OpenAI 兼容的 API（默认端口 8000）
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/qwen2_merged \
  --dtype auto \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code
```

### 方式 B：加载基座 + LoRA 适配器（无需合并）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/qwen2-1_5b_instruct \
  --adapter ./outputs/qwen2_qlora_sft \
  --dtype auto \
  --max-model-len 4096 \
  --trust-remote-code
```

> 关键参数说明：
>
> - `--max-model-len`：上下文长度；需要与显存匹配。过大将降低并发。
> - `--gpu-memory-utilization`：显存使用上限，避免 OOM。
> - `--tensor-parallel-size`：多卡并行切分张量（=GPU 数量）。
> - vLLM 默认启用 **PagedAttention + 连续批处理**，吞吐高。

------

## 7. 客户端调用（OpenAI SDK 或 HTTP）

### Python（OpenAI 兼容）

```python
# scripts/04_client_openai.py
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen2-local",  # 任意占位；vLLM 会忽略
    messages=[
        {"role": "system", "content": "你是 helpful 的助手。"},
        {"role": "user", "content": "解释一下自注意力为什么要除以sqrt(d_k)。"}
    ],
    temperature=0.2,
    max_tokens=256,
    top_p=0.9
)
print(resp.choices[0].message.content)
```

### 纯 HTTP（curl）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-local",
    "messages": [
      {"role": "system", "content": "你是 helpful 的助手。"},
      {"role": "user", "content": "给我一个Transformer的简短定义。"}
    ],
    "temperature": 0.2,
    "max_tokens": 128
  }'
```

------

## 8. 常见问题与排错

- **显存不足 / OOM**：
  - 降低 `max_model_len`、`batch_size`、`r`（LoRA 低秩），或提高 `gradient_accumulation_steps`；
  - 使用更小 base 模型（如 1.5B / 3B）；
  - vLLM 侧降低 `--gpu-memory-utilization` 并观察吞吐。
- **训练不收敛**：
  - 检查模板与标签屏蔽（prompt 部分 label=-100）；
  - 降低学习率、增加 epoch；
  - 清洗/扩充数据，保证 instruction 与 output 一致性。
- **中文生成断断续续**：
  - 确保分词器与模型匹配；
  - 将 `tokenizer.pad_token = tokenizer.eos_token`；
  - 右侧 padding；
  - 推理时设置 `temperature/top_p` 合理范围。
- **vLLM 启动失败**：
  - 检查 CUDA/驱动、PyTorch 与 vLLM 兼容矩阵；
  - 避免和 `bitsandbytes` 版本冲突；
  - 若使用 `--trust-remote-code`，确保来源安全。

------

## 9. 目录结构建议

```text
project/
├─ data/
│  ├─ train.jsonl
│  └─ valid.jsonl
├─ models/
│  └─ qwen2-1_5b_instruct/       # from_pretrained 本地快照
├─ outputs/
│  ├─ qwen2_qlora_sft/           # 训练得到的 LoRA adapter
│  └─ qwen2_merged/              # 合并后的完整模型（可选）
├─ scripts/
│  ├─ 00_download_model.py
│  ├─ 01_data_collator.py
│  ├─ 02_train_qlora.py
│  ├─ 03_merge_lora.py
│  └─ 04_client_openai.py
└─ HF_QLoRA_vLLM_End2End_Chinese_Annotated.md  # 本说明
```

------

## 10. 进一步优化（按需）

- **更稳的训练**：`cosine` LR、`weight_decay=0.1`、`gradient_checkpointing=True`、`optim="paged_adamw_8bit"`。
- **模板化**：适配 ChatML、Llama 3、Qwen2 官方模板（`tokenizer.apply_chat_template`）。
- **更快推理**：vLLM 设置 `--tensor-parallel-size` 多卡并行；合理的 `--max-num-seqs`、`--enforce-eager` 视情况调优。
- **评估**：加入 `perplexity`、任务指标（ROUGE、EM）与少量人工验收样本。

> 如果你告诉我显卡型号/显存与目标模型大小，我可以把 batch、accum、max_len、LoRA 超参替你**精确化到能跑**。