# 什么是DeepSpeed？	

------

## 1. 什么是 DeepSpeed？

- **提出方**：Microsoft (微软) 在 2020 年开源。
- **定位**：一个 **深度学习训练优化库**，主要用来让超大规模模型（如 GPT、BERT、LLM）能够更快、更省显存、更高效地训练。
- **核心目标**：
  1. **加速** → 提升训练速度。
  2. **省显存** → 能在有限显存里训练更大的模型。
  3. **可扩展** → 支持数百/上千 GPU 的分布式训练。

📌 你可以理解为：**DeepSpeed = PyTorch 的“外挂加速引擎”**，帮你在大模型训练时节省算力、显存，还能扩展到更多 GPU。

------

## 2. DeepSpeed 的核心功能

1. **ZeRO 优化器（Zero Redundancy Optimizer）**
   - 把模型参数、梯度、优化器状态 **切分并分布到多个 GPU**，避免冗余存储。
   - 三个阶段（Stage 1–3）逐步减少显存占用，可实现百亿/千亿参数训练。
2. **混合精度训练（FP16/BF16）**
   - 自动做数值稳定的混合精度，让训练更快、显存更省。
3. **高效并行策略**
   - 数据并行（Data Parallelism）
   - 模型并行（Model Parallelism）
   - 流水并行（Pipeline Parallelism）
   - 张量并行（Tensor Parallelism）
   - **可以组合使用，训练万亿参数模型**。
4. **推理加速**
   - 不仅能加速训练，还能优化推理，减少延迟和显存。

------

## 3. 怎么用 DeepSpeed？

它和 PyTorch 紧密结合，你写 PyTorch 代码，最后只需要 **几行配置** 就能启用 DeepSpeed。

### 3.1 安装

```bash
pip install deepspeed
```

### 3.2 典型训练脚本

假设你有一个 `train.py`（里面是标准 PyTorch 训练代码），启用 DeepSpeed 只要：

```bash
deepspeed train.py --deepspeed_config ds_config.json
```

### 3.3 配置文件示例（ds_config.json）

```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00015,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

解释：

- `fp16.enabled=true` → 开启混合精度训练。
- `zero_optimization.stage=2` → 使用 ZeRO Stage 2 来切分优化器状态和梯度。
- `offload_optimizer` → 把优化器状态放到 CPU，进一步节省显存。

------

## 4. 使用场景

- **训练大模型**：BERT、GPT、T5、LLaMA 等。
- **科研/工业界大规模实验**：在 8–1024 张 GPU 上做分布式训练。
- **资源有限的开发者**：显存不足时，可以用 ZeRO-Offload 在少量 GPU 上跑更大模型。

------

## 5. 和其他工具对比

| 框架                | 主要目标             | 特点                            |
| ------------------- | -------------------- | ------------------------------- |
| **DeepSpeed**       | 大模型训练与推理优化 | ZeRO, 混合精度, 分布式并行      |
| **PyTorch DDP**     | 分布式数据并行       | 简单稳定，但功能较少            |
| **Megatron-LM**     | 超大语言模型训练     | 强调张量并行和流水并行          |
| **Accelerate (HF)** | 简化分布式训练接口   | 上手快，但功能没 DeepSpeed 强大 |

------

✅ **一句话总结**：
 **DeepSpeed = PyTorch 的超强外挂**，核心是 ZeRO 优化器和分布式并行，让你能在有限显存/算力下训练和推理百亿、千亿甚至万亿参数的大模型。



# 那一些量化技术 微调技术，可以使用DeepSpeed？

问得好！DeepSpeed 不只是“加速和省显存”的库，它也在**支持大模型微调和量化**方面下了很多功夫。总结一下：

------

# 一、DeepSpeed 支持的 **微调技术**

1. **全参数微调 (Full Finetuning)**
   - 最传统的方式，更新所有参数。
   - DeepSpeed 提供优化显存的 ZeRO、梯度累积、offload，让**大模型全参微调**也能跑。
2. **LoRA (Low-Rank Adaptation)**
   - 把大模型参数分解成低秩矩阵，只更新小部分。
   - DeepSpeed-Training 已经支持 **LoRA + ZeRO**，结合起来可以在很少显存下对百亿参数 LLM 微调。
3. **Adapter / Prefix-tuning / P-tuning**
   - 在模型层里插入小模块，只训练这些模块。
   - 可以配合 DeepSpeed 的 ZeRO-Offload，把主干冻结、只更新 adapter。
4. **Parameter-Efficient Finetuning (PEFT)**
   - HuggingFace PEFT 库（支持 LoRA、Prefix、Prompt-tuning）可以和 DeepSpeed 一起用。
   - 配置好 `ds_config.json`，HF 的 `Trainer` 就能直接调 DeepSpeed 引擎。

------

# 二、DeepSpeed 支持的 **量化技术**

DeepSpeed 推出了 **DeepSpeed-Inference** 和 **DeepSpeed-Compression**，主要包括：

1. **8-bit / 16-bit 权重量化**
   - 可以在推理时把模型权重压缩到 8-bit，节省一半显存。
2. **ZeroQuant (深度量化技术)**
   - DeepSpeed 团队提出的技术，可以对**百亿规模模型**做高效量化，几乎无精度损失。
   - 特点：一次性量化 (post-training quantization)，支持 8-bit，甚至更低位。
3. **量化 + 微调（QAT/QLoRA）**
   - QLoRA（用 4-bit quantization + LoRA 微调）本身是 HuggingFace 提出的，但 DeepSpeed 也能配合 ZeRO 一起使用，让大模型在低显存显卡（24GB A100 或 40GB RTX）上微调成为可能。
   - 核心思路：**权重 4bit 冻结 + LoRA 插入模块更新**，DeepSpeed 管理显存和优化器。

------

# 三、DeepSpeed 配合微调/量化的典型场景

- **LoRA + ZeRO Stage 3**
   → 在一张 24GB 显卡上微调 70B LLaMA-2 的子集。
- **QLoRA + DeepSpeed**
   → 在消费级 GPU (RTX 3090/4090) 上做大模型指令微调。
- **全参微调 + ZeRO-Offload**
   → 用 CPU/NVMe Offload 技术，在 8 卡 A100 上训练百亿级 GPT。
- **推理加速 (DeepSpeed-Inference)**
   → 部署大模型时结合量化，显存占用能减半，速度翻倍。

------

# 四、总结规律

- **微调方面**：支持全参 + LoRA + Adapter 等各种 PEFT 技术。
- **量化方面**：支持 16bit、8bit、4bit (ZeroQuant, QLoRA)，可用于训练和推理。
- **最佳实践**：
  - 显存紧张：用 **QLoRA + DeepSpeed ZeRO-3**。
  - 多 GPU 集群：用 **全参微调 + ZeRO-Offload**。
  - 部署推理：用 **DeepSpeed-Inference + 8bit/4bit 量化**。



# 也就是DeepSpeed可以和huggingface中的微调技术结合起来，进行训练，然后可以使用vllm进行模型推理？

好嘞！下面给你一套**单卡 24GB 显存**可跑的 **QLoRA + DeepSpeed** 最小可行方案（以 HF Transformers/PEFT 为主，DeepSpeed 做训练器后端）。你只要把模型/数据名字替换成你能访问的即可。

# 0) 环境安装

```bash
pip install -U "transformers>=4.41" "datasets>=2.19" "accelerate>=0.33" \
  "peft>=0.11" "bitsandbytes>=0.43" deepspeed==0.14.4
# 如果是 ROCm/CPU 环境，bitsandbytes 可能不可用；这时换成 8bit/FP16 或者 AutoGPTQ。
```

------

# 1) DeepSpeed 配置（单 GPU 友好）

保存为 `ds_config.json`（单卡不需要 ZeRO-3 分片，Stage-2 足够；QLoRA 的显存主要靠 4-bit 权重 + LoRA）。

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

> 说明：单卡上 ZeRO-2 主要带来梯度/通信优化，真正省显存的是 **4-bit 量化 + LoRA**。若你只有 12GB，也可把 `train_batch_size`/`gradient_accumulation_steps` 调小/调大。
>
> 可以的 👍
>  你这份 `ds_config.json` 是 **DeepSpeed 训练配置文件**，主要用来告诉 DeepSpeed 如何管理批大小、优化器、精度、ZeRO 优化、梯度裁剪等。我逐项帮你解释：
>
> ------
>
> # DeepSpeed 配置参数详解
>
> ```json
> {
>   "train_batch_size": 8,
>   "gradient_accumulation_steps": 8,
>   "optimizer": {
>     "type": "AdamW",
>     "params": { "lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }
>   },
>   "bf16": { "enabled": true },
>   "fp16": { "enabled": false },
>   "zero_optimization": {
>     "stage": 2,
>     "allgather_partitions": true,
>     "reduce_scatter": true,
>     "overlap_comm": true
>   },
>   "gradient_clipping": 1.0,
>   "wall_clock_breakdown": false
> }
> ```
>
> ------
>
> ## 1. `train_batch_size`
>
> - **作用**：全局 batch size（不是单卡的 batch size，而是经过所有 GPU + 累积之后的总 batch size）。
> - **这里**：设为 8 → 整个训练过程每次更新的有效 batch 是 8 条样本。
> - 如果你有 1 张卡、每卡 batch=1，同时 `gradient_accumulation_steps=8`，那么有效 `train_batch_size = 1 × 8 = 8`，和这里保持一致。
>
> ------
>
> ## 2. `gradient_accumulation_steps`
>
> - **作用**：梯度累积的步数。
> - **好处**：用小 batch（省显存）模拟大 batch（稳定训练）。
> - **这里**：设为 8 → 每 8 次 forward/backward 后再做一次优化器更新。
> - 例子：如果 `per_device_train_batch_size=1`，实际等效 batch size = 1 × 8 = 8。
>
> ------
>
> ## 3. `optimizer`
>
> - **type**: `"AdamW"` → 优化器类型（AdamW 是 Transformer 系列常用）。
> - **params**:
>   - `lr: 2e-4` → 学习率。
>   - `betas: [0.9, 0.999]` → Adam 的一阶/二阶动量参数。
>   - `eps: 1e-8` → 数值稳定项，防止除零。
>   - `weight_decay: 0.01` → 权重衰减（L2 正则）。
>
> ------
>
> ## 4. `bf16` 与 `fp16`
>
> - **bf16.enabled=true** → 使用 bfloat16 精度训练（需要 A100/H100 或 4090 这种支持 BF16 的 GPU）。
> - **fp16.enabled=false** → 不用传统半精度 FP16。
> - **区别**：
>   - FP16 范围小，容易数值溢出。
>   - BF16 动态范围和 FP32 一样，稳定性好，推荐优先用。
>
> ------
>
> ## 5. `zero_optimization`
>
> DeepSpeed 的核心：**ZeRO (Zero Redundancy Optimizer)**，用来节省显存。
>
> - `"stage": 2`
>   - **Stage 1**: 分片优化器状态。
>   - **Stage 2**: 额外分片梯度。
>   - **Stage 3**: 连模型参数也分片（最省显存，但通信更多）。
>   - 这里用 Stage 2，是性能和显存的平衡方案。
> - `allgather_partitions: true`
>   - 在需要时动态聚合参数，而不是始终存一份副本。
> - `reduce_scatter: true`
>   - 在反向传播时用 reduce-scatter 操作，把梯度拆分后再分发，减少通信量。
> - `overlap_comm: true`
>   - 通信和计算并行进行，提高效率。
>
> ------
>
> ## 6. `gradient_clipping`
>
> - **作用**：梯度裁剪阈值。
> - **这里**：`1.0` → 如果梯度范数超过 1，就缩放到 1。
> - **目的**：防止梯度爆炸，尤其在大模型训练时很重要。
>
> ------
>
> ## 7. `wall_clock_breakdown`
>
> - **作用**：是否开启详细的性能分析。
> - **这里**：`false` → 默认不开启。
> - 如果设为 `true`，DeepSpeed 会输出通信/计算/等待的耗时，用于性能调优。
>
> ------
>
> # 总结（这份配置的意义）
>
> - **batch 策略**：用梯度累积（8 步）来模拟大 batch=8。
> - **优化器**：AdamW，学习率 2e-4。
> - **精度**：启用 BF16（稳定，省显存），不用 FP16。
> - **ZeRO**：Stage 2，分片优化器状态+梯度，通信优化打开。
> - **稳定性**：梯度裁剪=1.0。
> - **调试**：不开 wall clock breakdown。
>
> 这份配置属于一个**“中等规模模型，单机或少量 GPU，显存有限但追求训练效率”的通用方案**。
>
> 

------

# 2) 训练脚本（QLoRA + DeepSpeed）

保存为 `finetune_qlora_ds.py`。默认用一个开源可直接拉取的模型（如 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 便于示范；你可以换成 `meta-llama/Llama-2-7b-hf`、`meta-llama/Llama-3-8B` 或 `Mistral-7B-Instruct` 等你有权限的模型）。

```python
# -*- coding: utf-8 -*-
import os
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --------- 0. 配置 ----------
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # 换成你能访问的
DATA_NAME  = os.getenv("DATA_NAME",  "yelp_polarity")  # 示例数据；你也可以改成本地 JSON/CSV
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "out-qlora-ds")
USE_BF16   = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # A100/4090 支持 bfloat16

# --------- 1. 量化配置 (QLoRA 关键) ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # QLoRA 推荐 nf4
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16
)

# --------- 2. 模型 & 分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备 LoRA：常见 LLaMA/Mistral 线性层名称
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# 使模型适配 k-bit 训练并注入 LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------- 3. 数据集 ----------
# 示例：用 yelp_polarity 做指令微调式的 SFT（把分类数据拼成指令-回答风格文本）
ds = load_dataset(DATA_NAME)

def format_example(example):
    # 把 (text, label) 转成一个聊天/指令训练样本；实际项目换成你的 SFT 格式
    label_map = {0: "negative", 1: "positive"}
    prompt = f"Classify the sentiment of the following review as positive or negative.\n\nReview: {example['text']}\nAnswer:"
    return {"text": prompt + " " + label_map[int(example["label"])]}

train_data = ds["train"].select(range(20000)).map(format_example, remove_columns=ds["train"].column_names)
eval_data  = ds["test"].select(range(1000)).map(format_example,  remove_columns=ds["test"].column_names)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=1024, padding="max_length")
train_tok = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
eval_tok  = eval_data.map(tokenize,  batched=True, remove_columns=eval_data.column_names)

# causal LM 的 collator （不使用 MLM）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------- 4. 训练参数（启用 DeepSpeed） ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # 24GB 基本可承受；不足就设为 1 再用更多梯度累积
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,     # 有效 batch size = 1*8=8
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,       # 进一步省显存
    deepspeed="ds_config.json",        # ★ 关键：接入 DeepSpeed
    report_to="none"
)

# --------- 5. Trainer ----------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# --------- 6. 导出 (合并 LoRA 或保持 Adapter) ----------
# 方式A：只保存 LoRA adapter（体积小，推理时再加载到基座模型）
adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

# 方式B：合并 LoRA 权重到 16/32bit 基座（用于无 PEFT 推理；需有足够内存）
# from peft import merge_and_unload
# merged = merge_and_unload(trainer.model)
# merged = merged.to(dtype=torch.float16)
# merged.save_pretrained(os.path.join(OUTPUT_DIR, "merged_fp16"))
# tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "merged_fp16"))

print("Done. Adapters saved to:", adapter_dir)
```

运行：

```bash
python finetune_qlora_ds.py
# 或者用 DeepSpeed 启动（单卡/多卡都行）：
deepspeed finetune_qlora_ds.py --deepspeed ds_config.json
```

------

# 3) 推理（加载 LoRA 适配器 + 4-bit 基座）

保存为 `inference_qlora.py`：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # 你的基座
ADAPTER = "out-qlora-ds/lora_adapter"         # 上一步保存的适配器

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)

prompt = "Classify the sentiment of the following review as positive or negative.\n\nReview: This phone is amazing and the battery lasts forever.\nAnswer:"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

------

# 4) 用你自己的数据

- **自定义 SFT 文本**：如果你已有指令-回答式 JSON/JSONL（字段如 `{"prompt":"...", "response":"..."}`），把“第 3 部分数据处理”换成拼接 `prompt + response` 即可。
- **CSV/TSV**：用 `load_dataset("csv", data_files=...)` 读入，然后像上面一样 `map(format_example) → tokenize`。
- **中文/多语**：只要基座模型支持，对脚本无需改动。

------

# 5) 常见调参建议（24GB 单卡）

- OOM：把 `per_device_train_batch_size` 设为 1，`gradient_accumulation_steps` 调大到 16/32；`max_length` 设 1024 或更短。
- 训练不稳定：把 `lr` 调到 `1e-4 ~ 5e-5`，或把 `r` 从 16 改 8（LoRA 更小）。
- 速度慢：关掉 `evaluation_strategy` 或调大 `eval_steps`/`save_steps`；减小数据量做先跑通。
- 合并导出：需要在 FP16/FP32 下合并（可能需要 30GB+ CPU 内存），合并后部署更简单。

------

## 一句话总结

- **QLoRA** 用 4-bit 冻结基座 + 训练 LoRA 适配器，显存极省；
- **DeepSpeed** 作为训练后端（ZeRO-2、优化通信/梯度、调度器），稳定高效；
- **这套脚本**：**单卡 24GB** 即可跑 7B 量级模型的小规模 SFT，按需替换模型/数据就行。
   需要我帮你把数据格式（CSV/JSONL）对接到上面的 `format_example` 吗？直接贴几行样例，我来改成你能跑的版本。



# 什么是SFT数据？

你常听到的 **SFT 数据**，其实就是 **Supervised Fine-Tuning data（监督微调数据）**。  

---

## 1. 什么是 SFT 数据？
- **来源**：在大模型（LLM）训练流程里，SFT 指的是 **在预训练模型的基础上，用人工标注的「指令-回答」数据进行监督微调**。  
- **目的**：让模型学会按照人类需求完成具体任务，而不是只会“预测下一个 token”。  
- **本质**：一份 **成对的「输入（指令/提示）→ 输出（回答/目标）」数据集**。  

---

## 2. SFT 数据的常见格式
一般是 JSON/JSONL 或者 CSV，每条数据包含：
- `instruction`（用户的任务指令）
- `input`（可选，补充信息，比如一段文本）
- `output`（模型需要生成的目标答案）

### 示例 1（单纯问答）
```json
{
  "instruction": "请翻译以下句子成英文",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
```

### 示例 2（分类）
```json
{
  "instruction": "判断下面评论的情感是正面还是负面",
  "input": "这家餐厅的菜很难吃",
  "output": "负面"
}
```

### 示例 3（多轮对话 / Chat 格式）
也有数据是用 `messages` 存储对话：
```json
{
  "messages": [
    {"role": "user", "content": "写一首五言绝句，主题是春天"},
    {"role": "assistant", "content": "春风吹绿柳，燕子舞晴空。桃花映流水，芳香满长东。"}
  ]
}
```

---

## 3. 为什么要做 SFT？
- **预训练模型**（像 GPT、LLaMA）只会语言建模：学会“预测下一个词”。  
- **经过 SFT**：模型学会“听懂人话、照指令完成任务”。  
- **后续 RLHF（奖励模型 + 强化学习）**：进一步让模型输出更符合人类偏好。  

---

## 4. SFT 数据的来源
1. **人工标注**（最贵、最精确）  
   - 专业人员写指令和答案。  
2. **现成公开数据集**  
   - Alpaca、ShareGPT、Dolly、BELLE 等。  
3. **合成数据**  
   - 用强模型（比如 GPT-4）生成问答对，作为弱模型的 SFT 数据。  

---

✅ **一句话总结**：  
**SFT 数据 = 人类监督标注的「指令 → 答案」数据集，用来教大模型听懂任务、按需求回答。**  

---

