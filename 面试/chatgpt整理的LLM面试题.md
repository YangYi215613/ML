# LLM 面试 120 问（速背版，含代码片段 / 中文注释）

> 目的：覆盖 LLM 常见高频考点，给出**可背诵的精炼答案**与必要的小代码。按主题分组，≈ 120 题左右。

------

## A. 架构基础（Transformer）

**1. 什么是 Transformer？**
 基于**自注意力**的序列建模架构，摒弃 RNN/CNN；由多层 **(多头自注意力 + 前馈网络 + 残差 + 归一化)** 组成，支持**全并行**与长依赖建模。

**2. Q/K/V 分别是什么？**
 Q=查询、K=键、V=值；通过 `softmax(QK^T/√d_k)` 得到关联权重，对 V 加权汇聚形成上下文表征。

**3. 为什么要除以 √d_k？**
 点积随维度增大方差变大，缩放可稳定 softmax 的梯度与数值范围，避免极端饱和。

**4. 多头注意力的作用？**
 并行在不同子空间建模关系，捕捉多种模式；提升表达能力与稳定性。

**5. 位置编码有哪些？优缺点？**
 绝对正弦（可插值、零参数，外推一般）、可学习绝对（灵活但外推更差）、相对位置（迁移好）、**RoPE**（长上下文稳）、**ALiBi**（长度外推友好、实现简单）。

**6. Pre-LN vs Post-LN？**
 Pre-LN（子层前归一化）更易训练深层、收敛稳；Post-LN 原版较难训深层。

**7. LayerNorm vs RMSNorm？**
 LN 归一化均值+方差；RMSNorm 去均值只保留方差缩放，计算更省，性能近似。

**8. FFN 为什么扩张 4×hidden？**
 提升逐位置非线性变换容量；经验上性价比好（亦常用门控激活如 SwiGLU/GEGLU）。

**9. 残差连接的作用？**
 缓解梯度消失，促进跨层信息流动，稳定深网络训练。

**10. Decoder 的 Masked Self-Attn 做什么？**
 用**上三角因果掩码**，保证自回归只看历史 token，避免信息泄漏。

**11. 三类结构：Encoder-only / Decoder-only / Encoder–Decoder？**
 BERT（理解）、GPT（生成）、T5/BART（Seq2Seq：翻译/摘要）。

**12. O(L²) 复杂度怎么优化？**
 **FlashAttention**（访存优化且精确）、稀疏/局部注意力、低秩/核化（Performer/Linformer）、滑窗（Mistral）、**KV Cache**、**MQA/GQA**、**PagedAttention**。

**13. KV Cache 原理？**
 推理时缓存历史 K/V，后续步只计算新 token 与缓存交互，减少重复计算与显存拷贝。

**14. MQA/GQA 好处？**
 共享或分组共享 KV，显著减少 KV Cache 体积，提高并发吞吐。

**15. FlashAttention 的要点？**
 块状计算 + 精确重排访存，避免显存中间态，**精确注意力**同时提升速度/降低显存。

------

## B. 训练与优化

**16. 常用优化器与调度？**
 AdamW + 线性/余弦调度；**warmup(1–3%)**；梯度裁剪；权重衰减 0.1（常见）。

**17. 混合精度 FP16 vs BF16？**
 BF16 数值稳定性更好、无需 loss scale；A100/Hopper 上推荐 BF16。

**18. 并行策略有哪些？**
 DDP（数据并行）、ZeRO(1–3)（切分优化器状态/梯度/参数）、TP（张量并行）、PP（流水线并行）。

**19. 分词与语料处理？**
 BPE/Unigram(SentencePiece)；去重、去毒、质量过滤；温度采样混合多源语料。

**20. 学习率选择？**
 与 batch/模型大小相关：大模型常用 1e-4 ~ 2e-5；配合 warmup 与余弦退火。

**21. 损失与正则？**
 交叉熵；可加 label smoothing；dropout、激活正则、字典去重等。

**22. 避免灾难性遗忘？**
 小 LR、混合原任务数据、正则约束、累积更新、冻结底层。

**23. 梯度检查点（activation checkpointing）？**
 减少中间激活保存，换算力为显存，适合深层大模型训练。

**24. 梯度累积与有效 batch？**
 小显存下用 `gradient_accumulation_steps` 以等效大 batch 训练。

**25. 词嵌入权重 tied 的意义？**
 共享输入/输出嵌入，减少参数、对齐表征空间，常见于 LM。

**26. 训练稳定性关键点？**
 Pre-LN/RMSNorm、合适的 init、grad norm 限制、FP16 时用 loss scale 或切 BF16。

**27. 如何计算困惑度 PPL？**
 `PPL = exp(平均负对数似然)`；越低越好。

------

## C. 微调与对齐（SFT / RLHF / DPO / PEFT）

**28. SFT（指令微调）流程？**
 准备指令数据→模板构造→仅对**答案区**计 loss（prompt label=-100）→小 LR 微调。

**29. RLHF 三阶段？**
 SFT→训练 Reward Model(RM)→PPO 优化；对齐人类偏好但实现复杂、成本高。

**30. DPO 是什么？优缺点？**
 Direct Preference Optimization：基于 (chosen,rejected) 直接优化，无需 RM/PPO；实现简洁、稳定但依赖偏好数据质量。

**31. KTO/ORPO 简述？**
 均为偏好学习的简化/鲁棒变体：KTO 使用已标注偏好；ORPO 在 CE 上加偏好正则项。

**32. LoRA 原理？**
 把权重增量近似为低秩 AB，只训练 A/B 小矩阵；显存开销小、效果稳。

**33. QLoRA 流程？**
 4bit 量化加载基座（NF4+double quant）→ 训练 LoRA 适配器 → 选择性**合并**权重部署。

**34. Adapter/Prefix/Prompt/IA³ 对比？**
 Adapter 在层内插瓶颈；Prefix 在注意力加虚拟 KV；Prompt 在输入侧拼可训练提示；IA³ 在通道维乘缩放向量；均为参数高效方式。

**35. 什么时候选全参 vs PEFT？**
 数据多/预算足→全参；资源紧/需求快→LoRA/QLoRA/IA³；多任务共享→Adapter/Prefix。

**36. 冻结底层只训顶层的价值？**
 降低过拟合与显存，适用于窄域任务；但上限可能受限。

**37. 多模板/多任务混训？**
 统一到“文本到文本”，模板规范化；注意不同损失权重与采样温度。

**38. LoRA 权重是否必须合并？**
 非必须：推理端可直接加载基座+adapter；合并便于单权重分发/部分加速。

**39. 安全与对齐要点？**
 拒答策略、越狱防护、提示注入防御、输出过滤与审计、红队评测。

**40. 评测集？**
 PPL、MMLU、MT-Bench、Arena、HumanEval、GSM8K、LongBench 等；结合人评。

**41. 数据合成（Self-Instruct）？**
 用强模型生成指令-答案对，配以过滤与去重，扩充 SFT 数据。

------

## D. 解码与推理

**42. 常见解码策略？**
 Greedy（确定）、Beam（确定性更强）、Top-k/Top-p（多样性）、温度、重复惩罚、**Mirostat**（控制困惑度）。

**43. 长上下文扩展方法？**
 RoPE 插值、ALiBi、继续预训练到更长上下文、滑窗注意力、外部记忆/RAG。

**44. vLLM 优势？**
 **PagedAttention + 连续批处理**，高并发/高吞吐；支持 LoRA 适配器加载、OpenAI 兼容 API。

**45. TensorRT-LLM 要点？**
 图优化+算子融合+量化/稀疏，低时延高吞吐，部署到生产常用。

**46. 吞吐调优思路？**
 控制 `max_model_len`、批大小、并发；KV 量化、MQA/GQA；合适的并行度与显存使用率。

**47. 何为重复惩罚与长度惩罚？**
 重复惩罚提高已生成 token 的logit温度；长度惩罚影响 Beam 的长度偏好。

**48. 结构化输出/函数调用如何做？**
 使用 schema 约束/正则、`tool`/`function_call` 协议，或**约束解码**（如 JSON 语法约束）。

**49. Speculative/Assisted Decoding？**
 小草稿模型先生成候选，大模型快速验证与接纳；可显著提速。

**50. 批量推理注意？**
 尽量**右侧 padding**；同批次长度相近；控制 eos/stop tokens 以稳定停机。

**51. 模板与角色规范？**
 遵循模型官方 chat 模板（system/user/assistant），减少越狱与格式偏移。

**52. Token 计数与截断？**
 注意 `max_new_tokens` vs `max_length`；提示+输出合计不能超上下文长度。

------

## E. RAG（检索增强生成）

**53. 什么是 RAG？**
 在生成前检索外部知识，作为上下文供模型引用，降低幻觉、增强时效性。

**54. 切块与窗口策略？**
 按语义/段落切块，重叠窗口（20–30%），保证跨段语义连续性。

**55. 检索器选择？**
 BM25（稀疏）、Dense（向量）、Hybrid（组合）；多数生产用 Hybrid + 重排序。

**56. 嵌入模型选择？**
 多语种/中文优先选多语嵌入；维度与延迟权衡；避免与 LLM 冲突的 tokenizer。

**57. 向量库与相似度？**
 Faiss/ScaNN/Milvus/PGVector；内积/余弦；注意归一化与 MIPS 技巧。

**58. 重排序（Rerank）？**
 Cross-Encoder 提升相关性；K→K'；成本更高但显著提升最终命中。

**59. 查询改写？**
 多查询扩展（Multi-Query）与提示重写，覆盖不同问法与子主题。

**60. 幻觉与引用？**
 要求**逐段引用**与来源标注；对未命中强制拒答或回退一般知识。

**61. 评估？**
 Retriever 召回@k、重排序准确率、端到端 F1/ExactMatch、人评可读性与事实性。

------

## F. MoE（专家混合）与前沿

**62. MoE 原理？**
 稀疏激活：路由器为每个 token 选择少量专家（Top-1/2），等价 FLOPs 下提升容量。

**63. 负载均衡如何做？**
 加均衡损失（aux loss）、容量限制（capacity factor）、随机抖动，避免专家失衡。

**64. 通信瓶颈？**
 All-to-All 成本高；需 Expert Parallel + 通信优化（混合并行、流水化）。

**65. 稳定训练技巧？**
 路由温度/噪声、损失权重、预热、drop tokens、专家正则。

------

## G. 量化与蒸馏

**66. 后训练量化（PTQ）？**
 GPTQ/AWQ/AQLM：在不改动训练的前提下量化权重（INT8/4），需校准集、误差感知。

**67. KV Cache 量化？**
 对生成吞吐影响大；常用 8/6/4bit，需评估质量回退与稳定性。

**68. 蒸馏思路？**
 用 Teacher 预测/中间特征监督 Student；保真度高、时延小；可与量化结合。

**69. 量化感知训练（QAT）？**
 在训练中模拟量化噪声，提高量化后精度，但训练成本更高。

------

## H. 经典速答

**70. 为什么 Transformer 并行性强？**
 去掉循环，序列位置可同时计算注意力；GPU 友好。

**71. Self-Attn vs Cross-Attn？**
 Self：Q/K/V 同一序列；Cross：Q 来自解码器，K/V 来自编码器/检索上下文。

**72. SFT 为何要把 prompt 标签设为 -100？**
 只训练“答案段”，避免模型学会去预测提示本身。

**73. 什么是 Perplexity？**
 指数化的平均负 log-likelihood，数值越低越好。

**74. 为什么要 warmup？**
 初期梯度不稳定，warmup 有助于稳态进入。

**75. 如何降低幻觉？**
 RAG、链式思维验证、工具调用、结构化约束、拒答/澄清策略与后验辨别器。

**76. 数据泄漏如何避免？**
 划分严格、去除测试重合、哈希去重、外部知识截断于训练时间点。

**77. 为什么使用 RMSNorm 越来越多？**
 数值稳定、计算省、易扩到深层；效果与 LN 接近。

**78. SwiGLU/GEGLU 为什么常见？**
 门控激活提高 FFN 表达力与梯度传输效率，带来小幅收益。

------

## I. 关键代码片段（最小可背）

**1) Scaled Dot-Product Attention（PyTorch 伪代码，含因果掩码）**

```python
import torch, math

def attention(Q, K, V, mask=None):
    # Q,K,V: [B,h,L,d]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # [B,h,L,L]
    if mask is not None:  # mask: 1=keep, 0=mask, shape broadcastable to [B,h,L,L]
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V  # [B,h,L,d]
```

**2) 自回归解码（Greedy / Top-p）**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m_id = "Qwen/Qwen2-1.5B-Instruct"
mdl = AutoModelForCausalLM.from_pretrained(m_id, device_map="auto", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained(m_id)

prompt = "用一句话解释注意力机制。"
ids = tok(prompt, return_tensors="pt").to(mdl.device)

with torch.no_grad():
    out = mdl.generate(
        **ids, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.7, eos_token_id=tok.eos_token_id
    )
print(tok.decode(out[0], skip_special_tokens=True))
```

**3) LoRA 微调最小示例（PEFT）**

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

base = AutoModelForCausalLM.from_pretrained(m_id, device_map="auto", torch_dtype="auto")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(base, lora)

# 假设 train_ds 已准备好 (input_ids, attention_mask, labels)
trainer = Trainer(model=model, args=TrainingArguments("out_lora", per_device_train_batch_size=2, num_train_epochs=1))
trainer.train(); model.save_pretrained("out_lora")
```

**4) QLoRA 4bit 加载片段**

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    m_id, load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", device_map="auto"
)
```

**5) vLLM 启动与客户端**

```bash
# 服务器：加载合并后模型或 --adapter 方式
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/merged \
  --max-model-len 4096 --gpu-memory-utilization 0.92 --dtype auto --trust-remote-code
# 客户端（OpenAI 兼容）
from openai import OpenAI
cli = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
r = cli.chat.completions.create(model="local", messages=[{"role":"user","content":"一句话解释RoPE"}])
print(r.choices[0].message.content)
```

**6) 计算 PPL（困惑度）**

```python
import math, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

mdl = AutoModelForCausalLM.from_pretrained(m_id, device_map="auto", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained(m_id)

ds = load_dataset("json", data_files="data/valid.jsonl")["train"]
losses = []
for ex in ds:
    t = tok(ex["output"], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl(**t, labels=t["input_ids"])  # 自回归: 预测自身
    losses.append(out.loss.item())
print("PPL=", math.exp(sum(losses)/len(losses)))
```

------

## J. 面试“连环追问”速答模板

- **Q：为什么 O(L²) 是瓶颈？如何解决？**
   A：注意力要对每对位置算相似度，L²；用 FlashAttention/稀疏/低秩/滑窗、KV Cache、MQA/GQA、分块计算与分页缓存优化。
- **Q：RoPE 与 ALiBi 何时选？**
   A：RoPE 适合已训练到较长长度场景，插值/继续训更稳；ALiBi 零改权重外推友好、实现简单。
- **Q：DPO 与 PPO 的取舍？**
   A：DPO 无需 RM 与在线 RL，易训、稳定；PPO 可控性强但复杂昂贵。
- **Q：LoRA 目标层怎么选？**
   A：优先 Attention 的投影层（q_proj/k_proj/v_proj/o_proj）和 FFN 的 in/out；低秩 r=8~64 视任务规模。
- **Q：如何落地 RAG 降幻觉？**
   A：优质切块+Hybrid 检索 + Cross-Encoder rerank + 引用段落 + 失败时拒答/追问。
- **Q：为什么 BF16 常优于 FP16？**
   A：动态范围更大，避免梯度 underflow，无需损失缩放，稳定性更好。

------

