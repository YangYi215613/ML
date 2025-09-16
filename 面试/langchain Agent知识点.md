# 📌 LangChain 智能体开发速查表

## 1. 基础核心概念

- **PromptTemplate**：提示词模板，定义输入 → 输出格式。
- **LLM / ChatModel**：统一接口封装不同大模型（OpenAI、Qwen、Baichuan、DeepSeek 等）。
- **OutputParser**：把 LLM 输出结果转成结构化数据（JSON、对象等）。

------

## 2. 链（Chains）

- **LLMChain**：最基础链（Prompt → Model → Output）。
- **SequentialChain**：多个步骤依次执行。
- **ConversationChain**：带记忆的对话。

👉 常见面试题：
 **Q:** 为什么需要 Chain？
 **A:** 因为 LLM 本身是黑盒，`通过 Chain 可以组合不同模块（LLM、工具、记忆），实现复杂流程的封装和复用。

------

## 3. 记忆（Memory）

- `ConversationBufferMemory`：保存所有对话记录。
- `ConversationBufferWindowMemory`：只保存最近 k 轮对话。
- `VectorStoreRetrieverMemory`：基于向量检索的记忆（适合长上下文）。

👉 常见面试题：
 **Q:** Memory 和 RAG 有什么区别？
 **A:** Memory 偏向于短期对话上下文存储，而 RAG 是从外部知识库检索信息。前者解决对话连贯性，后者解决知识覆盖不足。

------

## 4. RAG（检索增强生成）

- **步骤：** 文本加载 → 切分 → 向量化 → 存入向量库 → 检索 → 拼接上下文 → 交给模型回答。
- **相关模块：**
  - `DocumentLoaders`（PDF、网页等）
  - `TextSplitter`（RecursiveCharacterTextSplitter 等）
  - `VectorStore`（FAISS、Pinecone、Milvus…）
  - `RetrievalQA`（LangChain 封装好的 QA Pipeline）

👉 常见面试题：
 **Q:** RAG 为什么比单纯调用 LLM 更可靠？
 **A:** 因为它把知识限制在特定语料里，减少模型幻觉（hallucination），还能更新外部知识而不依赖模型重新训练。

------

## 5. 工具调用（Tool Use / Function Calling）

- **Tools**：封装外部 API（搜索、数据库、天气查询…）。
- **Function Calling**：部分模型（如 OpenAI GPT、Qwen）支持函数调用模式，能直接输出结构化参数调用 API。

👉 常见面试题：
 **Q:** 如何在 LangChain 里接入一个外部 API？
 **A:** 用 `Tool` 封装 API，指定 `name`、`description` 和 `func`，再把 Tool 交给 Agent 调度。

------

## 6. 智能体（Agent）

- **AgentExecutor**：运行智能体。
- **常见类型：**
  - ReAct Agent（推理 + 行动循环）
  - Conversational Agent（多轮对话）
  - Plan & Execute（先规划，再执行）
- **LangGraph**：LangChain 新模块，支持更复杂的多 Agent 协作和流程编排。

👉 常见面试题：
 **Q:** Agent 和 Chain 的区别？
 **A:** Chain 是固定流程，Agent 是动态决策。Chain 适合确定步骤的任务，Agent 适合需要模型自主选择工具或规划步骤的任务。

------

## 7. Prompt 工程

- **技巧：**
  - Few-shot Prompt（给几个示例）
  - Chain-of-thought（引导模型逐步推理）
  - ReAct Prompt（推理 + 行动）
  - 输出格式约束（JSON schema 等）

👉 常见面试题：
 **Q:** 如何减少模型输出格式混乱的问题？
 **A:** 可以在 Prompt 中明确要求输出 JSON，配合 `OutputParser` 解析；如果模型支持 Function Calling，可以直接用结构化调用。

------

## 8. 系统与优化

- **性能优化**：缓存（LangChain Cache）、批处理、异步调用。
- **成本优化**：减少上下文长度、使用 Embedding 检索替代大模型调用。
- **安全性**：Prompt 注入防护、工具调用安全校验、权限管理。

------

# 🎯 面试答题小技巧

1. **框架对比**
   - LangChain：模块丰富，生态大，但稍显重量。
   - LlamaIndex：专注于 RAG 和文档问答，简单轻量。
   - Haystack：企业级，偏 NLP pipeline。
2. **示例项目**（最好准备一个 demo）
   - 场景：上传 PDF 文档，用户提问 → 系统检索相关内容 → 大模型总结回答。
   - 加分项：能说自己实现过 **RAG + 工具调用**（比如还能查实时天气/数据库）。



------

# 📌 LangChain 智能体（Agent）超详细速查表

## 1. 什么是智能体（Agent）

- **定义**：一种基于大模型（LLM）的任务执行框架，模型不仅生成回答，还能 **调用工具、记忆、规划步骤**，解决复杂任务。
- **关键点**：Chain 是固定流程，而 Agent 是**动态决策**，会根据输入决定调用哪个工具、执行多少次循环。

👉 面试答法：

> “Agent 的核心就是把 LLM 当作推理引擎，它能结合工具、记忆、环境，动态完成复杂任务，而不仅是简单的文本生成。”

------

## 2. LangChain 中 Agent 的组成部分

1. **LLM**：推理核心。
2. **Tools**：外部能力（API、数据库、Python 执行器等）。
3. **Memory**：存储上下文（对话历史、向量知识库）。
4. **Agent Executor**：负责运行循环（Reason → Act → Observe）。

👉 面试答法：

> “LangChain 的 Agent 就像一个执行器：输入问题 → LLM 决策 → 调用工具 → 观察结果 → 继续决策 → 最终输出答案。”

------

## 3. Agent 的类型

LangChain 提供了不同种类的智能体，常见有：

1. **ZeroShotAgent**
   - 不需要示例，直接基于工具描述来调用。
   - 使用场景：输入明确、工具描述清晰的情况。
2. **ConversationalAgent**
   - 结合对话历史，适合多轮交互。
   - 使用场景：智能客服、助手类应用。
3. **ReAct Agent**
   - 经典模式：**推理（Reasoning）+ 行动（Action）+ 观察（Observation）**
   - 循环执行直到得到最终答案。
   - 使用场景：复杂任务拆解、搜索问答。
4. **Plan and Execute Agent**
   - 两步走：先用 LLM 生成计划（Plan），再执行（Execute）。
   - 使用场景：任务较长、需要步骤规划的情况（比如旅行规划）。
5. **Custom Agent（自定义智能体）**
   - 基于 `AgentExecutor` 自己实现。
   - 使用场景：需要高度定制的业务流程。

------

## 4. Tools（工具）

- **定义**：一个函数接口，LLM 可以调用它完成特定任务。
- **内置工具示例**：
  - `SerpAPI`（搜索）
  - `SQLDatabase`（数据库查询）
  - `PythonREPL`（运行 Python 代码）
  - `VectorStore`（知识检索）

👉 面试答法：

> “在 LangChain 中，工具是 Agent 的手臂。LLM 本身不会算，但通过 Tools，它能查数据库、调 API、运行代码，从而解决更复杂的问题。”

------

## 5. Agent 工作流

典型执行循环（ReAct 思路）：

1. 用户输入问题。
2. LLM 分析 → 生成推理步骤（Reasoning）。
3. 选择工具（Action）。
4. 执行工具 → 得到结果（Observation）。
5. LLM 根据结果继续推理。
6. 最终生成答案（Final Answer）。

👉 面试答法：

> “Agent 的本质就是不断循环 ReAct：Reason → Act → Observe，直到得到最终答案。”

------

## 6. 记忆（Memory）

- **ConversationBufferMemory**：保存全部对话历史。
- **ConversationBufferWindowMemory**：只保存最近 k 轮。
- **ConversationKGMemory**：知识图谱形式的记忆。
- **VectorStoreRetrieverMemory**：结合向量检索的长期记忆。

👉 面试答法：

> “Memory 让 Agent 保持状态，避免每次从零开始。比如客户咨询多轮问题时，Agent 能记住之前的上下文。”

------

## 7. 高级功能

1. **多 Agent 协作**
   - 多个 Agent 分工合作（比如一个做规划，一个执行）。
   - LangChain 的新模块 **LangGraph** 专门支持这一点。
2. **输出控制**
   - 通过 `OutputParser` 保证输出格式（JSON、结构化数据）。
3. **安全与防护**
   - 防止 Prompt 注入（限制工具调用范围）。
   - 对工具调用结果做校验。

------

## 8. 常见面试问题与参考回答

**Q1:** Agent 和 Chain 的区别？
 👉 A：Chain 是固定流程，适合结构化任务；Agent 是动态决策，能根据输入和上下文灵活选择工具和步骤。

**Q2:** 说一下 ReAct Agent 的执行逻辑？
 👉 A：ReAct = Reasoning + Acting。模型会先推理，再选择工具执行，观察结果后继续推理，形成一个循环，直到得到答案。

**Q3:** 如何在 LangChain 中实现工具调用？
 👉 A：定义一个 Tool（包含 name、description、func），把它交给 AgentExecutor，LLM 就能按需调用它。

**Q4:** 如果模型输出格式混乱，怎么处理？
 👉 A：可以在 Prompt 中加格式约束，用 `OutputParser` 解析；更好的方式是用支持 Function Calling 的模型（如 OpenAI、Qwen）。

**Q5:** 在企业级场景里，怎么保证 Agent 的稳定性？
 👉 A：可以做三点：

1. 工具调用做权限与安全校验；
2. 对 LLM 输出做格式约束；
3. 加缓存与日志监控，提升性能和可追踪性。

------

## 9. 你可以背的总结性语句

- “LangChain 的 Agent 就是让大模型不只是回答问题，还能调用工具、存储记忆、动态规划任务。”
- “常见 Agent 类型有 ZeroShot、Conversational、ReAct、Plan & Execute，其中 ReAct 是最经典的模式。”
- “Agent 的工作流是 ReAct：Reason → Act → Observe → Final Answer。”
- “**Memory 解决上下文连贯性，RAG 解决知识覆盖问题**，两者可以结合使用。”
- “在实际开发中，我会用 **LangChain 的 AgentExecutor + Tools + Memory，来构建一个企业级智能体系统。**”





---





# 📌 LangChain Agent 开发详解（带代码）

## 1. Agent 开发的核心要素
1. **LLM** → 决策与推理  
2. **Tools** → 工具调用（搜索、数据库、计算器等）  
3. **Memory** → 上下文记忆  
4. **Agent Executor** → 智能体执行器  

👉 面试时要强调：**Agent = LLM + Tools + Memory + 执行循环**  

---

## 2. 定义工具（Tools）
在 LangChain 里，工具就是一个函数 + 描述，Agent 会选择合适的工具来调用。  

```python
from langchain.agents import Tool

# 定义一个简单工具：数学计算
def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = Tool(
    name="Multiply",
    func=lambda x: multiply(*map(int, x.split(","))),
    description="用来计算两个数字的乘积，输入格式为 'a,b'"
)
```

👉 **要点**：  
- 每个工具都需要 `name`、`func`、`description`。  
- LLM 根据描述来选择工具。  

---

## 3. 创建一个 ReAct Agent
ReAct 模式（Reasoning + Acting + Observing）是 **最常见的智能体模式**。

```python
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType

# 加载大模型
llm = OpenAI(temperature=0)

# 定义工具列表
tools = [multiply_tool]

# 初始化智能体
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct模式
    verbose=True
)

# 运行
result = agent.run("计算 23 和 7 的乘积")
print(result)
```

👉 **流程**：  
1. 用户提问  
2. LLM 生成 Reasoning → 选择工具 Multiply  
3. 执行 Multiply(23,7) → 161  
4. 输出 Final Answer  

---

## 4. Conversational Agent（带记忆的对话智能体）
```python
from langchain.memory import ConversationBufferMemory

# 定义记忆
memory = ConversationBufferMemory(memory_key="chat_history")

agent_with_memory = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # 对话Agent
    memory=memory,
    verbose=True
)

# 多轮对话
print(agent_with_memory.run("我有 12 个苹果"))
print(agent_with_memory.run("我再买 8 个，现在一共有多少？"))
```

👉 **要点**：  
- ConversationAgent 结合 Memory，可以追踪上下文。  
- 面试时可以说：**Memory 解决了对话连贯性问题。**  

---

## 5. Plan and Execute Agent（规划 + 执行）
适合任务较长、需要多个步骤的情况。

```python
from langchain.agents import load_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate

# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个会规划任务的助手"),
    ("user", "{input}")
])

# 定义智能体
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
agent_executor.invoke({"input": "帮我查一下北京今天的天气，并计算比上海温度高多少度"})
```

👉 **要点**：  
- Plan 阶段：LLM 拆解任务 → 查天气 + 数学计算  
- Execute 阶段：依次调用工具完成任务  

---

## 6. 多 Agent 协作（LangGraph）
最新趋势是 **多智能体协作**，LangChain 推出了 **LangGraph**，可以让多个 Agent 协同解决复杂任务。  

```python
# 示例结构（伪代码）
from langgraph.graph import StateGraph

# 定义两个Agent：Planner + Executor
planner = ...
executor = ...

workflow = StateGraph()
workflow.add_node("plan", planner)
workflow.add_node("execute", executor)
workflow.add_edge("plan", "execute")

graph = workflow.compile()
result = graph.invoke("规划一次上海3日游并估算预算")
```

👉 **要点**：  
- Planner 负责拆解任务  
- Executor 负责执行  
- 这种模式适合 **企业级流程编排**  

---

## 7. 面试常见加分点
- **如何减少模型幻觉？**  
👉 用 RAG（检索增强生成），限制模型在知识库范围内回答。  
- **如何保证输出可靠？**  
👉 用 `OutputParser` 或 Function Calling，确保结构化输出。  
- **Agent 适合哪些场景？**  
👉 多工具调用、多步骤任务（金融风控、智能客服、办公自动化）。  

---

# 🎯 总结（背诵重点）
1. **Agent = LLM + Tools + Memory + Executor**  
2. 常见类型：ZeroShotAgent、ConversationalAgent、ReActAgent、Plan&ExecuteAgent  
3. 核心流程：Reason → Act → Observe → Final Answer  
4. Tools 是 Agent 的手臂，Memory 保证对话连贯，RAG 解决知识不足  
5. 企业级应用：多 Agent 协作（LangGraph）、安全性与权限控制  

