我把你给的项目说明**完善成一个可演示、可讲解的完整工程蓝图**，并把关键代码都写出来：**后端（Django）+ LangChain Agent + FAISS 向量存储 + 文档/网页抓取与向量化 + Docker**。你可以在面试中边讲边展示代码结构、运行流程和 demo（上传文档 → 构建向量库 → 用 Agent 问问题并调用工具）。

下面是完整内容：项目结构、关键文件、实现细节、运行/演示步骤，以及面试讲解要点。代码里我用占位的 API KEY / 路径，面试时说明你会替换为真实密钥即可。

------

# 项目概览（补充完善后的描述 — 面试用一句话）

**大模型 AI 工具箱平台**：基于 Python + LangChain + FAISS + Django + Docker，集成多家第三方 API 与爬虫（Crawl4AI 或自建爬虫），把企业文档与网页数据向量化存储到 FAISS，提供 RAG 问答与 Agent 自动化工具调用的 RESTful 服务，并通过 Django Admin 管理数据。

------

# 项目结构（建议）

```
ai-toolbox/
├─ backend/
│  ├─ Dockerfile
│  ├─ docker-compose.yml
│  ├─ requirements.txt
│  ├─ manage.py
│  ├─ ai_toolbox/          # Django project
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  └─ wsgi.py
│  ├─ ingestion/           # 文档/网页抓取与向量化
│  │  ├─ apps.py
│  │  ├─ tasks.py
│  │  └─ ingest.py
│  ├─ agent_service/       # LangChain Agent 实现
│  │  ├─ __init__.py
│  │  ├─ tools.py
│  │  ├─ agent_runner.py
│  │  └─ rag.py
│  ├─ api/                 # Django REST API
│  │  ├─ views.py
│  │  ├─ serializers.py
│  │  └─ urls.py
│  └─ models.py
└─ README.md
```

------

# 核心流程（演示时解释）

1. 上传/收集数据：通过 Django 上传 PDF/Docx/URL 或 Crawl4AI 抓取网页。
2. 文本预处理：TextSplitter 切分、去噪、元数据打包。
3. Embedding：使用 OpenAI Embeddings 或 SentenceTransformers 生成向量。
4. 向量库（FAISS）：保存向量与文档片段（metadata）。
5. RAG 问答：检索 top_k 文档片段 + 将上下文拼接到 Prompt 交给 LLM 生成回答。
6. Agent 模式：当问题需要外部工具（计算、外部 API、数据库查询）时，Agent 会选择 Tool 去执行并把结果反馈给 LLM，最终给用户答案。
7. 提供 REST API：前端调用 `/api/query` 或 `/api/agent_query` 获得结果。

------

# 关键技术点与你要背诵/讲的要点

- RAG：DocumentLoader → TextSplitter → Embedding → VectorStore (FAISS) → Retriever → LLM。优点：降低幻觉、易更新知识库。
- Agent：LLM + Tools + Memory + Executor；ReAct（Reason→Act→Observe）为经典执行模型。
- Tools：应该为每个外部能力封装 name/description/func，Agent 根据描述选择调用。
- 安全/稳定：工具调用权限控制、输出格式约束（OutputParser/Function Calling）、缓存与监控。
- 性能：分片、批量 embedding、异步 ingestion、向量检索限制 top_k、缓存 LLM 输出。

------

# 代码实现（最小可运行核心片段）

> **注意**：下面代码已尽量精简为关键可直接运行片段（替换 API keys）。面试时你可以拿这些代码逐步讲解。

### requirements.txt

```
Django>=4.2
djangorestframework
langchain>=0.0.200
openai
faiss-cpu
sentence-transformers
python-multipart
requests
beautifulsoup4
gunicorn
```

------

### backend/ai_toolbox/settings.py（核心片段）

```python
# 省略标准 Django 配置
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'ingestion',
    'agent_service',
    'api',
]

# Embedding / LLM 配置（示例）
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
FAISS_INDEX_PATH = "/data/faiss.index"
```

------

### backend/ingestion/ingest.py

负责：加载文档/网页 → 切分 → 生成 embedding → 存入 FAISS

```python
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def load_pdf(path):
    loader = UnstructuredPDFLoader(path)
    docs = loader.load()
    return docs

def load_url(url):
    loader = UnstructuredURLLoader([url])
    return loader.load()

def ingest_documents(docs, index_path="faiss_index"):
    # 文本切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # FAISS 存储（如果存在则追加）
    if os.path.exists(index_path):
        faiss_store = FAISS.load_local(index_path, embeddings)
        faiss_store.add_documents(texts)
    else:
        faiss_store = FAISS.from_documents(texts, embeddings)
        faiss_store.save_local(index_path)
    return True
```

> 面试点：说明 `chunk_size` vs `overlap` 为什么设置（保留上下文、避免切断句子）。并能说明用 OpenAIEmbeddings 或本地 sentence-transformers 的利弊（成本 vs 隐私/速度）。

------

### backend/agent_service/rag.py

基本的 RAG 查询 pipeline。

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")

def get_retriever(index_path=INDEX_PATH, k=4):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = FAISS.load_local(index_path, embeddings)
    return store.as_retriever(search_kwargs={"k": k})

def rag_answer(query):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    retriever = get_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="基于以下上下文，回答问题，若上下文没有答案，请说明无法回答。上下文：\n{context}\n\n问题：{question}"
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    res = qa_chain.run(query)
    return res
```

> 面试点：能解释不同的 chain_type（map_reduce, stuff, refine）适用场景和成本/延迟权衡。

------

### backend/agent_service/tools.py

定义几个工具（搜索、计算、调用外部 API）。

```python
from langchain.agents import Tool
import requests
import json

def calc_func(inp: str):
    # 简单安全计算器：仅数字和运算符
    import re
    if re.match(r'^[0-9+\-*/ ().]+$', inp.strip()):
        return str(eval(inp))
    return "Invalid expression"

calc_tool = Tool(
    name="Calculator",
    func=lambda q: calc_func(q),
    description="用于数学计算，只接受算术表达式"
)

def web_search(query: str):
    # 占位：若集成 SerpAPI 或 Crawl4AI 在此实现
    return f"搜索结果（模拟）: {query}"

web_tool = Tool(
    name="WebSearch",
    func=lambda q: web_search(q),
    description="在网络上搜索信息，返回简要摘要"
)
```

> 面试点：强调安全（比如 calc_tool 要限制输入，防止命令注入）。展示你会用 function calling 或严格输出解析来避免 LLM 随机输出导致调用错误。

------

### backend/agent_service/agent_runner.py

初始化 Agent 并示例调用（ReAct 模式）。

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from .tools import calc_tool, web_tool
from langchain.memory import ConversationBufferMemory

def build_agent():
    llm = ChatOpenAI(temperature=0)
    tools = [calc_tool, web_tool]
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)
    return agent

def run_agent_query(agent, query):
    return agent.run(query)
```

演示：`agent.run("帮我查下上海今天的天气，并算出温差")`（实际会调用 web_tool 再计算）。

------

### backend/api/views.py（REST endpoints）

提供上传 ingest、RAG 查询、Agent 查询接口。

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ingestion.ingest import load_pdf, load_url, ingest_documents
from agent_service.rag import rag_answer
from agent_service.agent_runner import build_agent, run_agent_query

agent = build_agent()

class IngestAPIView(APIView):
    def post(self, request):
        # 支持文件上传或 url
        url = request.data.get('url')
        file = request.FILES.get('file')
        docs = []
        if file:
            path = "/tmp/" + file.name
            with open(path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)
            docs = load_pdf(path)
        elif url:
            docs = load_url(url)
        else:
            return Response({"detail": "no input"}, status=status.HTTP_400_BAD_REQUEST)

        ingest_documents(docs)
        return Response({"detail": "ingested"}, status=status.HTTP_200_OK)

class RAGQueryAPIView(APIView):
    def post(self, request):
        q = request.data.get("query")
        if not q:
            return Response({"detail": "no query"}, status=status.HTTP_400_BAD_REQUEST)
        answer = rag_answer(q)
        return Response({"answer": answer})

class AgentQueryAPIView(APIView):
    def post(self, request):
        q = request.data.get("query")
        if not q:
            return Response({"detail": "no query"}, status=status.HTTP_400_BAD_REQUEST)
        res = run_agent_query(agent, q)
        return Response({"result": res})
```

------

### backend/Dockerfile（示例）

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "ai_toolbox.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2"]
```

------

# 展示 / Demo 步骤（面试演示脚本）

1. 启动（本地或 Docker）
   - `docker build -t ai-toolbox .`
   - `docker run -e OPENAI_API_KEY=xxx -p 8000:8000 ai-toolbox`
      或用 `python manage.py runserver`（开发模式）。
2. Ingest（构建向量库）
   - POST `/api/ingest` 上传 PDF 或 `{"url": "https://example.com/doc"}`。
   - 说明：会切分、生成 embedding、保存到 FAISS。
3. RAG 问答（基础）
   - POST `/api/query` body `{"query": "关于 X 的问题"}`。
   - 展示：系统如何检索相关片段并回答。
4. Agent 问题（工具调用）
   - POST `/api/agent_query` body `{"query": "帮我计算：23*17 并查找最近新闻"}`
   - 展示：Agent 会选择 Calculator + WebSearch（如果实现）；并展示 ReAct 的交互日志（`verbose=True` 会打印 Reason/Action/Observation）。
5. 讲解：如何把该系统部署到 prod（多副本、向量服务单独部署、权限、日志）。

------

# 面试中你可以重点陈述的细节（问题+参考回答）

- **为什么用 FAISS？**
  - 回答要点：快速、内存/磁盘部署灵活、与 langchain 集成好；适合低延迟的向量检索。
- **检索拼接上下文有哪些策略？**
  - `stuff`：把 top_k 直接塞到 prompt（简单但可能超限）
  - `map_reduce`：分段生成 summary 再合并（节省 token）
  - `refine`：初步答案后逐步修正（精确但成本更高）
- **Agent 怎么保证安全？**
  - 工具白名单 + 输入校验、输出格式校验（OutputParser 或 Function Calling）、日志审计、权限控制。
- **如何评估与优化？**
  - 指标：响应时间、准确率（人工抽样）、成本（API 调用 token）。优化手段：缓存、降采样检索、较低成本模型作复核等。

------

# 扩展建议（面试加分）

- **用 Redis + Faiss**：把向量索引分片与持久化结合（大数据集）。
- **用 Milvus/Pinecone/Weaviate**：生产级别向量数据库（多副本、布隆过滤、向量过滤）。
- **引入队列（Celery/RQ）**：把 ingestion / embedding 异步化，避免阻塞 API。
- **监控与审计**：记录每次 Agent 的工具调用与 LLM 输出，用于回溯与调优。
- **自动化测试**：用示例问答和断言定期跑回归测试（确保 RAG 更新不破坏回答质量）。

------

# 面试演练台词（示例）

> “这个项目的核心是把企业内部非结构化数据（PDF、文档、网页）转为向量并存入 FAISS，查询时用 Retriever 找到相关片段，再把片段连同问题发给 LLM。对于需要外部能力的场景，比如计算、调用外部 API、数据库查询，我们通过 LangChain Agent 的 Tools 把这些能力封装成可调用函数，LLM 将决定何时调用哪个工具，执行结果再反馈回模型以生成最终答案。我们还考虑了安全（工具白名单、输出格式校验）和性能（缓存、异步 ingestion）。在 demo 我可以展示从上传文档到 RAG 问答，再到 Agent 自动调用工具的全流程。”

------

# 你面试时可能会被问到的深挖问题（及简答）

1. **如何防 Prompt Injection？**
   - 策略：把外部文本当作“参考资料”不直接执行其中的命令，限制工具接口仅接受结构化参数，校验输入、输出，并对重要操作加二次确认或人工审核。
2. **为什么选择 ReAct 而非固定 chain？**
   - ReAct 在面对需要动态工具选择和多步决策的问题时更灵活，能让模型在执行中做“思考-行动-观察”的闭环。
3. **如何衡量 RAG 的效果？**
   - 用自动/人工评测 (precision@k, nDCG), 抽样检查 hallucination rate, 用 A/B 测试不同 retriever / prompt 设置。

------





#  技术实现

## langchain 如何使用 agent 去实现文档  法规相关内容的判定，如果有问题，是不是得需要生成相关的报告？



使用 LangChain Agent 来实现文档法规相关内容的判定是一个很实用的应用场景。我来为你详细介绍实现方案：

## 整体架构设计

**核心组件：**

- **Document Loader**: 加载和预处理法规文档
- **Vector Store**: 存储法规条文的向量化表示
- **Agent**: 协调各个工具完成判定任务
- **Tools**: 包括文档检索、规则匹配、报告生成等工具
- **Memory**: 保存判定过程和历史记录

## 具体实现步骤

### 1. 法规文档预处理

```python
from langchain.document_loaders import PDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 加载法规文档
def load_regulation_docs():
    loaders = [
        PDFLoader("regulations/law1.pdf"),
        PDFLoader("regulations/law2.pdf"),
        # 添加更多法规文档
    ]
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # 向量化存储
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings()
    )
    
    return vectorstore
```

### 2. 创建专用工具

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

def create_regulation_tools(vectorstore):
    # 法规检索工具
    def search_regulations(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=5)
        return "\n".join([doc.page_content for doc in docs])
    
    # 合规性判定工具
    def compliance_check(content: str, regulation_type: str) -> str:
        # 这里可以集成更复杂的规则引擎
        relevant_regs = search_regulations(f"{regulation_type} {content}")
        # 使用 LLM 进行判定
        prompt = f"""
        基于以下法规条文：
        {relevant_regs}
        
        请判定以下内容是否合规：
        {content}
        
        请给出明确的合规/违规判定，并说明理由。
        """
        return llm(prompt)
    
    # 报告生成工具
    def generate_compliance_report(findings: str) -> str:
        prompt = f"""
        基于以下合规性检查结果：
        {findings}
        
        请生成一份正式的合规性报告，包括：
        1. 执行摘要
        2. 详细发现
        3. 风险评估
        4. 改进建议
        """
        return llm(prompt)
    
    tools = [
        Tool(
            name="search_regulations",
            description="搜索相关法规条文",
            func=search_regulations
        ),
        Tool(
            name="compliance_check",
            description="进行合规性判定",
            func=compliance_check
        ),
        Tool(
            name="generate_report",
            description="生成合规性报告",
            func=generate_compliance_report
        )
    ]
    
    return tools
```

### 3. 构建 Agent

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

def create_compliance_agent():
    # 初始化向量存储
    vectorstore = load_regulation_docs()
    
    # 创建工具
    tools = create_regulation_tools(vectorstore)
    
    # 初始化 LLM
    llm = OpenAI(temperature=0)
    
    # 创建记忆组件
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 初始化 Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    
    return agent
```

### 4. 使用示例

```python
# 创建合规性 Agent
compliance_agent = create_compliance_agent()

# 进行文档合规性检查
def check_document_compliance(document_content: str, regulation_areas: list):
    query = f"""
    请对以下文档内容进行合规性检查：
    
    文档内容：
    {document_content}
    
    需要检查的法规领域：{', '.join(regulation_areas)}
    
    请：
    1. 搜索相关法规条文
    2. 进行逐项合规性判定
    3. 如发现问题，生成详细的合规性报告
    """
    
    result = compliance_agent.run(query)
    return result

# 实际使用
document = "某公司的数据处理政策..."
areas = ["数据保护法", "网络安全法", "个人信息保护法"]

compliance_result = check_document_compliance(document, areas)
print(compliance_result)
```

## 高级功能扩展

### 1. 多层级判定

```python
def create_tiered_compliance_check():
    # 第一层：关键词匹配
    # 第二层：语义相似度
    # 第三层：专家规则引擎
    # 第四层：LLM 深度分析
    pass
```

### 2. 自动报告生成

```python
class ComplianceReportGenerator:
    def __init__(self):
        self.template = """
        # 合规性检查报告
        
        ## 执行摘要
        {summary}
        
        ## 检查详情
        {details}
        
        ## 风险评估
        {risks}
        
        ## 改进建议
        {recommendations}
        """
    
    def generate_report(self, findings):
        # 结构化报告生成逻辑
        pass
```

### 3. 实时监控

```python
from langchain.callbacks import BaseCallbackHandler

class ComplianceMonitor(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        # 记录工具使用
        pass
    
    def on_tool_end(self, output, **kwargs):
        # 记录判定结果
        pass
```

## 关键考虑点

**准确性保证：**

- 使用多重验证机制
- 定期更新法规库
- 人工审核关键判定

**性能优化：**

- 合理的文档切分策略
- 高效的向量检索
- 缓存常用判定结果

**可扩展性：**

- 模块化工具设计
- 支持新增法规领域
- 灵活的报告模板

这种基于 Agent 的方案可以有效地处理复杂的法规合规性判定任务，并在发现问题时自动生成详细报告。你可以根据具体的业务需求调整工具配置和判定逻辑。



## rag 的数据如何进行管理和维护

RAG 系统的数据管理和维护是确保系统长期稳定运行的关键。我来为你详细介绍完整的数据管理体系：

## 1. 数据生命周期管理

### 数据摄入阶段

```python
from datetime import datetime
import hashlib
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DocumentMetadata:
    doc_id: str
    source: str
    created_at: datetime
    updated_at: datetime
    version: str
    hash: str
    tags: List[str]
    status: str  # active, archived, deleted

class RAGDataManager:
    def __init__(self):
        self.metadata_store = {}  # 实际应该用数据库
        self.version_control = {}
    
    def ingest_document(self, content: str, source: str, tags: List[str] = None):
        # 生成文档指纹
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        doc_id = f"{source}_{doc_hash[:8]}"
        
        # 检查是否已存在
        if self._document_exists(doc_hash):
            return self._handle_duplicate(doc_id, content)
        
        # 创建元数据
        metadata = DocumentMetadata(
            doc_id=doc_id,
            source=source,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0",
            hash=doc_hash,
            tags=tags or [],
            status="active"
        )
        
        # 处理和存储
        processed_chunks = self._process_document(content, metadata)
        self._store_document(processed_chunks, metadata)
        
        return doc_id
    
    def _document_exists(self, doc_hash: str) -> bool:
        return any(meta.hash == doc_hash for meta in self.metadata_store.values())
```

### 版本控制系统

```python
class DocumentVersionControl:
    def __init__(self):
        self.versions = {}
    
    def create_version(self, doc_id: str, content: str, change_type: str):
        if doc_id not in self.versions:
            self.versions[doc_id] = []
        
        version_info = {
            'version': len(self.versions[doc_id]) + 1,
            'timestamp': datetime.now(),
            'change_type': change_type,  # create, update, delete
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'content': content
        }
        
        self.versions[doc_id].append(version_info)
        return version_info['version']
    
    def get_version_history(self, doc_id: str):
        return self.versions.get(doc_id, [])
    
    def rollback_to_version(self, doc_id: str, version: int):
        if doc_id in self.versions and version <= len(self.versions[doc_id]):
            target_version = self.versions[doc_id][version - 1]
            # 重新处理并更新向量存储
            return self._restore_version(doc_id, target_version)
        return False
```

## 2. 增量更新机制

### 智能更新检测

```python
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RAGUpdateManager:
    def __init__(self, data_manager, vectorstore):
        self.data_manager = data_manager
        self.vectorstore = vectorstore
        self.update_queue = []
    
    def setup_automated_updates(self):
        # 定时检查
        schedule.every(1).hours.do(self._check_for_updates)
        
        # 文件系统监控
        event_handler = DocumentChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path='/data/documents', recursive=True)
        observer.start()
    
    def _check_for_updates(self):
        """检查数据源变化"""
        # 检查数据库变化
        db_changes = self._check_database_updates()
        
        # 检查API数据变化
        api_changes = self._check_api_updates()
        
        # 检查文件系统变化
        file_changes = self._check_file_updates()
        
        all_changes = db_changes + api_changes + file_changes
        
        for change in all_changes:
            self._process_update(change)
    
    def _process_update(self, change):
        """处理单个更新"""
        if change['type'] == 'create':
            self._handle_new_document(change)
        elif change['type'] == 'update':
            self._handle_document_update(change)
        elif change['type'] == 'delete':
            self._handle_document_deletion(change)
    
    def _handle_document_update(self, change):
        """处理文档更新"""
        doc_id = change['doc_id']
        new_content = change['content']
        
        # 获取旧版本
        old_chunks = self.vectorstore.get_by_doc_id(doc_id)
        
        # 生成新的chunks
        new_chunks = self._process_content(new_content)
        
        # 计算差异
        chunks_to_add, chunks_to_remove, chunks_to_update = self._calculate_diff(
            old_chunks, new_chunks
        )
        
        # 批量更新
        with self.vectorstore.batch_update():
            if chunks_to_remove:
                self.vectorstore.delete(chunks_to_remove)
            if chunks_to_add:
                self.vectorstore.add(chunks_to_add)
            if chunks_to_update:
                self.vectorstore.update(chunks_to_update)
```

### 智能分块更新

```python
class IntelligentChunkManager:
    def __init__(self):
        self.chunk_embeddings = {}
        self.similarity_threshold = 0.95
    
    def calculate_chunk_diff(self, old_chunks, new_chunks):
        """计算chunk级别的差异"""
        changes = {
            'added': [],
            'removed': [],
            'modified': []
        }
        
        old_embeddings = {chunk.id: chunk.embedding for chunk in old_chunks}
        new_embeddings = {chunk.id: chunk.embedding for chunk in new_chunks}
        
        # 找出新增的chunks
        for chunk in new_chunks:
            if not self._find_similar_chunk(chunk, old_chunks):
                changes['added'].append(chunk)
        
        # 找出删除的chunks
        for chunk in old_chunks:
            if not self._find_similar_chunk(chunk, new_chunks):
                changes['removed'].append(chunk)
        
        # 找出修改的chunks
        for new_chunk in new_chunks:
            similar_old = self._find_similar_chunk(new_chunk, old_chunks)
            if similar_old and self._chunks_differ(similar_old, new_chunk):
                changes['modified'].append((similar_old, new_chunk))
        
        return changes
    
    def _find_similar_chunk(self, target_chunk, chunk_list):
        """找到相似的chunk"""
        for chunk in chunk_list:
            similarity = self._calculate_similarity(target_chunk, chunk)
            if similarity > self.similarity_threshold:
                return chunk
        return None
```

## 3. 数据质量管理

### 质量检测系统

```python
class DataQualityManager:
    def __init__(self):
        self.quality_rules = self._load_quality_rules()
        self.quality_metrics = {}
    
    def validate_document(self, content: str, metadata: Dict) -> Dict:
        """全面的文档质量检测"""
        quality_report = {
            'overall_score': 0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 内容质量检查
        content_score = self._check_content_quality(content)
        
        # 元数据完整性检查
        metadata_score = self._check_metadata_completeness(metadata)
        
        # 重复性检查
        duplication_score = self._check_duplication(content)
        
        # 语言质量检查
        language_score = self._check_language_quality(content)
        
        # 计算综合评分
        quality_report['overall_score'] = (
            content_score * 0.4 +
            metadata_score * 0.2 +
            duplication_score * 0.2 +
            language_score * 0.2
        )
        
        return quality_report
    
    def _check_content_quality(self, content: str) -> float:
        """内容质量检查"""
        issues = []
        score = 100
        
        # 长度检查
        if len(content.strip()) < 50:
            issues.append("内容过短")
            score -= 20
        
        # 编码检查
        try:
            content.encode('utf-8')
        except UnicodeError:
            issues.append("编码问题")
            score -= 30
        
        # 格式检查
        if self._has_formatting_issues(content):
            issues.append("格式问题")
            score -= 10
        
        return max(0, score)
    
    def cleanup_low_quality_data(self, threshold: float = 60.0):
        """清理低质量数据"""
        low_quality_docs = []
        
        for doc_id, metadata in self.data_manager.metadata_store.items():
            if metadata.quality_score < threshold:
                low_quality_docs.append(doc_id)
        
        # 标记或删除低质量文档
        for doc_id in low_quality_docs:
            self._handle_low_quality_document(doc_id)
```

### 数据一致性检查

```python
class DataConsistencyChecker:
    def __init__(self, vectorstore, metadata_store):
        self.vectorstore = vectorstore
        self.metadata_store = metadata_store
    
    def run_consistency_check(self) -> Dict:
        """运行完整的一致性检查"""
        report = {
            'orphaned_vectors': [],
            'missing_vectors': [],
            'metadata_mismatches': [],
            'duplicate_documents': []
        }
        
        # 检查孤立的向量
        report['orphaned_vectors'] = self._find_orphaned_vectors()
        
        # 检查缺失的向量
        report['missing_vectors'] = self._find_missing_vectors()
        
        # 检查元数据不匹配
        report['metadata_mismatches'] = self._check_metadata_consistency()
        
        # 检查重复文档
        report['duplicate_documents'] = self._find_duplicate_documents()
        
        return report
    
    def fix_consistency_issues(self, report: Dict):
        """修复一致性问题"""
        # 清理孤立向量
        if report['orphaned_vectors']:
            self.vectorstore.delete_vectors(report['orphaned_vectors'])
        
        # 重建缺失向量
        for doc_id in report['missing_vectors']:
            self._rebuild_document_vectors(doc_id)
        
        # 修复元数据不匹配
        for mismatch in report['metadata_mismatches']:
            self._fix_metadata_mismatch(mismatch)
```

## 4. 性能优化

### 索引管理

```python
class VectorIndexManager:
    def __init__(self):
        self.index_stats = {}
        self.optimization_schedule = {}
    
    def optimize_index(self, force_rebuild: bool = False):
        """索引优化"""
        current_stats = self._get_index_statistics()
        
        if force_rebuild or self._should_rebuild_index(current_stats):
            self._rebuild_index()
        else:
            self._incremental_optimization()
    
    def _should_rebuild_index(self, stats: Dict) -> bool:
        """判断是否需要重建索引"""
        # 基于删除比例、碎片化程度等决定
        deletion_ratio = stats['deleted_count'] / stats['total_count']
        fragmentation_score = stats['fragmentation_score']
        
        return deletion_ratio > 0.3 or fragmentation_score > 0.5
    
    def create_custom_index(self, field: str, index_type: str):
        """创建自定义索引"""
        if index_type == 'hash':
            return self._create_hash_index(field)
        elif index_type == 'btree':
            return self._create_btree_index(field)
        elif index_type == 'semantic':
            return self._create_semantic_index(field)
```

### 缓存策略

```python
class RAGCacheManager:
    def __init__(self):
        self.query_cache = {}
        self.document_cache = {}
        self.embedding_cache = {}
    
    def get_cached_result(self, query: str, cache_type: str = 'query'):
        """获取缓存结果"""
        cache_key = self._generate_cache_key(query)
        
        if cache_type == 'query' and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result['data']
        
        return None
    
    def cache_result(self, query: str, result: any, cache_type: str = 'query', ttl: int = 3600):
        """缓存结果"""
        cache_key = self._generate_cache_key(query)
        cache_entry = {
            'data': result,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
        if cache_type == 'query':
            self.query_cache[cache_key] = cache_entry
        elif cache_type == 'document':
            self.document_cache[cache_key] = cache_entry
        elif cache_type == 'embedding':
            self.embedding_cache[cache_key] = cache_entry
    
    def invalidate_cache(self, pattern: str = None):
        """失效缓存"""
        if pattern:
            # 根据模式失效特定缓存
            self._invalidate_by_pattern(pattern)
        else:
            # 清空所有缓存
            self.query_cache.clear()
            self.document_cache.clear()
            self.embedding_cache.clear()
```

## 5. 监控和告警

### 系统监控

```python
class RAGMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.thresholds = self._load_thresholds()
    
    def collect_metrics(self):
        """收集系统指标"""
        metrics = {
            'document_count': self._get_document_count(),
            'vector_count': self._get_vector_count(),
            'query_latency': self._get_average_query_latency(),
            'memory_usage': self._get_memory_usage(),
            'disk_usage': self._get_disk_usage(),
            'index_size': self._get_index_size(),
            'update_frequency': self._get_update_frequency()
        }
        
        # 检查阈值
        self._check_thresholds(metrics)
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict):
        """检查告警阈值"""
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                if value > threshold['critical']:
                    self._trigger_alert(metric_name, value, 'critical')
                elif value > threshold['warning']:
                    self._trigger_alert(metric_name, value, 'warning')
    
    def generate_health_report(self) -> str:
        """生成健康状况报告"""
        metrics = self.collect_metrics()
        
        report = f"""
        # RAG系统健康报告
        
        ## 数据概况
        - 文档数量: {metrics['document_count']}
        - 向量数量: {metrics['vector_count']}
        - 索引大小: {metrics['index_size']} MB
        
        ## 性能指标
        - 查询延迟: {metrics['query_latency']} ms
        - 内存使用: {metrics['memory_usage']}%
        - 磁盘使用: {metrics['disk_usage']}%
        
        ## 更新统计
        - 更新频率: {metrics['update_frequency']} 次/小时
        
        ## 告警信息
        {self._format_alerts()}
        """
        
        return report
```

这套完整的 RAG 数据管理体系可以确保数据的质量、一致性和系统的长期稳定运行。你可以根据具体的业务场景选择合适的组件进行实施。



## 如何进行权限管理，项目中的，请详细进行介绍。

在项目中 **权限管理**（Access Control / Authorization）是一个非常核心的部分，它决定了谁能访问哪些资源、能执行什么操作。做好权限管理，既能保护系统安全，又能让团队协作高效。下面我给你详细拆解一下权限管理的思路和实践：

------

## 一、权限管理的基本目标

1. **安全性**：防止未授权的用户访问敏感资源。
2. **合规性**：符合企业内部和外部法规（如隐私保护、合规审计）。
3. **灵活性**：可以方便地新增/修改权限，不影响现有逻辑。
4. **可维护性**：结构清晰，避免权限逻辑散落在代码里。

------

## 二、常见的权限模型

### 1. **ACL（Access Control List，访问控制列表）**

- 每个资源有一个列表，记录哪些用户/角色可以访问。
- 适用于资源粒度较小的场景。
- 缺点：管理复杂度随资源数增加而指数增长。

### 2. **RBAC（Role-Based Access Control，基于角色的访问控制）** ⭐常用

- 用户 → 角色 → 权限。
- 示例：
  - 用户张三 → 角色「管理员」 → 拥有「新增用户、删除用户」权限。
  - 用户李四 → 角色「普通用户」 → 拥有「查看文档」权限。
- 优点：清晰、易维护。
- 缺点：当权限场景复杂（需要动态条件）时不够灵活。

### 3. **ABAC（Attribute-Based Access Control，基于属性的访问控制）**

- 根据用户属性、资源属性、环境属性来动态决定权限。
- 示例：
  - 用户部门 = 财务部 且 访问时间 = 工作日 9-18点 → 允许访问报销系统。
- 优点：灵活，可实现复杂策略。
- 缺点：设计与运维成本高。

### 4. **PBAC（Policy-Based Access Control，基于策略的访问控制）**

- 使用统一的策略引擎（如 **Casbin、OPA**），通过配置规则来决定是否允许访问。
- 更适合大型项目，规则可集中化管理。

------

## 三、权限管理的设计步骤

### 1. **需求分析**

- 确定有哪些资源（页面、接口、按钮、数据）。
- 确定哪些角色/用户群体。
- 确定不同用户在不同场景下的操作边界。

### 2. **权限模型选择**

- 简单系统：RBAC 就够用。
- 复杂系统（跨部门、跨环境）：ABAC 或 PBAC。

### 3. **数据建模**

以 RBAC 为例，一般有 4 张核心表：

1. **用户表**（User）
   - id, username, password, dept_id, …
2. **角色表**（Role）
   - id, role_name, role_desc, …
3. **权限表**（Permission）
   - id, perm_name, resource, action, …
4. **关联表**
   - user_role（用户与角色关系）
   - role_permission（角色与权限关系）

### 4. **权限校验方式**

- **前端**：隐藏无权限的按钮/菜单。
- **后端**：拦截接口请求，校验用户权限。
- **中间件**：统一的权限认证模块，减少重复逻辑。

### 5. **权限管理后台**

- 提供界面给管理员配置：
  - 分配角色给用户
  - 配置角色的权限
  - 动态调整权限（例如：某人临时加入项目组）

------

## 四、项目中的落地实践

### 1. **接口层权限校验**

- 通过 JWT / OAuth2.0 携带用户信息。
- 在 API 网关或后端服务中解析 Token，获取用户角色 → 校验权限。

### 2. **数据级别权限**

- 不只是能不能访问，还要限制 **能访问哪些数据**。
- 示例：
  - 财务部用户只能看到本部门的报销单。
  - 管理员能看到全公司数据。

### 3. **日志与审计**

- 记录用户的访问行为（谁在什么时间访问了什么资源，是否成功）。
- 满足安全审计要求。

### 4. **工具/框架选型**

- Java/Spring 项目：Spring Security + JWT + RBAC。
- Go 项目：Casbin（强大、灵活的权限库）。
- 前端：在 Vue/React 里用路由守卫和组件级权限控制。

------

## 五、实际案例示意

例如一个项目管理系统：

- 角色：超级管理员、项目经理、普通成员。
- 权限：
  - 超级管理员：所有功能。
  - 项目经理：创建项目、分配任务、查看报表。
  - 普通成员：查看和执行被分配的任务。
- 数据权限：
  - 普通成员只能访问自己参与的项目数据。
  - 项目经理可以访问自己管理的项目数据。



## 详细介绍一种最常见的权限管理方式，使用django进行实现，脉络清晰，内容完整。



下面用 **RBAC（基于角色的访问控制）** 作为“最常见的一种权限管理方式”，并把它在 **Django** 中的落地实现讲清楚 —— 从建模、赋权、在视图/模板/API 的校验、对象级权限、缓存失效、审计与测试，到常见陷阱和最佳实践。脉络清晰、可直接复制粘贴到项目中使用。

------

# 1 总体思路（简要）

- **模型级权限**（Django 内置）：操作某类对象（add/change/delete/view）。
- **角色 = Group**：把 Permission 赋给 Group（角色），把用户加入 Group。
- **权限检查点**：
  - 前端（隐藏按钮/菜单，UX）—— **辅助**，不能依赖
  - 后端（视图/中间件/依赖）—— **强制**（必做）
  - 数据层（filter/owner 检查 或 DB 行级安全）—— 防止遗漏
- **对象级权限**（可选）：若每条数据需要不同授权（owner、协作者），用 `owner` 检查或 `django-guardian`。

------

# 2 准备：requirements / settings（简短）

```text
# requirements.txt 示例
Django>=4.0
djangorestframework           # 如果做 API
django-guardian               # 若需对象级权限
djangorestframework-simplejwt # 若使用 JWT（可选）
```

settings.py 关键项（若使用 django-guardian）：

```py
INSTALLED_APPS += [
    'rest_framework',    # 如果用 DRF
    'guardian',
    'yourapp',
]

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'guardian.backends.ObjectPermissionBackend',  # 用于对象级权限
)

ANONYMOUS_USER_ID = -1  # guardian 要求
```

------

# 3 模型设计（以文章 Article 为例）

`yourapp/models.py`：

```py
from django.db import models
from django.conf import settings

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='articles', on_delete=models.CASCADE)
    is_published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 自定义权限（除了 Django 自动的 add/change/delete/view）
        permissions = [
            ("can_publish", "Can publish article"),
        ]

    def __str__(self):
        return self.title
```

- `Meta.permissions` 会在 migrate 时写入 `auth_permission` 表。
- 推荐把 `resource:action` 的语义记清楚，例如 `blog.can_publish`。

------

# 4 创建角色（Group）并分配权限（可在 admin 做，也可程序化）

**管理命令：`manage.py create_roles`（示例）**

```py
# yourapp/management/commands/create_roles.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.apps import apps

class Command(BaseCommand):
    help = "Create default groups and assign permissions"

    def handle(self, *args, **options):
        Article = apps.get_model('yourapp', 'Article')
        perms = {
            'Admin': ['add_article','change_article','delete_article','view_article','can_publish'],
            'Editor': ['add_article','change_article','view_article','can_publish'],
            'Reader': ['view_article'],
        }
        for role_name, codenames in perms.items():
            group, _ = Group.objects.get_or_create(name=role_name)
            group.permissions.clear()
            for codename in codenames:
                try:
                    if codename == 'can_publish':
                        perm = Permission.objects.get(codename='can_publish', content_type__app_label='yourapp')
                    else:
                        perm = Permission.objects.get(codename=codename, content_type__model='article')
                    group.permissions.add(perm)
                except Permission.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f'Permission {codename} not found'))
        self.stdout.write(self.style.SUCCESS('Roles created/updated'))
```

运行：

```bash
python manage.py migrate
python manage.py create_roles
```

或直接在 Django Admin 的 **Groups** 页面分配权限。

------

# 5 在视图层做校验（Django 视图 / CBV / 模板）

## 5.1 基于装饰器（函数视图）

```py
from django.contrib.auth.decorators import permission_required

@permission_required('yourapp.can_publish', raise_exception=True)
def publish_article(request, pk):
    # 权限通过后执行
    ...
```

## 5.2 类视图（CBV）使用 `PermissionRequiredMixin`

```py
from django.contrib.auth.mixins import PermissionRequiredMixin
from django.views.generic import View

class PublishArticleView(PermissionRequiredMixin, View):
    permission_required = 'yourapp.can_publish'
    raise_exception = True

    def post(self, request, *args, **kwargs):
        ...
```

## 5.3 模板层渲染控制

```django
{% if perms.yourapp.can_publish %}
  <a href="{% url 'publish' article.id %}">Publish</a>
{% endif %}
```

> 注意：模板控制只是 UX 层面，必须配合后端校验。

------

# 6 对象级权限（两种常见方案）

## 6.1 简单的 owner（归属者）检查 —— 最直接、无需外部库

- 在查询和 get_object 前进行过滤或比较 `obj.owner == request.user`。

示例（ViewSet / CBV）：

```py
def get_queryset(self):
    qs = super().get_queryset()
    if self.request.user.is_superuser:
        return qs
    return qs.filter(owner=self.request.user)  # 只看自己的
```

或检查单个对象：

```py
obj = get_object_or_404(Article, pk=pk)
if obj.owner != request.user and not request.user.has_perm('yourapp.change_article'):
    raise PermissionDenied
```

## 6.2 使用 `django-guardian` 实现细粒度对象级权限

安装并配置（见上面 settings）后：

赋权：

```py
from guardian.shortcuts import assign_perm
assign_perm('change_article', some_user, article)
```

检查（任意地方）：

```py
request.user.has_perm('yourapp.change_article', article)  # True/False
```

DRF 中直接在 `has_object_permission` 使用 `user.has_perm(..., obj)` 即可。

------

# 7 Django REST Framework (DRF) 示例：自定义权限类

`permissions.py`：

```py
from rest_framework.permissions import BasePermission, SAFE_METHODS

class IsOwnerOrHasModelPermission(BasePermission):
    """
    Allow access if:
      - request.user is owner of obj OR
      - user has model permission (change/delete) OR
      - for safe methods, article.is_published or user has view permission
    """
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        # safe read
        if request.method in SAFE_METHODS:
            if getattr(obj, 'is_published', False):
                return True
            return request.user.has_perm('yourapp.view_article', obj) \
                   or request.user.has_perm('yourapp.view_article')
        # owner
        if hasattr(obj, 'owner') and obj.owner == request.user:
            return True
        # change/delete
        if request.method in ('PUT', 'PATCH'):
            return request.user.has_perm('yourapp.change_article', obj)
        if request.method == 'DELETE':
            return request.user.has_perm('yourapp.delete_article', obj)
        return False
```

在 ViewSet 中使用：

```py
from .permissions import IsOwnerOrHasModelPermission

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrHasModelPermission]
```

> 注意：`user.has_perm(..., obj)` 需要 `guardian` 或相应后端支持，或者你在代码里实现 owner 检查（如上）。

------

# 8 权限缓存与失效（性能优化）

- Django 的 `User` 对象会缓存权限（`_perm_cache`）以减少 DB 查询。**当权限变更（给用户/组分配权限）时需要清理该缓存**。
- 推荐使用 signals 清理缓存：

```py
from django.db.models.signals import m2m_changed
from django.dispatch import receiver
from django.contrib.auth.models import Group, User

@receiver(m2m_changed, sender=Group.permissions.through)
def clear_perm_cache_on_group_perm_change(sender, instance, action, **kwargs):
    if action in ('post_add','post_remove','post_clear'):
        for user in instance.user_set.all():
            if hasattr(user, '_perm_cache'):
                del user._perm_cache

@receiver(m2m_changed, sender=User.user_permissions.through)
def clear_perm_cache_on_user_perm_change(sender, instance, action, **kwargs):
    if action in ('post_add','post_remove','post_clear') and hasattr(instance, '_perm_cache'):
        del instance._perm_cache

@receiver(m2m_changed, sender=User.groups.through)
def clear_perm_cache_on_group_membership_change(sender, instance, action, **kwargs):
    if action in ('post_add','post_remove','post_clear') and hasattr(instance, '_perm_cache'):
        del instance._perm_cache
```

- 若在分布式环境中把权限集缓存到 Redis，还需要在权限变更时触发分布式失效（例如通过 Pub/Sub 或消息队列）。

------

# 9 审计 / 日志（必做）

- 记录关键操作（user, action, resource, object_id, succeeded/denied, timestamp, IP）。
- 可以实现中间件或在自定义权限类/装饰器内记录被拒绝的尝试，示例（简化）：

```py
import logging
logger = logging.getLogger('permissions')

def log_permission_attempt(user, action, obj, allowed):
    logger.info("perm attempt user=%s action=%s obj=%s allowed=%s", user.pk if user else None, action, getattr(obj,'pk', None), allowed)
```

将其置入权限判断逻辑中，便于事后审计与入侵检测。

------

# 10 单元/集成测试（示例）

```py
from django.test import TestCase
from django.contrib.auth.models import User, Group, Permission
from django.urls import reverse
from .models import Article

class PermissionTests(TestCase):
    def setUp(self):
        self.editor = User.objects.create_user('editor', 'e@example.com', 'pwd')
        editors = Group.objects.create(name='Editors')
        perm = Permission.objects.get(codename='change_article')
        editors.permissions.add(perm)
        self.editor.groups.add(editors)
        self.article = Article.objects.create(title='Hi', content='X', owner=self.editor)

    def test_editor_can_change_own_article(self):
        self.client.login(username='editor', password='pwd')
        resp = self.client.post(reverse('article_edit', args=[self.article.id]), {'title':'New'})
        self.assertNotEqual(resp.status_code, 403)
```

- CI 中必须包含权限测试用例，尤其是敏感路径（删除、导出、财务相关）。

------

# 11 管理后台与运维建议

- 使用 **Django Admin** 做初期权限/组管理（自动支持 Permission 与 Group）。
- 生产环境：建立一个「权限管理手册」列出所有 resource:action 映射，便于审计与合规。
- 将能赋予高权限的账号数量降到最少（最小特权原则）。
- 当用 JWT 传输权限（例如把 roles/perm 放入 token payload）时要注意：**权限变更无法立即生效** —— 使用短有效期的 access token + refresh token，或实现 token 黑名单。

------

# 12 常见陷阱与最佳实践（总结）

- **不要只靠前端做权限控制**（必须在后端验证）。
- **把权限按逻辑分组到 Group（角色）**，避免把权限直接分配给大量用户。
- 对象级权限复杂时优先考虑 `guardian` 或统一策略引擎（如 Casbin/OPA）。
- 修改权限后要清理缓存（`_perm_cache` 或自定义缓存）。
- 写好审计日志与自动化测试（CI 必跑）。
- 若数据非常敏感，考虑数据库行级安全（如 PostgreSQL RLS）配合应用层校验作为双保险。

------

# 13 推荐的文件结构（示例）

```
yourproject/
  yourapp/
    models.py          # Article, Meta.permissions
    admin.py           # ArticleAdmin 限制非 superuser 只见自己等
    views.py           # CBV/FV + permission checks
    permissions.py     # DRF 自定义权限类
    signals.py         # 清理用户权限缓存的 signal handlers
    management/commands/create_roles.py
    tests.py
  settings.py
  requirements.txt
```

------

如果你愿意，我可以：

- 把上面的示例 **生成成一个最小可运行的 Django 示例仓库**（含 `settings.py`、`models.py`、`views.py`、`urls.py`、管理命令与测试用例），或者
- 直接给出 **完整的 DRF + JWT + django-guardian** 的实现代码（可拷贝运行）。















