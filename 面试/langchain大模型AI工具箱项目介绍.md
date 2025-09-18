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

## langchain 如何使用 agent 去实现文档  法规相关内容的判定，如果有问题，是不是得需要生成相关的报告



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

### RBAC（Role-Based Access Control，基于角色的访问控制）** ⭐常用

- 用户 → 角色 → 权限。
- 示例：
  - 用户张三 → 角色「管理员」 → 拥有「新增用户、删除用户」权限。
  - 用户李四 → 角色「普通用户」 → 拥有「查看文档」权限。
- 优点：清晰、易维护。
- 缺点：当权限场景复杂（需要动态条件）时不够灵活。

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





# 工具调用



## 百度搜索

好的！下面给你一个**可直接落地**的「百度搜索」LangChain Tool 封装方案，默认走 **SerpAPI 的 Baidu 引擎**（稳定、省心、合法合规），并附带：

- LangChain `BaseTool` 实现（同步/异步）
- 结果统一结构化（title / link / snippet / position / source）
- 超时、重试、限速与最少依赖
- 在你的 Agent 中的接入示例
- Django REST API 封装示例（便于前端调用）

> 说明：官方百度搜索没有对外通用开放 API；生产上建议通过 **SerpAPI**（支持 `engine=baidu`）。如果你不想用第三方服务，可再做一个 HTML 解析 fallback，但要注意服务条款与 robots.txt。我先给出**SerpAPI 方案**（推荐）。

------

# 1) 安装与环境变量

```bash
pip install httpx pydantic langchain
# 若你还没有 LangChain 的工具基类，确保版本 >= 0.1.x
```

在运行环境中设置（Docker 推荐用 `ENV` 或 secrets）：

```bash
export SERPAPI_API_KEY=xxxxxx
```

------

# 2) LangChain Tool：BaiduSearchTool（SerpAPI）

```python
# tools/baidu_search_tool.py
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field, validator
from langchain.tools import BaseTool

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

class BaiduSearchInput(BaseModel):
    """Parameters for Baidu search."""
    query: str = Field(..., description="搜索关键词")
    num_results: int = Field(5, ge=1, le=20, description="返回的结果条数（1-20）")
    # 常用可选参数
    lr: Optional[str] = Field(None, description="语言/地区限定，如 'lang_zh-CN'")
    safe: Optional[bool] = Field(False, description="是否启用安全搜索（SerpAPI段支持）")
    page: Optional[int] = Field(1, ge=1, description="分页（从1开始）")
    # 时间范围在 Baidu/SerpAPI 里并非完全等价，这里提供一个语义化占位
    timeframe: Optional[str] = Field(None, description="时间范围: 'day'|'week'|'month'（可选）")

    @validator("query")
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query 不能为空")
        return v.strip()

class BaiduSearchTool(BaseTool):
    """Baidu search via SerpAPI."""
    name: str = "baidu_search"
    description: str = (
        "使用 Baidu（通过 SerpAPI）进行网页搜索，返回结构化结果。"
        "输入: {query, num_results, lr, safe, page, timeframe}；"
        "输出: 标题、链接、摘要、位置等。"
    )
    args_schema: type = BaiduSearchInput

    # 可选：简单限速（每次调用间隔最少 seconds_between_calls 秒）
    seconds_between_calls: float = 0.4
    _last_call_ts: float = 0.0

    # 超时与重试
    timeout_s: float = 10.0
    max_retries: int = 2

    # 你可以在实例化时传入 api_key；若不传则读环境变量
    serpapi_key: Optional[str] = None

    def _throttle(self) -> None:
        now = time.time()
        delta = now - self._last_call_ts
        if delta < self.seconds_between_calls:
            time.sleep(self.seconds_between_calls - delta)
        self._last_call_ts = time.time()

    def _build_params(self, inp: BaiduSearchInput) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "engine": "baidu",
            "q": inp.query,
            "api_key": self.serpapi_key or os.getenv("SERPAPI_API_KEY", ""),
            "no_cache": "true",          # 避免陈旧缓存
            "num": inp.num_results,
            "pn": (inp.page - 1) * inp.num_results if inp.page and inp.page > 1 else 0,
        }
        if inp.lr:
            params["lr"] = inp.lr
        if inp.safe:
            params["safe"] = "active"
        # timeframe：SerpAPI 对 baidu 不一定原生支持，这里作为将来扩展的占位
        if inp.timeframe in {"day", "week", "month"}:
            params["time_period"] = inp.timeframe  # 若未来支持则可直通；否则可忽略
        return params

    def _call_serpapi(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not params.get("api_key"):
            raise RuntimeError("缺少 SERPAPI_API_KEY，请配置环境变量或在工具实例化时传入 serpapi_key")
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.get(SERPAPI_ENDPOINT, params=params)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(0.5)  # 简单退避
        raise RuntimeError(f"SerpAPI 请求失败: {last_err}")

    @staticmethod
    def _normalize_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将 SerpAPI 的返回统一为:
        {title, link, snippet, position, source}
        """
        items: List[Dict[str, Any]] = []
        # SerpAPI baidu 返回字段可能是 'organic_results'
        for i, r in enumerate(payload.get("organic_results", []), start=1):
            items.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet") or r.get("displayed_link"),
                "position": r.get("position") or i,
                "source": "baidu",
            })
        return items

    def _run(self, query: str, num_results: int = 5, lr: Optional[str] = None,
             safe: Optional[bool] = False, page: Optional[int] = 1,
             timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """同步调用（供 Agent 默认使用）"""
        self._throttle()
        inp = BaiduSearchInput(
            query=query, num_results=num_results, lr=lr, safe=safe, page=page, timeframe=timeframe
        )
        params = self._build_params(inp)
        raw = self._call_serpapi(params)
        return self._normalize_results(raw)

    async def _arun(self, query: str, num_results: int = 5, lr: Optional[str] = None,
                    safe: Optional[bool] = False, page: Optional[int] = 1,
                    timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """异步调用（若你的 Agent 走异步链路可以用这个）"""
        # 简单用同步包装，也可以用 httpx.AsyncClient 改写为真正异步
        return self._run(query, num_results, lr, safe, page, timeframe)
```

------

# 3) 在 Agent/Chain 中接入

```python
# agents/init_agent.py
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools.baidu_search_tool import BaiduSearchTool

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    baidu_tool = BaiduSearchTool()  # 或 BaiduSearchTool(serpapi_key="...")

    tools = [baidu_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    print(agent.run("使用 baidu_search 搜索：清华大学 计算机系 最新 新闻"))
```

在指令式工具选择里，你可以写提示让 Agent 明确该用 `baidu_search` 工具（比如“若需要检索网页，请调用 baidu_search 工具”）。

------

# 4) Django REST API 封装（可选）

```python
# api/views.py
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from tools.baidu_search_tool import BaiduSearchTool

baidu_tool = BaiduSearchTool()

@api_view(["GET"])
def baidu_search_view(request):
    query = request.GET.get("q", "").strip()
    if not query:
        return Response({"error": "q 参数必填"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        num = int(request.GET.get("num", 5))
        page = int(request.GET.get("page", 1))
    except ValueError:
        return Response({"error": "num/page 必须为整数"}, status=status.HTTP_400_BAD_REQUEST)

    lr = request.GET.get("lr")
    safe = request.GET.get("safe", "false").lower() in {"1", "true", "yes"}
    timeframe = request.GET.get("timeframe")

    try:
        results = baidu_tool._run(
            query=query, num_results=num, lr=lr, safe=safe, page=page, timeframe=timeframe
        )
        return Response({"query": query, "results": results})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**路由：**

```python
# api/urls.py
from django.urls import path
from .views import baidu_search_view

urlpatterns = [
    path("tools/baidu_search", baidu_search_view),
]
```

**调用示例：**

```
GET /api/tools/baidu_search?q=生成式AI&num=5&page=1
```

------

# 5) 返回结果结构（前端友好）

统一返回：

```json
{
  "query": "生成式AI",
  "results": [
    {
      "title": "……",
      "link": "https://www.baidu.com/……",
      "snippet": "……",
      "position": 1,
      "source": "baidu"
    }
  ]
}
```

------

# 6) 生产建议 & 可选增强

- **缓存**：用 `(query, page, num)` 做键，短期缓存（如 5–15 分钟）减少费用与延迟。
- **审计**：记录调用日志（query、时间、IP/用户、返回条数、耗时、错误码）。
- **并发保护**：在高并发场景可用队列/令牌桶实现更严格的 QPS 控制。
- **安全**：避免将用户原始 query 直接落库（含个人信息时要脱敏）。
- **多引擎冗余**：可加 `engine="google"`/“bing” 平滑切换（依然通过 SerpAPI），或对比聚合。
- **结果富化**：解析站点 favicon、主域名，或调用你的 Crawl4AI 对命中的链接做**二跳抓取**+RAG 摘要。



## 帮我封装几个tool，一个是查取相应的企业文档和法规，根据上传的后续文件，匹配法规发现是否合格，如果不合格，生成不合格的报告，其中说明原因。

下面给你一套可直接接入 Agent 的“三件套”Tools，用于：

1. 检索企业文档/法规；2) 依据上传文件进行法规匹配与合规判定；3) 若不合格，自动生成**不符合项报告**（含原因与溯源）。
    代码用 **Python + LangChain + Faiss**，输出用 **Pydantic** 约束为固定 JSON，便于前端/后端消费。你可以把这三件套挂到同一个 Agent，也可按 REST API 调用。

------

# 设计概览

- **RegDocSearchTool**：在“法规与企业文档”向量库中检索相关片段（返回 chunk 证据）。
- **ComplianceCheckTool**：对“新上传文件”执行抽取→匹配→判定，输出逐条**要求-证据-结论**。
- **NCReportTool**（Non-Conformance Report）：基于判定结果，生成**不符合项报告**（结构化 JSON + Markdown 摘要），可直接落库或回传给前端下载。

> 实现要点：
>
> - 向量库：法规、企业标准、流程制度统一编码入 **Faiss**（建议 bge-m3 / e5-multilingual）。
> - 结构化抽取：用你前面“知识抽取工具”的抽取能力（日期、金额、责任人、指标等）。
> - 判定逻辑：**规则 + LLM 裁决**（Hybrid），可写 Rule 函数，也可让 LLM 做释义比对，且必须“只依据证据”。
> - 溯源：所有结论都挂 evidence（chunk_id、页码、offset、原文摘录）。

------

## 统一数据结构（Pydantic）

```python
# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Evidence(BaseModel):
    source_id: str               # 文档ID/法规ID
    chunk_id: str
    page: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    text_snippet: Optional[str] = None

class Requirement(BaseModel):
    req_id: str
    title: str
    statement: str               # 要求条文（人类可读）
    category: str                # 安全/质量/环保/信息安全...
    severity: Literal["low","medium","high","critical"] = "medium"

class MatchResult(BaseModel):
    requirement: Requirement
    matched_evidence: List[Evidence] = []
    decision: Literal["pass","fail","not_applicable","uncertain"]
    reason: str                  # 判定理由（必须基于证据）
    confidence: float = Field(ge=0, le=1)

class ComplianceSummary(BaseModel):
    doc_id: str
    checked_at: str
    overall: Literal["pass","fail","partial","uncertain"]
    stats: dict
    items: List[MatchResult]
```

------

## 1) 法规/文档检索工具：`RegDocSearchTool`

- 输入：查询语句（或文档上下文），可指定只搜“法规”或“企业文档”或“全部”。
- 输出：命中片段的**结构化证据**，供后续匹配使用。

```python
# tools/regdoc_search_tool.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from schemas import Evidence
from your_vector_store import faiss_search   # 你已有的Faiss封装

class RegDocSearchInput(BaseModel):
    query: str = Field(..., description="语义查询")
    top_k: int = Field(8, ge=1, le=20)
    scope: Literal["all","regulations","enterprise"] = "all"  # 搜索范围
    filters: Optional[Dict[str, Any]] = None                  # 例如 { "country":"CN","dept":"QA" }

class RegDocSearchTool(BaseTool):
    name = "regdoc_search"
    description = "在法规/企业文档库中进行向量检索，返回证据片段（带溯源信息）。"
    args_schema = RegDocSearchInput

    def _run(self, query: str, top_k: int = 8, scope: str = "all", filters=None) -> List[Evidence]:
        hits = faiss_search(query=query, top_k=top_k, scope=scope, filters=filters)
        results = []
        for h in hits:
            results.append(Evidence(
                source_id=h["doc_id"], chunk_id=h["chunk_id"], page=h.get("page"),
                start=h.get("start"), end=h.get("end"),
                text_snippet=h["text"][:600]
            ))
        return [e.dict() for e in results]

    async def _arun(self, **kwargs):
        return self._run(**kwargs)
```

------

## 2) 合规匹配与判定工具：`ComplianceCheckTool`

- 输入：上传文件 `doc_id`（或原文 text）；可选指定法规主题/类别。
- 步骤：
  1. 对上传文件做**结构化抽取**（沿用你已有的抽取管道）。
  2. 根据抽取出的关键信息 + 主题，调用 `regdoc_search` 搜法规条文（RAG）。
  3. 构造“**要求 Requirement**”候选集（可来源于：法规条文模板库 / LLM 释义归一）。
  4. 逐条判定：
     - **规则判定**（例：必须包含签署日期 / 某字段≥阈值 / 必须有责任人）
     - **LLM 对齐**：给“条文statement + 证据片段 + 上传文档相关段”做释义比对，强制**结构化输出**（pass/fail/NA/uncertain + reason + confidence）。
  5. 汇总整体结论与统计。

```python
# tools/compliance_check_tool.py
import datetime as dt
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional
from schemas import Requirement, Evidence, MatchResult, ComplianceSummary
from tools.regdoc_search_tool import RegDocSearchTool
from your_extract import extract_from_document        # 你的“知识抽取工具”
from your_llm import structured_judge                 # 结构化判定LLM封装
from your_rules import run_rule_checks_for_requirement # 规则函数集合

class ComplianceInput(BaseModel):
    doc_id: Optional[str] = Field(None, description="上传文件ID")
    text: Optional[str] = Field(None, description="原始文本（当没有doc_id时使用）")
    themes: List[str] = Field(default_factory=list, description="法规主题（如 ['信息安全','质量']）")
    top_k: int = 8

class ComplianceCheckTool(BaseTool):
    name = "compliance_check"
    description = ("对上传文件进行法规匹配与合规判定。"
                   "输出逐条要求的结论、理由与证据，并汇总整体结论。")
    args_schema = ComplianceInput

    def _build_requirements(self, evidences: List[Evidence], themes: List[str]) -> List[Requirement]:
        """
        生成待判定的 Requirement 列表。
        做法A：从“法规模板库”按主题拉取基础要求，再用LLM/检索补充释义；
        做法B：对命中的法规片段进行“规范化”抽取为Requirement。
        这里给个极简示例。
        """
        reqs = []
        for i, ev in enumerate(evidences[:10], start=1):
            reqs.append(Requirement(
                req_id=f"REQ_{i}",
                title="关键条文要求",
                statement=ev.text_snippet or "要求：文档需包含明确的责任人、日期与合规指标。",
                category=themes[0] if themes else "general",
                severity="medium"
            ))
        return reqs

    def _judge_one(self, requirement: Requirement, doc_extract: dict, ev_list: List[Evidence]) -> MatchResult:
        # 1) 规则快速判定（结构化抽取的字段直接验证）
        rule_res = run_rule_checks_for_requirement(requirement, doc_extract)
        if rule_res and rule_res["decision"] in {"pass","fail","not_applicable"}:
            return MatchResult(
                requirement=requirement,
                matched_evidence=ev_list[:3],
                decision=rule_res["decision"], reason=rule_res["reason"],
                confidence=rule_res.get("confidence", 0.8)
            )
        # 2) LLM 释义比对（仅使用提供的证据和上传文档摘要）
        llm_out = structured_judge(
            requirement=requirement,
            evidence_text="\n\n---\n".join([e.text_snippet or "" for e in ev_list]),
            doc_summary=self._compact_doc_summary(doc_extract)
        )
        return MatchResult(
            requirement=requirement,
            matched_evidence=ev_list[:3],
            decision=llm_out.decision,
            reason=llm_out.reason,
            confidence=llm_out.confidence
        )

    @staticmethod
    def _compact_doc_summary(doc_extract: dict) -> str:
        # 将抽取结果压缩成简明摘要，作为 LLM 判定输入
        rp = [p.get("name") for p in doc_extract.get("responsible_parties", [])]
        am = doc_extract.get("amounts", [])
        kpis = doc_extract.get("kpis", [])
        dts = doc_extract.get("dates", [])
        return (f"RESP:{rp[:3]} | AMOUNTS:{len(am)} | KPIS:{len(kpis)} | DATES:{len(dts)}")

    def _run(self, doc_id: Optional[str] = None, text: Optional[str] = None,
             themes: List[str] = [], top_k: int = 8) -> dict:
        assert (doc_id or text), "必须提供 doc_id 或 text"
        # 1) 结构化抽取（你已有的抽取服务）
        doc_extract = extract_from_document(text, doc_id) if text else extract_from_document(None, doc_id)
        # 2) 构造检索查询（可据抽取字段拼接query）
        query = " ".join(themes) or "合规 要求 责任人 日期 KPI"
        evidences = []
        # 用前面定义的工具检索法规/企业文档
        reg_search = RegDocSearchTool()
        ev_raw = reg_search._run(query=query, top_k=top_k, scope="all")
        evidences = [Evidence(**e) for e in ev_raw]

        # 3) 生成 Requirement 候选
        reqs = self._build_requirements(evidences, themes)

        # 4) 逐条判定
        items: List[MatchResult] = []
        for r in reqs:
            # 为每个要求挑选最相关的若干证据（这里简单用前N条，你可用相似度再筛）
            ev_for_r = evidences[:4]
            items.append(self._judge_one(r, doc_extract["fields"], ev_for_r))

        # 5) 汇总
        pass_cnt = sum(1 for i in items if i.decision == "pass")
        fail_cnt = sum(1 for i in items if i.decision == "fail")
        na_cnt   = sum(1 for i in items if i.decision == "not_applicable")
        un_cnt   = sum(1 for i in items if i.decision == "uncertain")
        overall = "fail" if fail_cnt > 0 else ("partial" if un_cnt or na_cnt else "pass")

        summary = ComplianceSummary(
            doc_id=doc_id or "ad_hoc",
            checked_at=dt.datetime.now().isoformat(),
            overall=overall,
            stats={"pass": pass_cnt, "fail": fail_cnt, "na": na_cnt, "uncertain": un_cnt},
            items=items
        )
        return summary.dict()

    async def _arun(self, **kwargs):
        return self._run(**kwargs)
```

### 判定用的 LLM 封装（示例）

```python
# your_llm.py
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI

class JudgeOut(BaseModel):
    decision: str    # pass/fail/not_applicable/uncertain
    reason: str
    confidence: float

def structured_judge(requirement, evidence_text, doc_summary) -> JudgeOut:
    schema = StructuredOutputParser.from_names_and_descriptions({
        "decision": "pass|fail|not_applicable|uncertain",
        "reason": "基于证据的简要理由（不可臆测）",
        "confidence": "0~1之间的小数"
    })
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是合规审查助手，只能依据提供的证据进行判断。"),
        ("system", f"严格按以下JSON输出：\n{schema.get_format_instructions()}"),
        ("human", 
         f"要求: {requirement.title}\n"
         f"条文: {requirement.statement}\n"
         f"上传文档摘要: {doc_summary}\n"
         f"证据片段(仅可据此判断):\n{evidence_text}\n"
         "若证据不足以支持“合格”，请返回 fail 或 uncertain，并说明原因。")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    out = llm.invoke(prompt.format_messages())
    data = schema.parse(out.content)
    return JudgeOut(**data)
```

### 规则校验函数（示例）

```python
# your_rules.py
def run_rule_checks_for_requirement(requirement, doc_fields):
    """
    示例：若条文涉及“必须明确责任人/签署日期”，用抽取到的字段直接判定。
    """
    text = (requirement.statement or "").lower()
    have_rp = bool(doc_fields.get("responsible_parties"))
    have_date = bool(doc_fields.get("dates"))
    # 极简示意
    if "责任人" in requirement.statement and not have_rp:
        return {"decision":"fail", "reason":"未抽取到责任人信息", "confidence":0.9}
    if "日期" in requirement.statement and not have_date:
        return {"decision":"fail", "reason":"未抽取到签署/生效日期", "confidence":0.85}
    # 若无法仅靠规则断言合格
    if "责任人" in requirement.statement and have_rp:
        return {"decision":"pass", "reason":"已检测到责任人字段", "confidence":0.75}
    return None  # 交给LLM裁决
```

------

## 3) 不符合项报告工具：`NCReportTool`

- 输入：`ComplianceSummary`（上一步的结果）。
- 输出：
  - `report_json`：可直接写库（含不符合项列表、风险等级、整改建议占位）；
  - `report_md`：Markdown 摘要（用于下载或预览）；
  - 可扩展导出 PDF（在后端转化）。

```python
# tools/nc_report_tool.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict
from schemas import ComplianceSummary

class NCReportInput(BaseModel):
    summary: Dict = Field(..., description="ComplianceSummary 字典")

class NCReportTool(BaseTool):
    name = "nc_report_generator"
    description = "将合规判定结果转为不符合项报告（JSON + Markdown 摘要）。"
    args_schema = NCReportInput

    def _run(self, summary: Dict) -> Dict:
        cs = ComplianceSummary(**summary)
        nonconfs = [it for it in cs.items if it.decision == "fail"]
        title = f"不符合项报告 - {cs.doc_id}"
        md_lines = [f"# {title}",
                    f"- 检查时间：{cs.checked_at}",
                    f"- 总体结论：**{cs.overall.upper()}**",
                    f"- 统计：{cs.stats}",
                    "",
                    "## 不符合项清单"]
        for i, it in enumerate(nonconfs, start=1):
            md_lines += [
              f"### {i}. {it.requirement.title} ({it.requirement.severity})",
              f"- 条文：{it.requirement.statement}",
              f"- 结论：**FAIL**",
              f"- 原因：{it.reason}",
              f"- 证据：",
            ]
            for ev in it.matched_evidence[:2]:
                md_lines.append(f"  - {ev.source_id} · {ev.chunk_id} · p{ev.page} · “{(ev.text_snippet or '')[:120]}…”")
            md_lines.append("")
        # 若全通过，也给出通过说明
        if not nonconfs:
            md_lines.append("> 本次检查未发现不符合项。")
        report_md = "\n".join(md_lines)

        return {
            "report_title": title,
            "report_json": {
                "doc_id": cs.doc_id,
                "overall": cs.overall,
                "stats": cs.stats,
                "nonconformities": [it.dict() for it in nonconfs]
            },
            "report_md": report_md
        }

    async def _arun(self, **kwargs):
        return self._run(**kwargs)
```

------

## 在 Agent 中挂载与调用

```python
# agent_init.py
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools.regdoc_search_tool import RegDocSearchTool
from tools.compliance_check_tool import ComplianceCheckTool
from tools.nc_report_tool import NCReportTool

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [RegDocSearchTool(), ComplianceCheckTool(), NCReportTool()]
    return initialize_agent(
        tools=tools, llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
```

------

## Django REST API（简化示例）

```python
# views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from tools.compliance_check_tool import ComplianceCheckTool
from tools.nc_report_tool import NCReportTool

check_tool = ComplianceCheckTool()
report_tool = NCReportTool()

@api_view(["POST"])
def compliance_check(request):
    payload = request.data
    result = check_tool._run(
        doc_id=payload.get("doc_id"),
        text=payload.get("text"),
        themes=payload.get("themes", []),
        top_k=payload.get("top_k", 8)
    )
    return Response(result)

@api_view(["POST"])
def nc_report(request):
    summary = request.data  # 应为 ComplianceSummary 字典
    result = report_tool._run(summary=summary)
    return Response(result)
```

------

## 生产落地建议（关键点）

- **模板化 Requirement**：把常用法规条款沉淀为“机器可读模板”（req_id/statement/检查规则指针），LLM 只做“语义补全/边界判断”。
- **强溯源**：所有“pass/fail”必须附证据 chunk（doc_id, chunk_id, page, snippet）；前端可做高亮定位。
- **可审计**：记录模型版本、提示词版本、Faiss 索引版本。
- **批处理/异步**：大文件 & 多文档用 Celery；结果可轮询或回调。
- **阈值策略**：对 `uncertain` 设提醒/复核队列；对 `fail` 根据 `severity` 触发通知（邮件/IM Webhook）。

------

如果你把你目前“法规条款模板/类别列表”发我，我可以把 `run_rule_checks_for_requirement` 里的一堆检查函数写细（比如签署日期必填、信息安全等保等级、数据留存期限、责任人角色映射、KPI阈值等），并给你配好**few-shot**判定提示词。



> 其实**上传文件**本身并不一定需要 Agent，它完全可以是一个普通的 API 流程（`POST /api/files/upload/` → 存储 → 解析 → 入库）。
>
> 之所以在**大模型工具箱平台**里会考虑用到 **Agent 模式**，主要是为了 **自动化调用多个工具**。场景区别可以这样理解：
>
> ------
>
> ### 1. 普通上传 API（不需要 Agent）
>
> - 用户上传 PDF/Docx → 系统解析切分 → 向量化写入 FAISS。
> - 全流程是 **确定的**，不需要 LLM 来“决定”下一步干什么。
>    👉 这种情况直接写 Django API 就好。
>
> ------
>
> ### 2. Agent 模式适合的情况
>
> 当需求是 **模糊的/复合的/用户用自然语言提问**时，Agent 才有价值：
>
> - 例如用户输入一句话：
>
>   > “帮我检查这份合同是否符合公司采购标准，如果需要，还帮我在网上查一下相关法规。”
>
> - Agent 就会：
>
>   1. 先调用 **FileIngestTool**（上传文件解析+入库）；
>   2. 再调用 **ComplianceCheckTool**（跑合规检查）；
>   3. 如果合规性规则不够，还会触发 **BaiduSearchTool** 去检索外部法规；
>   4. 最后组合结果生成一份报告。
>
> 👉 这里 **Agent 的价值** = 自动选择和编排工具，而不是单纯接文件。
>
> ------
>
> ### 3. 如何结合
>
> 你可以这样设计：
>
> - **上传 API**：普通 REST API，走固定流程，效率高。
> - **Agent API**：当用户用自然语言发起任务时，Agent 才会串联调用 “上传→检索→合规→搜索→报告”。
>
> 









