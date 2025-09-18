# 大模型 AI 工具箱平台（面试版 + 代码骨架）

> **技术栈**：Python · Django/DRF · LangChain · FAISS · HuggingFace Embeddings · Crawl4AI（可选）· SerpAPI/自建百度检索 · Docker · RESTful API

------

## 1) 电梯陈述（30–60 秒）

- 我做了一个**面向企业知识与外部网页的 RAG 工具箱**：聚合第三方 API，通过 LangChain 封装为可调用的 Tools，支持**网页抓取 + 企业文档向量化 + 检索问答 + 合规校验**一条龙，统一以 REST API 暴露。
- 核心亮点：可插拔工具（Tool Registry）、多源索引（Web + 本地/对象存储）、**中文语料优化（bge-zh）**、FAISS 百万级向量检索、合规规则 DSL，容器化一键部署。

------

## 2) 总体架构

```
┌──────────┐  REST  ┌──────────────┐  调度/封装   ┌─────────────┐
│  前端/调用方│◀──────▶│  Django + DRF │◀──────────▶│  LangChain   │
└──────────┘        └──────────────┘  Tools/Agents  └─────────────┘
                                          ▲    ▲         ▲
                 文档上传/管理  元数据/审计  │    │         │
                                          │    │         │
         ┌──────────┐    ┌─────────┐   ┌─────────────┐ ┌─────────┐
         │  存储/OSS │    │  MySQL  │   │  外部检索API │ │ Crawl4AI │
         └──────────┘    └─────────┘   └─────────────┘ └─────────┘
                 ▲                         │            
                 │向量化/索引              │Web 搜索      
                 │                         ▼            
                 ┌──────────────────────────────────────┐
                 │              FAISS 向量库             │
                 └──────────────────────────────────────┘
```

------

## 3) 功能与 Tools 一览

- **BaiduSearchTool**：百度检索聚合（SerpAPI *或* 自建轻量解析）
- **WebCrawlTool**：给定 URL 列表（或站点前缀）抓取正文并清洗
- **FileIngestTool**：将上传文档（PDF/Docx/TXT/MD）解析→切分→向量化→写入 FAISS
- **RAGQueryTool**：基于向量检索 + 重排（可选）进行问答，返回引用
- **ComplianceCheckTool**：读取企业标准/法规规则（YAML/JSON），对上传文件进行**合格性检查**，输出报告（JSON + Markdown）
- **SummarizeTool**：长文自动摘要（Map-Reduce 或 Refine）
- **Agent（可选）**：ReAct/Tool-Calling 代理，自动选择上述工具

------

## 4) 依赖（requirements.txt）

```txt
Django>=5.0
djangorestframework>=3.15
langchain>=0.2.11
langchain-community>=0.2.10
faiss-cpu>=1.8.0
sentence-transformers>=3.0.1
unstructured>=0.15.7
pypdf>=4.2.0
python-docx>=1.1.2
beautifulsoup4>=4.12.3
httpx>=0.27.0
uvicorn>=0.30.1
python-dotenv>=1.0.1
# 可选
crawl4ai>=0.3.10
serpapi>=0.1.5
```

------

## 5) Embeddings 与向量库初始化

```python
# app/core/vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# 中文/中英混合：bge-small-zh-v1.5 或 bge-m3（多语种）
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

def build_faiss_from_texts(texts, metadatas=None):
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
```

------

## 6) 文档解析与入库（FileIngestTool）

```python
# app/tools/file_ingest.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from app.core.vector_store import build_faiss_from_texts
import os, glob

SUPPORTS = (".pdf", ".txt", ".md", ".docx")

def load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = PyPDFLoader(path).load()
    else:
        pages = UnstructuredFileLoader(path).load()
    return pages

def chunk_docs(docs, sz=800, overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=sz, chunk_overlap=overlap)
    return splitter.split_documents(docs)

class FileIngestTool:
    name = "file_ingest"
    description = "解析并向量化上传文档，写入FAISS，返回向量库对象或持久化路径。"

    def __init__(self, persist_dir: str = "./var/indexes"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

    def __call__(self, folder_or_file: str, index_name: str):
        paths = []
        if os.path.isdir(folder_or_file):
            for ext in SUPPORTS:
                paths += glob.glob(os.path.join(folder_or_file, f"**/*{ext}"), recursive=True)
        else:
            paths = [folder_or_file]
        all_docs = []
        for p in paths:
            all_docs.extend(load_file(p))
        chunks = chunk_docs(all_docs)
        texts = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        vs = build_faiss_from_texts(texts, metas)
        vs.save_local(os.path.join(self.persist_dir, index_name))
        return {"ok": True, "index_path": os.path.join(self.persist_dir, index_name)}
```

------

## 7) RAG 检索问答（RAGQueryTool）

```python
# app/tools/rag_query.py
from langchain_community.vectorstores import FAISS
from app.core.vector_store import embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI  # 可替换为本地大模型适配器

SYS_TMPL = (
"你是企业知识助手。基于给定上下文回答问题；若未知，直说不知道。\n"
"回答使用中文，列出引用片段的文件与页码（若可用）。"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYS_TMPL),
    ("human", "问题：{question}\n\n上下文：\n{context}")
])

class RAGQueryTool:
    name = "rag_query"
    description = "从FAISS索引中检索并回答问题，返回答案及引用。"

    def __init__(self, index_path: str):
        self.db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  # 占位，可替换

    def format_ctx(self, docs):
        out = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "-")
            out.append(f"[source={src} page={page}]\n{d.page_content}")
        return "\n\n".join(out)

    def __call__(self, question: str, k: int = 5):
        docs = self.db.similarity_search(question, k=k)
        ctx = self.format_ctx(docs)
        msg = PROMPT.format_messages(question=question, context=ctx)
        ans = self.llm.invoke(msg)
        cites = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
        return {"answer": ans.content, "citations": cites}
```

------

## 8) 百度搜索工具（BaiduSearchTool）

> 两种实现：优先 **SerpAPI** 的 baidu 引擎（简单、稳定）；或**轻量 HTML 解析**（不建议大规模爬取，注意站点条款）。

```python
# app/tools/baidu_search.py
import httpx, re
from bs4 import BeautifulSoup
from serpapi import GoogleSearch  # 也支持 engine='baidu'

class BaiduSearchTool:
    name = "baidu_search"
    description = "百度站内搜索工具，返回title/url/snippet 列表。支持 SerpAPI 或简易解析。"

    def __init__(self, serpapi_key: str | None = None):
        self.key = serpapi_key

    def serpapi(self, q, num=10):
        params = {"engine": "baidu", "q": q, "api_key": self.key, "num": num}
        results = GoogleSearch(params).get_dict()
        out = []
        for r in results.get("organic_results", []):
            out.append({"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")})
        return out

    def lightweight(self, q, num=10):
        # 轻量解析（仅示例），实际使用需遵守 Robots 与站点条款
        url = f"https://www.baidu.com/s?wd={q}"
        html = httpx.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        items, out = soup.select(".result"), []
        for it in items[:num]:
            title = it.select_one("h3").get_text(strip=True) if it.select_one("h3") else ""
            a = it.select_one("a")
            href = a.get("href") if a else ""
            desc = it.get_text(" ", strip=True)
            out.append({"title": title, "url": href, "snippet": desc[:200]})
        return out

    def __call__(self, q: str, num: int = 8):
        if self.key:
            return self.serpapi(q, num)
        return self.lightweight(q, num)
```

------

## 9) 合规性检查工具（ComplianceCheckTool）

> 将**企业标准/法规**抽象为**规则 DSL**（YAML/JSON）：支持**正则检查、关键字包含、数值阈值、日期时效**等；输出**JSON+Markdown 报告**。

```python
# app/tools/compliance.py
import re, json, yaml, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

@dataclass
class Rule:
    id: str
    title: str
    type: str   # regex/keyword/threshold/date_valid
    target: str # text/meta
    pattern: str | None = None
    keywords: List[str] | None = None
    field: str | None = None
    op: str | None = None   # gt/gte/lt/lte/eq
    value: float | None = None
    date_field: str | None = None
    not_before: str | None = None
    not_after: str | None = None

class ComplianceCheckTool:
    name = "compliance_check"
    description = "依据企业标准/法规规则文件，对指定文档进行合规检查，输出报告。"

    def load_rules(self, path: str) -> List[Rule]:
        data = yaml.safe_load(open(path, "r", encoding="utf-8"))
        return [Rule(**r) for r in data.get("rules", [])]

    def load_text(self, file_path: str) -> str:
        if file_path.lower().endswith(".pdf"):
            docs = PyPDFLoader(file_path).load()
        else:
            docs = UnstructuredFileLoader(file_path).load()
        return "\n\n".join([d.page_content for d in docs])

    def check(self, text: str, meta: Dict[str, Any], rules: List[Rule]):
        findings = []
        def add(rule, ok, detail):
            findings.append({"rule_id": rule.id, "title": rule.title, "pass": ok, "detail": detail})
        for r in rules:
            if r.type == "regex":
                ok = bool(re.search(r.pattern, text, re.I|re.M))
                add(r, ok, f"pattern={r.pattern}")
            elif r.type == "keyword":
                ok = all(kw.lower() in text.lower() for kw in (r.keywords or []))
                add(r, ok, f"keywords={r.keywords}")
            elif r.type == "threshold":
                val = float(meta.get(r.field, "nan"))
                ops = {"gt": lambda a,b: a>b, "gte": lambda a,b: a>=b, "lt": lambda a,b: a<b, "lte": lambda a,b: a<=b, "eq": lambda a,b: a==b}
                ok = False if str(val)=="nan" else ops[r.op](val, r.value)
                add(r, ok, f"{r.field} {r.op} {r.value}, actual={val}")
            elif r.type == "date_valid":
                val = meta.get(r.date_field)
                try:
                    d = dt.date.fromisoformat(val)
                    nb = dt.date.fromisoformat(r.not_before) if r.not_before else None
                    na = dt.date.fromisoformat(r.not_after) if r.not_after else None
                    ok = (nb is None or d>=nb) and (na is None or d<=na)
                except Exception:
                    ok=False
                add(r, ok, f"{r.date_field} in [{r.not_before},{r.not_after}], actual={val}")
        passed = all(x["pass"] for x in findings) if findings else True
        return {"passed": passed, "items": findings}

    def to_markdown(self, report: Dict[str, Any]) -> str:
        lines = ["# 合规性报告", f"总体结论：{'✅ 合格' if report['passed'] else '❌ 不合格'}", "", "## 明细"]
        for it in report["items"]:
            icon = "✅" if it["pass"] else "❌"
            lines.append(f"- {icon} [{it['rule_id']}] {it['title']}  —— {it['detail']}")
        return "\n".join(lines)

    def __call__(self, file_path: str, rule_yaml: str, meta: Dict[str, Any] | None = None):
        rules = self.load_rules(rule_yaml)
        text = self.load_text(file_path)
        meta = meta or {}
        result = self.check(text, meta, rules)
        md = self.to_markdown(result)
        return {"report": result, "markdown": md}
```

**规则 YAML 示例（rules/enterprise.yaml）**

```yaml
rules:
  - id: STD-1
    title: 必须包含“安全”章节
    type: keyword
    target: text
    keywords: ["安全", "安全要求"]
  - id: STD-2
    title: 质保期阈值
    type: threshold
    field: warranty_months
    op: gte
    value: 12
  - id: STD-3
    title: 合同签署日期在有效区间
    type: date_valid
    date_field: contract_date
    not_before: "2024-01-01"
    not_after: "2026-12-31"
```

------

## 10) Web 抓取工具（可选，Crawl4AI 或轻量版）

```python
# app/tools/web_crawl.py
from crawl4ai import WebCrawler  # 可选依赖

class WebCrawlTool:
    name = "web_crawl"
    description = "抓取给定URL正文，去噪清洗后返回文本，用于索引或分析。"

    def __call__(self, url: str):
        crawler = WebCrawler()
        out = crawler.run(url)
        return {"title": out.title, "text": out.clean_text, "links": out.links}
```

------

## 11) Tool 注册与 Agent（可选）

```python
# app/agent/registry.py
from app.tools.baidu_search import BaiduSearchTool
from app.tools.file_ingest import FileIngestTool
from app.tools.rag_query import RAGQueryTool
from app.tools.compliance import ComplianceCheckTool
from app.tools.web_crawl import WebCrawlTool

class ToolRegistry:
    def __init__(self):
        self.tools = {}
    def add(self, name, tool):
        self.tools[name] = tool
    def get(self, name):
        return self.tools[name]

registry = ToolRegistry()
registry.add("baidu_search", BaiduSearchTool())
registry.add("file_ingest", FileIngestTool())
# RAGQueryTool 需传索引路径，运行时添加：registry.add("rag_query", RAGQueryTool(index_path))
registry.add("compliance_check", ComplianceCheckTool())
registry.add("web_crawl", WebCrawlTool())
```

------

## 12) Django 接口（示例）

```python
# app/api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from app.agent.registry import registry
from app.tools.rag_query import RAGQueryTool
import os

class BaiduSearchAPI(APIView):
    def post(self, request):
        q = request.data.get("q")
        n = int(request.data.get("n", 8))
        res = registry.get("baidu_search")(q, n)
        return Response({"code":0, "data": res})

class IngestAPI(APIView):
    def post(self, request):
        f = request.FILES.get("file")
        index = request.data.get("index", "default")
        path = default_storage.save(f.name, f)
        res = registry.get("file_ingest")(default_storage.path(path), index)
        return Response({"code":0, "data": res})

class RAGQueryAPI(APIView):
    def post(self, request):
        question = request.data.get("q")
        index_path = request.data.get("index_path")
        tool = RAGQueryTool(index_path)
        out = tool(question)
        return Response({"code":0, "data": out})

class ComplianceAPI(APIView):
    def post(self, request):
        f = request.FILES.get("file")
        rules = request.FILES.get("rules")
        meta = request.data.get("meta", {})
        fpath = default_storage.save(f.name, f)
        rpath = default_storage.save(rules.name, rules)
        tool = registry.get("compliance_check")
        out = tool(default_storage.path(fpath), default_storage.path(rpath), meta)
        return Response({"code":0, "data": out})
```

**URL 路由（urls.py）**

```python
from django.urls import path
from app.api.views import BaiduSearchAPI, IngestAPI, RAGQueryAPI, ComplianceAPI
urlpatterns = [
    path("api/tools/baidu_search/", BaiduSearchAPI.as_view()),
    path("api/rag/ingest/", IngestAPI.as_view()),
    path("api/rag/query/", RAGQueryAPI.as_view()),
    path("api/compliance/check/", ComplianceAPI.as_view()),
]
```

------

## 13) cURL 快速演示

```bash
# 1) 百度搜索
curl -X POST http://localhost:8000/api/tools/baidu_search/ \
  -H 'Content-Type: application/json' \
  -d '{"q":"企业安全生产 法规 要点", "n":5}'

# 2) 文档入库
curl -F "file=@docs/handbook.pdf" -F "index=corp_kb" \
  http://localhost:8000/api/rag/ingest/

# 3) RAG 问答
curl -X POST http://localhost:8000/api/rag/query/ \
  -H 'Content-Type: application/json' \
  -d '{"q":"本公司消防检查频率是什么？","index_path":"./var/indexes/corp_kb"}'

# 4) 合规检查（含元数据）
curl -F "file=@docs/contract.pdf" -F "rules=@rules/enterprise.yaml" \
  -F 'meta={"warranty_months":18, "contract_date":"2025-07-01"};type=application/json' \
  http://localhost:8000/api/compliance/check/
```

------

## 14) Docker 与启动

```yaml
# docker-compose.yml（精简）
version: '3.9'
services:
  web:
    build: .
    ports: ["8000:8000"]
    volumes: ["./var:/app/var"]
  # 如需数据库/消息队列可继续扩展
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python","manage.py","runserver","0.0.0.0:8000"]
```

------

## 15) 面试 Q&A 备选

- **为什么选 FAISS + bge-zh？** 中文检索表现好、速度快、离线可用；对隐私合规友好。
- **如何做长文与多文件的引用可追溯？** 切分粒度 + `source/page` 元数据，答案中返回引用清单。
- **合规检查如何扩展？** 通过规则 DSL（新增 type、组合逻辑 AND/OR、跨文档对比、LLM 验证器二次判定）。
- **外网检索可靠性？** 首选官方/SerpAPI，必要时加缓存与快照；遵守网站条款与 Robots。
- **如何避免幻觉？** RAG 上下文可视化 + 置信度阈值 + 未命中兜底“我不知道”。



> 已内置工具（可直接用/按需改）
>
> 1. **BaiduSearchTool**（SerpAPI 或轻量解析版）
> 2. **FileIngestTool**（上传文档→解析→切分→向量化→FAISS）
> 3. **RAGQueryTool**（向量检索+大模型回答，携带引用）
> 4. **ComplianceCheckTool**（基于 YAML 规则的合规检查，输出 JSON+Markdown 报告）
> 5. **WebCrawlTool**（可选：Crawl4AI 抓取正文）
>
> - Tool Registry、Django REST API、cURL 演示、Docker 启动脚手架。







# 大模型 AI 工具箱：数据库与权限设计（Django + MySQL）

> 目标：支持**多租户**、**RBAC 权限**、**对象级（Object-Level）访问控制**，覆盖文件/索引/规则/报告等核心实体；兼容 Django/DRF；与 FAISS 本地索引配合使用（元数据入库，向量文件落磁盘/对象存储）。

------

## 1) 实体与关系（ER 概览）

```
┌────────────┐   1..*   ┌───────────────┐   1..*   ┌───────────┐
│ Organization│────────▶│   Project      │────────▶│  Document │
└──────┬──────┘          └──────┬────────┘          └────┬──────┘
       │1..*                    1│                         1│
       ▼                         ▼                          ▼
┌────────────┐          ┌──────────────┐          ┌────────────────┐
│  User      │          │   Index      │          │ ComplianceRule │
└──────┬─────┘          └──────┬───────┘          └──────┬─────────┘
       │1..*                    │1..*                      │1..*
       ▼                        ▼                          ▼
┌────────────┐          ┌──────────────┐          ┌────────────────┐
│ Membership │          │  Job/Task    │          │ ComplianceReport│
└──────┬─────┘          └──────────────┘          └────────────────┘
       │
       ▼
┌────────────┐     ┌───────────────┐
│ Role       │◀───▶│ PermissionMap │  (RBAC: 角色-权限映射)
└────────────┘     └───────────────┘

+ AuditLog(审计)、APIKey(机器对接)、WebSearchCache(外部检索缓存)
```

------

## 2) MySQL 表结构（DDL）

> 兼容 Django ORM；`org_id + project_id` 做多租户边界；对象级访问通过表字段 + 代码校验实现。

```sql
-- 组织 / 多租户
CREATE TABLE organization (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  name          VARCHAR(128) NOT NULL,
  code          VARCHAR(64) UNIQUE,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 用户（可复用 Django 自带 auth_user，这里给出精简版参考）
CREATE TABLE app_user (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  email         VARCHAR(255) UNIQUE,
  password_hash VARCHAR(255),
  display_name  VARCHAR(128),
  is_active     TINYINT(1) DEFAULT 1,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 组织成员关系（及角色）
CREATE TABLE membership (
  id        BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id    BIGINT NOT NULL,
  user_id   BIGINT NOT NULL,
  role      VARCHAR(32) NOT NULL, -- owner/admin/editor/viewer
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_user (org_id, user_id),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 角色权限映射（RBAC）
CREATE TABLE permission_map (
  id        BIGINT PRIMARY KEY AUTO_INCREMENT,
  role      VARCHAR(32) NOT NULL,
  action    VARCHAR(64) NOT NULL,  -- e.g., doc.upload, doc.view, index.build, rule.edit
  scope     VARCHAR(32) NOT NULL   -- org/project/object 全局/租户级/项目级/对象级
) ENGINE=InnoDB;

-- 项目（同一组织下的逻辑隔离）
CREATE TABLE project (
  id         BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id     BIGINT NOT NULL,
  name       VARCHAR(128) NOT NULL,
  code       VARCHAR(64),
  created_by BIGINT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_name (org_id, name),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 文档（原文件元数据；正文不必入库，存储路径+hash）
CREATE TABLE document (
  id           BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id       BIGINT NOT NULL,
  project_id   BIGINT NOT NULL,
  filename     VARCHAR(255) NOT NULL,
  storage_uri  VARCHAR(512) NOT NULL,
  content_hash CHAR(64) NOT NULL, -- sha256
  mime_type    VARCHAR(64),
  pages        INT,
  uploaded_by  BIGINT,
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_org_project (org_id, project_id),
  UNIQUE KEY uk_org_hash (org_id, content_hash),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
  FOREIGN KEY (uploaded_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 向量索引：FAISS 的磁盘路径 + 语义模型信息
CREATE TABLE vec_index (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT NOT NULL,
  project_id  BIGINT NOT NULL,
  name        VARCHAR(128) NOT NULL,
  model_name  VARCHAR(128) NOT NULL,
  dim         INT NOT NULL,
  persist_uri VARCHAR(512) NOT NULL, -- e.g., file:///var/indexes/xxx
  total_chunks INT DEFAULT 0,
  created_by  BIGINT,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_proj_name (org_id, project_id, name),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 文档与索引的映射（哪些文档被编入了哪个索引）
CREATE TABLE index_document (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  index_id    BIGINT NOT NULL,
  document_id BIGINT NOT NULL,
  chunks      INT DEFAULT 0,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_idx_doc (index_id, document_id),
  FOREIGN KEY (index_id) REFERENCES vec_index(id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 合规规则（DSL YAML/JSON 存储；可版本化）
CREATE TABLE compliance_rule (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT NOT NULL,
  project_id  BIGINT NOT NULL,
  code        VARCHAR(64) NOT NULL, -- 规则集编号
  title       VARCHAR(255) NOT NULL,
  version     VARCHAR(32) DEFAULT '1.0',
  fmt         VARCHAR(16) DEFAULT 'yaml',
  content     MEDIUMTEXT NOT NULL, -- 规则 DSL
  created_by  BIGINT,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_proj_code_ver (org_id, project_id, code, version),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 合规报告（对某文档在某规则集下的检查结果）
CREATE TABLE compliance_report (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id        BIGINT NOT NULL,
  project_id    BIGINT NOT NULL,
  document_id   BIGINT NOT NULL,
  rule_code     VARCHAR(64) NOT NULL,
  rule_version  VARCHAR(32) NOT NULL,
  passed        TINYINT(1) NOT NULL,
  json_result   MEDIUMTEXT NOT NULL,
  md_render     MEDIUMTEXT,
  created_by    BIGINT,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_doc (document_id),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 任务/作业（解析、索引构建、报告生成等）
CREATE TABLE job_task (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT NOT NULL,
  project_id  BIGINT NOT NULL,
  type        VARCHAR(64) NOT NULL,  -- ingest/index/build/report
  status      VARCHAR(32) NOT NULL,  -- pending/running/succeeded/failed
  payload     JSON,
  result      JSON,
  created_by  BIGINT,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE,
  FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES app_user(id)
) ENGINE=InnoDB;

-- 外部搜索缓存（如百度/法规站点），降低重复请求
CREATE TABLE web_search_cache (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT NOT NULL,
  query_hash  CHAR(64) NOT NULL,
  engine      VARCHAR(32) NOT NULL, -- baidu/serpapi/... 
  json_result MEDIUMTEXT NOT NULL,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_q_e (org_id, query_hash, engine)
) ENGINE=InnoDB;

-- API Key（机器集成/服务间调用）
CREATE TABLE api_key (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT NOT NULL,
  name        VARCHAR(64) NOT NULL,
  token_hash  CHAR(64) NOT NULL,
  scopes      VARCHAR(255), -- 逗号分隔：rag.read,doc.write,rule.admin
  expires_at  DATETIME,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_org_name (org_id, name),
  FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 审计日志
CREATE TABLE audit_log (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  org_id      BIGINT,
  project_id  BIGINT,
  actor_id    BIGINT,
  action      VARCHAR(64) NOT NULL,  -- doc.upload, index.build, rule.edit, report.create
  target_type VARCHAR(32),           -- document/index/rule/report
  target_id   BIGINT,
  ip          VARCHAR(64),
  user_agent  VARCHAR(255),
  detail      JSON,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
```

------

## 3) Django 模型（精简映射）

> 若使用 Django 自带 `auth.User` 与 `Group`，可将 `app_user` 替换为外键到 `settings.AUTH_USER_MODEL`。

```python
# apps/core/models.py
from django.db import models
from django.conf import settings

class Organization(models.Model):
    name = models.CharField(max_length=128)
    code = models.CharField(max_length=64, unique=True, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Membership(models.Model):
    OWNER='owner'; ADMIN='admin'; EDITOR='editor'; VIEWER='viewer'
    ROLES=((OWNER,'Owner'),(ADMIN,'Admin'),(EDITOR,'Editor'),(VIEWER,'Viewer'))
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    role = models.CharField(max_length=16, choices=ROLES)
    created_at = models.DateTimeField(auto_now_add=True)

class Project(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    code = models.CharField(max_length=64, null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

class Document(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    storage_uri = models.CharField(max_length=512)
    content_hash = models.CharField(max_length=64)
    mime_type = models.CharField(max_length=64, null=True, blank=True)
    pages = models.IntegerField(null=True, blank=True)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

class VecIndex(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    model_name = models.CharField(max_length=128)
    dim = models.IntegerField()
    persist_uri = models.CharField(max_length=512)
    total_chunks = models.IntegerField(default=0)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        unique_together=(('org','project','name'),)

class IndexDocument(models.Model):
    index = models.ForeignKey(VecIndex, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunks = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        unique_together=(('index','document'),)

class ComplianceRule(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    code = models.CharField(max_length=64)
    title = models.CharField(max_length=255)
    version = models.CharField(max_length=32, default='1.0')
    fmt = models.CharField(max_length=16, default='yaml')
    content = models.TextField()
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        unique_together=(('org','project','code','version'),)

class ComplianceReport(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    rule_code = models.CharField(max_length=64)
    rule_version = models.CharField(max_length=32)
    passed = models.BooleanField()
    json_result = models.TextField()
    md_render = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

class JobTask(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    type = models.CharField(max_length=64)
    status = models.CharField(max_length=32)
    payload = models.JSONField(null=True, blank=True)
    result = models.JSONField(null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class AuditLog(models.Model):
    org = models.ForeignKey(Organization, null=True, on_delete=models.SET_NULL)
    project = models.ForeignKey(Project, null=True, on_delete=models.SET_NULL)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=64)
    target_type = models.CharField(max_length=32, null=True, blank=True)
    target_id = models.BigIntegerField(null=True, blank=True)
    ip = models.CharField(max_length=64, null=True, blank=True)
    user_agent = models.CharField(max_length=255, null=True, blank=True)
    detail = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
```

------

## 4) 权限模型

### 4.1 角色与动作（示例）

- **owner**：租户级一切权限
- **admin**：项目创建、规则管理、索引管理、成员管理（限组织内）
- **editor**：上传文档、构建索引、发起合规检查、查看报告
- **viewer**：只读（查看文档元数据、查询 RAG、查看报告）

**动作枚举**（可落库 `permission_map`，也可硬编码）：

- `org.manage`, `project.manage`, `member.manage`
- `doc.upload`, `doc.delete`, `doc.view`
- `index.build`, `index.delete`, `index.view`
- `rule.edit`, `rule.view`
- `report.create`, `report.view`

### 4.2 DRF 权限类（对象级校验）

```python
# apps/core/permissions.py
from rest_framework.permissions import BasePermission, SAFE_METHODS
from .models import Membership, Project

ROLE_LEVEL = {"viewer":1, "editor":2, "admin":3, "owner":4}

class IsOrgMember(BasePermission):
    def has_permission(self, request, view):
        org_id = view.kwargs.get('org_id') or request.data.get('org_id')
        if not org_id or not request.user.is_authenticated:
            return False
        return Membership.objects.filter(org_id=org_id, user=request.user).exists()

class HasRoleOrReadOnly(BasePermission):
    def __init__(self, min_role="editor"):
        self.min_role = min_role
    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        org_id = view.kwargs.get('org_id') or request.data.get('org_id')
        m = Membership.objects.filter(org_id=org_id, user=request.user).first()
        return bool(m and ROLE_LEVEL[m.role] >= ROLE_LEVEL[self.min_role])

class ProjectScopedPermission(BasePermission):
    """对象级校验：请求涉及的 project/org 是否与用户成员关系一致"""
    def has_object_permission(self, request, view, obj):
        org_id = getattr(obj, 'org_id', None) or getattr(getattr(obj,'org',None),'id', None)
        if not org_id:
            return False
        return Membership.objects.filter(org_id=org_id, user=request.user).exists()
```

**视图中使用**：

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from .permissions import IsOrgMember, HasRoleOrReadOnly

class DocumentUploadAPI(APIView):
    permission_classes = [IsOrgMember, HasRoleOrReadOnly(min_role='editor')]
    def post(self, request, org_id, project_id):
        # TODO: 校验该 project 属于 org，并记录审计日志
        return Response({"code":0, "msg":"ok"})
```

### 4.3 行级过滤（QuerySet 约束）

```python
# apps/core/query.py
from .models import Document, Membership

def scoped_documents(user, org_id, project_id=None):
    if not Membership.objects.filter(org_id=org_id, user=user).exists():
        return Document.objects.none()
    qs = Document.objects.filter(org_id=org_id)
    if project_id:
        qs = qs.filter(project_id=project_id)
    return qs
```

------

## 5) 认证与 API Key

- **用户认证**：JWT（`Authorization: Bearer <token>`），登录成功后携带 `org_id`/`project_id` 作为上下文。
- **API Key**：机器对接走 `X-API-Key`；服务端读取 `api_key.token_hash`（使用 SHA-256 存储，密钥仅一次性展示），校验 `scopes` 与有效期。

**中间件（简化）**：

```python
# apps/core/middleware.py
from django.utils.deprecation import MiddlewareMixin
from .models import AuditLog

class AuditMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        try:
            AuditLog.objects.create(
                org_id = getattr(request, 'org_id', None),
                project_id = getattr(request, 'project_id', None),
                actor_id = getattr(getattr(request,'user',None),'id', None),
                action = f"{request.method} {request.path}",
                ip = request.META.get('REMOTE_ADDR'),
                user_agent = request.META.get('HTTP_USER_AGENT'),
                detail = {"status": getattr(response,'status_code',None)}
            )
        except Exception:
            pass
        return response
```

------

## 6) 典型访问控制流程（上传 → 索引 → 合规）

1. **上传** `POST /orgs/{org_id}/projects/{pid}/documents/`：
   - 权限：`editor+`
   - 校验用户是 org 成员；写 `document`，生成 `job_task`（解析/切分/向量化）。
2. **构建索引** `POST /orgs/{org_id}/projects/{pid}/indexes/{name}/build`：
   - 权限：`editor+`
   - 校验 `vec_index` 归属；只允许同 org/project 的文档加入。
3. **合规检查** `POST /orgs/{org_id}/projects/{pid}/reports/`：
   - 权限：`editor+` 生成，`viewer+` 查看；
   - 选择 `document` + `rule(code,version)` 生成 `compliance_report`。

------

## 7) 种子数据（角色权限映射）

```sql
INSERT INTO permission_map(role, action, scope) VALUES
 ('owner','*','org'),
 ('admin','project.manage','org'),
 ('admin','member.manage','org'),
 ('admin','rule.edit','project'),
 ('admin','index.build','project'),
 ('editor','doc.upload','project'),
 ('editor','index.build','project'),
 ('editor','report.create','project'),
 ('viewer','doc.view','project'),
 ('viewer','index.view','project'),
 ('viewer','report.view','project');
```

------

## 8) 安全与合规加固清单

- **最小权限**：默认 viewer，操作前做 `org_id/project_id` 归属校验；
- **对象级权限**：所有 `GET/POST/DELETE` 都通过 `IsOrgMember + HasRoleOrReadOnly + ProjectScopedPermission`；
- **审计**：关键操作写 `AuditLog`；
- **密钥**：API Key 只保存哈希，开启过期与回收；
- **限流**：对外工具（百度/抓取）加频控与缓存；
- **数据隔离**：磁盘索引以 `org_id/project_id` 划分目录；
- **删除策略**：文档软删 + 引用检查（若在索引中使用，限制删除或提供重建流程）。

------

## 9) 面试要点速记

- 说明**多租户边界**：所有核心表含 `org_id`，并在查询层 **强制 where org_id=**；
- 解释**RBAC + 对象级**：角色决定“能做什么”，对象级校验决定“能对谁做”；
- **可观察性**：审计 + 任务状态 + 指标（成功率/延迟/