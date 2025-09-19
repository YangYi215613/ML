# 项目结构

```
ai-toolbox/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ deps.py
│  ├─ db/
│  │  ├─ __init__.py
│  │  ├─ models.py
│  │  ├─ session.py
│  ├─ vectorstore/
│  │  ├─ __init__.py
│  │  ├─ faiss_store.py
│  ├─ services/
│  │  ├─ ingest.py
│  │  ├─ rag.py
│  │  ├─ crawl.py
│  ├─ tools/
│  │  ├─ baidu_search.py
│  │  ├─ report_generator.py
│  │  ├─ rag_tools.py
│  ├─ agents/
│  │  ├─ compliance_agent.py
│  ├─ routers/
│  │  ├─ documents.py
│  │  ├─ chat.py
│  │  ├─ health.py
│  ├─ utils/
│  │  ├─ file_io.py
│  │  ├─ hashing.py
│  │  ├─ text.py
│  │  ├─ pdf.py
├─ data/
│  ├─ uploads/             # 原始上传文件存储
│  ├─ reports/             # 生成的合格性报告（.md/.pdf）
│  └─ index/               # FAISS 索引持久化目录
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ README.md
```

------

## 1) requirements.txt

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
pydantic==2.9.2
pydantic-settings==2.4.0
sqlmodel==0.0.22
SQLAlchemy==2.0.34
langchain==0.2.16
langchain-community==0.2.16
faiss-cpu==1.8.0.post1
openai==1.43.0
requests==2.32.3
beautifulsoup4==4.12.3
tiktoken==0.7.0
reportlab==4.2.2
```

> 说明：
>
> - 选用 `OpenAI` 作为 LLM 与 Embeddings（可换成 Azure/OpenAI 兼容接口）。
> - PDF 生成用 `reportlab`（简洁稳定）；报告同时输出 `.md`。

------

## 2) .env.example

```env
OPENAI_API_KEY=sk-xxx
# Embedding & Model 名称可按需替换
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
# 索引与数据目录
DATA_DIR=./data
UPLOAD_DIR=./data/uploads
REPORT_DIR=./data/reports
INDEX_DIR=./data/index
# 可选：百度搜索 API（若使用官方开放平台）
BAIDU_API_KEY=
BAIDU_CSE_ID=
```

------

## 3) app/config.py

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    chat_model: str = Field("gpt-4o-mini", alias="CHAT_MODEL")

    data_dir: Path = Field(Path("./data"), alias="DATA_DIR")
    upload_dir: Path = Field(Path("./data/uploads"), alias="UPLOAD_DIR")
    report_dir: Path = Field(Path("./data/reports"), alias="REPORT_DIR")
    index_dir: Path = Field(Path("./data/index"), alias="INDEX_DIR")

    # Baidu 搜索（可选）
    baidu_api_key: str | None = Field(None, alias="BAIDU_API_KEY")
    baidu_cse_id: str | None = Field(None, alias="BAIDU_CSE_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.report_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
```

------

## 4) app/db/session.py

```python
from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
from ..config import settings

DB_PATH = Path(settings.data_dir) / "toolbox.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
```

------

## 5) app/db/models.py

```python
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    original_name: str
    content_hash: str
    mime_type: str
    size: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    report_name: str
    report_path_md: str
    report_path_pdf: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

------

## 6) app/schemas.py

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocCreateResp(BaseModel):
    id: int
    filename: str
    original_name: str
    size: int
    created_at: datetime

class DocUpdateReq(BaseModel):
    new_text: Optional[str] = None

class DocListItem(BaseModel):
    id: int
    original_name: str
    filename: str
    size: int
    updated_at: datetime

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    generate_report: bool = False
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    report_id: Optional[int] = None

class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 1

class BaiduSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ReportInfo(BaseModel):
    id: int
    name: str
    created_at: datetime
```

------

## 7) app/utils/hashing.py

```python
import hashlib
from pathlib import Path

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
```

------

## 8) app/utils/text.py

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

def chunk_text(text: str) -> list[str]:
    return splitter.split_text(text)
```

------

## 9) app/utils/file_io.py

```python
from pathlib import Path
from typing import Optional

ALLOWED_SUFFIX = {'.txt', '.md', '.pdf', '.csv', '.json'}

def safe_filename(name: str) -> str:
    # 非严格实现，演示用
    return "".join(c for c in name if c.isalnum() or c in ('.','-','_')).strip("._")

def ensure_allowed(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_SUFFIX

def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return None
```

------

## 10) app/utils/pdf.py

```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from pathlib import Path

# 简易 Markdown -> PDF：这里只渲染为等宽段落，满足审阅/下载需求

def md_to_pdf(md_text: str, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    x = 20 * mm
    y = height - 20 * mm

    for line in md_text.splitlines():
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.drawString(x, y, line[:120])  # 简单裁剪
        y -= 6 * mm
    c.save()
```

------

## 11) app/vectorstore/faiss_store.py

```python
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ..config import settings

class FAISSStore:
    """封装 FAISS 的持久化加载/保存与检索器构造。"""
    def __init__(self):
        self.emb = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embedding_model)
        self.index_dir = Path(settings.index_dir)
        self.vs: FAISS | None = None

    def load(self):
        if (self.index_dir / "index.faiss").exists():
            self.vs = FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vs = FAISS.from_texts([""], embedding=self.emb)  # 空索引占位
            self.save()
        return self.vs

    def save(self):
        assert self.vs is not None
        self.vs.save_local(str(self.index_dir))

    def get(self) -> FAISS:
        if self.vs is None:
            return self.load()
        return self.vs

faiss_store = FAISSStore()
```

------

## 12) app/services/ingest.py

```python
from pathlib import Path
from sqlmodel import Session
from ..db.models import Document
from ..utils.hashing import sha256_of_file
from ..utils.file_io import read_text_file
from ..utils.text import chunk_text
from ..vectorstore.faiss_store import faiss_store
from langchain_openai import OpenAIEmbeddings
from ..config import settings

# 支持 PDF/CSV/JSON 的解析可以按需扩展，这里先处理纯文本/MD

def upsert_file(session: Session, path: Path, original_name: str, mime: str) -> Document:
    content_hash = sha256_of_file(path)

    # DB 记录落库
    doc = Document(
        filename=path.name,
        original_name=original_name,
        content_hash=content_hash,
        mime_type=mime,
        size=path.stat().st_size,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # 向量化 & 入库
    text = read_text_file(path)
    if not text:
        text = f"[BINARY PLACEHOLDER] {original_name}"
    chunks = chunk_text(text)

    vs = faiss_store.get()
    # 用 metadata 标注 doc_id，便于后续追溯
    vs.add_texts(texts=chunks, metadatas=[{"doc_id": doc.id, "filename": doc.filename}] * len(chunks))
    faiss_store.save()
    return doc


def delete_doc(session: Session, doc_id: int) -> bool:
    doc = session.get(Document, doc_id)
    if not doc:
        return False
    # 这里只删除 DB 和上传文件；FAISS 的删除可以通过重建索引实现（简化）
    try:
        (Path(settings.upload_dir) / doc.filename).unlink(missing_ok=True)
    except Exception:
        pass
    session.delete(doc)
    session.commit()
    return True


def rebuild_index(session: Session):
    """从所有文档重建 FAISS 索引（包含删除后的清理）。"""
    docs = session.exec(Document.__table__.select()).fetchall()
    texts = []
    metas = []
    for row in docs:
        path = Path(settings.upload_dir) / row.filename
        text = read_text_file(path) or f"[BINARY PLACEHOLDER] {row.original_name}"
        for chunk in chunk_text(text):
            texts.append(chunk)
            metas.append({"doc_id": row.id, "filename": row.filename})
    emb = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embedding_model)
    from langchain_community.vectorstores import FAISS
    vs = FAISS.from_texts(texts, emb, metadatas=metas) if texts else FAISS.from_texts([""], emb)
    faiss_store.vs = vs
    faiss_store.save()
```

------

## 13) app/services/rag.py

```python
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from ..vectorstore.faiss_store import faiss_store
from ..config import settings

BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful compliance assistant. Answer in Chinese by default. Cite source filenames when helpful."),
    ("human", "用户问题：{question}\n检索到的上下文：\n{context}\n请结合上下文回答。如果上下文不足，请说明假设与不确定性。")
])

llm = ChatOpenAI(api_key=settings.openai_api_key, model=settings.chat_model, temperature=0.2)


def make_retrieval_chain(top_k: int = 5):
    retriever = faiss_store.get().as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs):
        lines = []
        for d in docs:
            fn = d.metadata.get("filename", "?")
            lines.append(f"[来源:{fn}]\n{d.page_content}")
        return "\n\n".join(lines)

    chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)}
             | BASE_PROMPT
             | llm)
    return chain


def rag_answer(question: str, top_k: int = 5) -> str:
    chain = make_retrieval_chain(top_k)
    res = chain.invoke(question)
    return res.content
```

------

## 14) app/services/crawl.py

```python
import requests
from bs4 import BeautifulSoup

# 轻量 Web 抓取（演示）：生产可替换为 Crawl4AI/自建爬虫

def simple_crawl(url: str, max_pages: int = 1) -> list[dict]:
    pages = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text("\n")
        pages.append({"url": url, "text": text[:5000]})
    except Exception as e:
        pages.append({"url": url, "error": str(e)})
    return pages
```

------

## 15) app/tools/baidu_search.py

```python
from typing import List
import requests

class BaiduSearchTool:
    """百度搜索工具：优先使用官方 CSE/开放平台；若无 key，可退化到简单提示。"""
    def __init__(self, api_key: str | None = None, cse_id: str | None = None):
        self.api_key = api_key
        self.cse_id = cse_id

    def run(self, query: str, top_k: int = 5) -> List[dict]:
        if not (self.api_key and self.cse_id):
            return [{"title": "未配置百度 API，返回占位结果", "link": "", "snippet": query}]
        # 示例：如接入百度自定义搜索（伪接口；根据你实际的 API 文档对接）
        try:
            url = "https://www.googleapis.com/customsearch/v1"  # 占位：若用谷歌 CSE
            params = {"key": self.api_key, "cx": self.cse_id, "q": query, "num": top_k}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            items = data.get("items", [])
            return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]
        except Exception as e:
            return [{"title": "搜索失败", "link": "", "snippet": str(e)}]
```

> 注：百度搜索处以占位实现演示。你可替换为百度开放平台实际接口。

------

## 16) app/tools/rag_tools.py

```python
from ..services.rag import rag_answer

class RAGQueryTool:
    """封装为工具，便于接入 Agent。"""
    def run(self, query: str, top_k: int = 5) -> str:
        return rag_answer(query, top_k=top_k)
```

------

## 17) app/tools/report_generator.py

```python
from pathlib import Path
from datetime import datetime
from sqlmodel import Session
from ..db.models import Report
from ..config import settings
from ..utils.pdf import md_to_pdf

HEADER = """# 文档合规性检查报告\n\n- 生成时间：{ts}\n- 生成方式：Agent + RAG\n- 置信度：{confidence}\n\n---\n"""

SECTION = """\n## 问题/检查点\n{q}\n\n## 检索依据（节选）\n{ctx}\n\n## 结论\n{ans}\n"""

FOOTER = """\n---\n*本报告由系统自动生成，仅供参考。*\n"""


def save_report(session: Session, question: str, answer: str, context_preview: str, confidence: str = "中") -> int:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md_text = HEADER.format(ts=ts, confidence=confidence) + \
              SECTION.format(q=question, ctx=context_preview[:2000], ans=answer) + FOOTER

    report_name = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    md_path = Path(settings.report_dir) / report_name
    md_path.write_text(md_text, encoding='utf-8')

    # 同步输出为 PDF（简易渲染）
    pdf_name = report_name.replace('.md', '.pdf')
    pdf_path = Path(settings.report_dir) / pdf_name
    try:
        md_to_pdf(md_text, pdf_path)
    except Exception:
        pdf_path = None

    rec = Report(report_name=report_name, report_path_md=str(md_path), report_path_pdf=str(pdf_path) if pdf_path else None)
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec.id
```

------

## 18) app/agents/compliance_agent.py

```python
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from ..config import settings
from ..tools.baidu_search import BaiduSearchTool
from ..tools.rag_tools import RAGQueryTool

# 该 Agent 用于：根据用户问题 + 附件内容（已向量化）执行法规/标准检查

def build_agent():
    llm = ChatOpenAI(api_key=settings.openai_api_key, model=settings.chat_model, temperature=0)

    baidu = BaiduSearchTool(settings.baidu_api_key, settings.baidu_cse_id)
    rag_tool = RAGQueryTool()

    tools = [
        Tool(
            name="RAG-Query",
            func=lambda q: rag_tool.run(q, top_k=5),
            description="在内部法规/文档知识库中检索并回答合规性问题。"
        ),
        Tool(
            name="Baidu-Search",
            func=lambda q: str(baidu.run(q, top_k=3)),
            description="当知识库不足时，进行网络检索以补充公开信息。"
        )
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    return agent
```

------

## 19) app/routers/health.py

```python
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    return {"status": "ok"}
```

------

## 20) app/routers/documents.py

```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi import Form
from pathlib import Path
from typing import List
from sqlmodel import Session, select
from ..db.session import get_session
from ..db.models import Document
from ..schemas import DocCreateResp, DocListItem, DocUpdateReq
from ..config import settings
from ..utils.file_io import safe_filename, ensure_allowed, read_text_file
from ..utils.hashing import sha256_of_file
from ..services.ingest import upsert_file, delete_doc, rebuild_index

router = APIRouter(prefix="/api/docs", tags=["documents"])

@router.post("/upload", response_model=DocCreateResp)
async def upload_file(
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    mode: str = Form("insert"),  # insert|upsert
):
    fname = safe_filename(file.filename)
    path = Path(settings.upload_dir) / fname

    # 基础类型校验
    if not ensure_allowed(Path(fname)):
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    data = await file.read()
    path.write_bytes(data)

    # upsert 简化：这里统一当新文档处理；如果需要基于哈希覆盖，可在此对比 content_hash
    doc = upsert_file(session, path, original_name=file.filename, mime=file.content_type or "text/plain")

    return DocCreateResp(id=doc.id, filename=doc.filename, original_name=doc.original_name, size=doc.size, created_at=doc.created_at)


@router.get("/list", response_model=List[DocListItem])
async def list_docs(session: Session = Depends(get_session)):
    rows = session.exec(select(Document)).all()
    return [DocListItem(id=r.id, original_name=r.original_name, filename=r.filename, size=r.size, updated_at=r.updated_at) for r in rows]


@router.delete("/{doc_id}")
async def remove_doc(doc_id: int, session: Session = Depends(get_session)):
    ok = delete_doc(session, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到文档")
    # 重建索引，确保向量库同步（简化做法）
    rebuild_index(session)
    return {"deleted": doc_id}


@router.post("/reindex")
async def reindex(session: Session = Depends(get_session)):
    rebuild_index(session)
    return {"status": "rebuilt"}
```

------

## 21) app/routers/chat.py

```python
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlmodel import Session
from typing import Optional, List
from ..db.session import get_session
from ..services.rag import rag_answer, make_retrieval_chain
from ..agents.compliance_agent import build_agent
from ..tools.report_generator import save_report
from ..schemas import ChatResponse
from ..config import settings
from pathlib import Path
from ..services.ingest import upsert_file

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/ask", response_model=ChatResponse)
async def ask(
    session: Session = Depends(get_session),
    message: str = Form(...),
    use_rag: bool = Form(True),
    generate_report: bool = Form(False),
    top_k: int = Form(5),
    files: Optional[List[UploadFile]] = None,
):
    # 1) 接收附件 -> 入库 -> 即刻可被 RAG 检索
    context_preview_parts = []
    if files:
        for f in files:
            fname = f.filename
            data = await f.read()
            p = Path(settings.upload_dir) / fname
            p.write_bytes(data)
            doc = upsert_file(session, p, original_name=f.filename, mime=f.content_type or "text/plain")
            context_preview_parts.append(f"[新附件:{doc.original_name} 已入库]")

    # 2) 生成答案（RAG or 纯 LLM）
    if use_rag:
        agent = build_agent()
        # Agent 会优先使用 RAG-Query，不足再 Baidu-Search
        answer = agent.run(message)
        # 取少量上下文预览：调用检索链一次以拿到上下文格式
        chain = make_retrieval_chain(top_k)
        res = chain.invoke(message)
        ctx = res.response_metadata.get("input", {}).get("context") if hasattr(res, 'response_metadata') else None
        context_preview = "\n".join(context_preview_parts) + "\n" + (ctx if isinstance(ctx, str) else "")
    else:
        # 直出 LLM（无检索）
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key=settings.openai_api_key, model=settings.chat_model, temperature=0.2)
        answer = llm.invoke(message).content
        context_preview = "\n".join(context_preview_parts)

    # 3) 如需生成合规性报告
    report_id = None
    if generate_report:
        report_id = save_report(session, question=message, answer=answer, context_preview=context_preview)

    return ChatResponse(answer=answer, report_id=report_id)


@router.get("/report/{report_id}")
async def download_report(report_id: int, session: Session = Depends(get_session)):
    from ..db.models import Report
    rec = session.get(Report, report_id)
    if not rec:
        return {"error": "not found"}
    return {
        "id": rec.id,
        "name": rec.report_name,
        "md_path": rec.report_path_md,
        "pdf_path": rec.report_path_pdf,
    }
```

------

## 22) app/deps.py

```python
from .db.session import init_db
from .vectorstore.faiss_store import faiss_store

# 在应用启动时初始化 DB 与 FAISS

def on_startup():
    init_db()
    faiss_store.load()
```

------

## 23) app/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, documents, chat
from .deps import on_startup

app = FastAPI(title="AI Toolbox RAG Platform", version="0.1.0")

# CORS（按需放宽）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)

@app.on_event("startup")
async def _startup():
    on_startup()

# 便于本地调试：uvicorn app.main:app --reload
```

------

## 24) Dockerfile

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

------

## 25) docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}
      - CHAT_MODEL=${CHAT_MODEL}
      - DATA_DIR=/app/data
      - UPLOAD_DIR=/app/data/uploads
      - REPORT_DIR=/app/data/reports
      - INDEX_DIR=/app/data/index
      - BAIDU_API_KEY=${BAIDU_API_KEY}
      - BAIDU_CSE_ID=${BAIDU_CSE_ID}
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
```

------

## 26) README.md（关键用法）

~~~md
# AI Toolbox（FastAPI + LangChain + FAISS）

## 启动

```bash
cp .env.example .env  # 填写 OPENAI_API_KEY 等
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
~~~

或使用 Docker：

```bash
docker compose up --build
```

API 文档：`http://localhost:8000/docs`

## 典型流程

1. **上传文档（向量化入库）**

```bash
curl -F "file=@/path/to/policy.md" -F "mode=insert" http://localhost:8000/api/docs/upload
```

1. **对话 + 可选附件 + RAG**

```bash
curl -F "message=该设备是否满足GB/T xxx标准？" \
     -F "use_rag=true" \
     -F "generate_report=true" \
     -F "files=@/path/to/new_requirement.md" \
     http://localhost:8000/api/chat/ask
```

返回中 `report_id` 可用于下载：

```bash
curl http://localhost:8000/api/chat/report/1
```

1. **删除文档 & 重建索引**

```bash
curl -X DELETE http://localhost:8000/api/docs/1
curl -X POST   http://localhost:8000/api/docs/reindex
```

## 自定义工具

- `Baidu-Search`：外部检索补充。
- `RAG-Query`：内部知识库检索。
- 报告生成：`/api/chat/ask` 中 `generate_report=true` 触发，`.md` + `.pdf` 输出。

## 注意

- Demo 支持 `.txt/.md/.csv/.json/.pdf(占位)`；生产请扩展解析器与权限。
- FAISS 删除以**重建**实现，实际可用向量标记或分片索引。
- Agent 采用 `zero-shot-react`，可按需切换到更稳健的 `structured chat`。



# JWT认证

在 FastAPI 或类似 Python 项目中，**JWT（JSON Web Token）认证**常见实现步骤如下：

---

### 1. 安装依赖

```shell
pip install python-jose[cryptography] passlib[bcrypt]
```

---

### 2. 配置密钥和算法

```python
# config.py
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
```

---

### 3. 工具函数

```python
from jose import jwt
from datetime import datetime, timedelta

def create_access_token(data: dict, expires_delta: int = 60):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None
```

---

### 4. 登录接口（颁发 token）

```python
from fastapi import APIRouter, HTTPException, Depends, Form

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    # 这里应校验用户名密码（略）
    # 假设校验通过
    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}
```

---

### 5. 依赖项：校验 token 并获取当前用户

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return payload["sub"]
```

---

### 6. 受保护接口

```python
@router.get("/protected")
def protected_route(current_user: str = Depends(get_current_user)):
    return {"user": current_user, "msg": "You are authenticated!"}
```

---

## 总结

- 登录时返回 JWT token，前端保存。
- 访问受保护接口时，前端在 `Authorization: Bearer <token>` 头中带上 token。
- 后端用依赖项校验 token，获取当前用户信息。

---

如需完整示例或集成到你的项目结构，可继续提问！