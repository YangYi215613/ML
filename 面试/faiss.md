FAISS（Facebook AI Similarity Search）中的数据**本质上是高维向量**，用于高效的相似度检索。  
在你的项目中，FAISS 主要存储的是**文本、文档等内容的向量化结果**，每个向量通常会关联一些元数据（如文档ID、文件名等）。

---

## 1. **FAISS 存储的数据结构**

- **向量本身**：通常是 N 维的浮点数组（如 384维、768维等）。
- **元数据**：如 doc_id、filename、chunk_id 等，便于检索结果和原始数据关联。
- **索引结构**：FAISS 内部会构建倒排索引、IVF、HNSW 等结构，加速相似度检索。

---

## 2. **存储流程（以 LangChain 为例）**

- 先用 embedding 模型将文本转为向量。
- 用 `FAISS.add_texts(texts, metadatas=...)` 存入，FAISS 会自动为每个向量分配一个内部索引（int型）。
- 元数据（如 doc_id、filename）会和向量一起存储，便于后续检索时返回。

---

## 3. **是否需要自己建索引？**

- **不需要自己建主键索引**，FAISS 会自动为每个向量分配内部索引（如 0, 1, 2, ...）。
- 你只需在 `metadatas` 里存上自己的业务ID（如 doc_id），检索时就能反查原始文档。
- 如果你需要删除或更新某条数据，建议在元数据里保存唯一标识（如 doc_id + chunk_id）。

---

## 4. **示例代码**

````python
# texts: List[str]
# metadatas: List[dict]，如 [{"doc_id": 123, "filename": "xxx.txt"}, ...]
faiss_index = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
# 检索时
docs = faiss_index.similarity_search(query, k=5)
for doc in docs:
    print(doc.metadata["doc_id"], doc.page_content)
````

---

**总结：**
- FAISS 存的是向量和元数据，内部有自动索引。
- 你只需在元数据里存业务ID，方便检索结果和原始数据关联。
- 不需要自己建主键索引，FAISS 自动管理。

如需支持删除/更新，建议元数据里有唯一标识。