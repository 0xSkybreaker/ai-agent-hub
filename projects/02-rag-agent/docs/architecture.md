# RAG Agent 架构文档

> 适合给新加入的开发者做 Onboarding，或给技术团队做架构评审。

---

## 1. 一句话概述

这是一个**文档智能问答系统**——你把 PDF / Word / Markdown 等文档丢进去，然后用自然语言提问，系统从文档里找到答案并告诉你答案出自哪里。

---

## 2. 整体架构（四层模型）

```
┌─────────────────────────────────────────────────┐
│                  用户界面层                       │
│         CLI (Typer+Rich)  │  FastAPI REST       │
│         rag-agent query   │  POST /api/v1/query │
│         rag-agent chat    │  SSE 流式           │
└──────────────┬──────────────┬───────────────────┘
               │              │
┌──────────────▼──────────────▼───────────────────┐
│                 应用编排层                        │
│   Generator (检索→提示词→生成→引用)              │
│   ConversationMemory (多轮对话)                  │
│   IndexingPipeline (文档→分块→嵌入→存储)         │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│ 检索层 │ │生成层  │ │ 摄入层 │
│       │ │       │ │       │
│Retriever│LLM    │ │Loader │
│+       │Client  │ │+      │
│ChromaDB│+       │ │Splitter│
│       │Prompts │ │       │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
┌───▼─────────▼─────────▼─────────────────────────┐
│                 基础设施层                        │
│   Config (pydantic-settings)                    │
│   NVIDIA NIM API (Embedding + Chat)             │
│   Utils (loguru, tenacity, hashing)             │
└─────────────────────────────────────────────────┘
```

**一句话总结每层**：
- **界面层**：CLI 给人用，API 给程序调用
- **编排层**：把下面的零件组装成完整的问答流程
- **能力层**：每个模块只做一件事——检索、生成、摄入各自独立
- **基础层**：配置、外部 API、工具函数，被上面所有层共享

---

## 3. 核心数据流（两条主线）

整个系统本质上就两条数据流：**写入（摄入文档）**和**读取（回答问题）**。

### 3.1 写入流：文档如何变成可检索的知识

```
用户放一个 PDF
    │
    ▼
┌──────────────────┐
│ 1. Loader        │  根据文件后缀 (.pdf / .docx / .md ...)
│   自动选加载器    │  自动分发到对应的 Loader
│                  │  PDF → pdfplumber 逐页提取
│                  │  Word → python-docx 提取段落+表格
└──────┬───────────┘
       │  Document[] (每个 Document = 一页或一个逻辑单元)
       ▼
┌──────────────────┐
│ 2. Splitter      │  RecursiveCharacterTextSplitter
│   递归分块        │  优先按段落(\n\n)切，再按换行(\n)、句子(. )
│                  │  每块 1000 字符，块间重叠 200 字符
│                  │  保留元数据：来源、页码、分块序号
└──────┬───────────┘
       │  Chunk[] (每个 Chunk = 文本 + metadata)
       ▼
┌──────────────────┐
│ 3. Embedding     │  调 NVIDIA nv-embedqa-e5-v5
│   向量化          │  文本 → 1024 维浮点数向量
│                  │  ChromaDB 自动批量调用，无需手工分批
└──────┬───────────┘
       │  embeddings + texts + metadatas
       ▼
┌──────────────────┐
│ 4. ChromaDB      │  持久化到 data/chroma/
│   向量存储        │  余弦距离索引
│                  │  支持 metadata 过滤（按来源删除等）
└──────────────────┘
```

**增量索引**：索引前先算文件 SHA-256，和 ChromaDB 里存的哈希对比。一样就跳过（"unchanged"），不一样就删旧块重新索引。

### 3.2 读取流：一个问题如何找到答案

```
用户问 "RAG 有什么好处？"
    │
    ▼
┌──────────────────┐
│ 1. Embed Query   │  把问题也向量化
│   问题向量化      │  用同一个 embedding 模型
└──────┬───────────┘
       │ query_embedding
       ▼
┌──────────────────┐
│ 2. 语义检索       │  ChromaDB.similarity_search()
│   Top-K 最近邻    │  余弦距离，取 Top-5
│                  │  过滤低于 threshold 的结果
└──────┬───────────┘
       │ RetrievedDocument[] (文本 + 元数据 + 相关度得分)
       ▼
┌──────────────────┐
│ 3. 构建提示词     │  把检索结果格式化为上下文：
│   Prompt 拼装    │  [Document 1 — Source: report.pdf, Page 3]
│                  │  ...chunk content...
│                  │  加上对话历史 + 系统指令 + 用户问题
└──────┬───────────┘
       │ formatted prompt
       ▼
┌──────────────────┐
│ 4. LLM 生成      │  调 NVIDIA nemotron-super-49b
│   生成答案        │  非流式：一次返回完整答案
│                  │  流式：逐 token 通过 SSE 推送
└──────┬───────────┘
       │ answer text
       ▼
┌──────────────────┐
│ 5. 提取引用       │  从检索元数据提取唯一来源
│   来源标注        │  格式化为可读的引用列表
│                  │  答案存入对话记忆（下轮继续用）
└──────────────────┘
```

### 3.3 为什么这样设计？

```
写路径和读路径只共享两个东西：
  1. Embedding 模型（保证问题和文档在同一向量空间）
  2. ChromaDB（写入的终点 = 读取的起点）

除此之外完全解耦 ——
  换 LLM 不影响索引，
  换向量数据库不影响生成，
  加新文件格式不影响问答。
```

---

## 4. 代码目录结构（模块职责）

```
src/rag_agent/
│
├── config.py              # 所有配置的单一入口 (pydantic-settings)
│                           # 读 .env → Settings 对象 → 全局共享
│
├── ingestion/              # 职责：把外部文档变成内部 Document 对象
│   ├── base.py             #   Document dataclass + LoaderRegistry (策略模式)
│   ├── text_loader.py      #   .txt / .md 加载器
│   ├── pdf_loader.py       #   .pdf (pdfplumber 主, pypdf 回退)
│   ├── docx_loader.py      #   .docx (段落 + 表格提取)
│   └── web_loader.py       #   .html / URL (BeautifulSoup 清洗)
│
├── chunking/               # 职责：把长文档切成适合检索的小块
│   ├── splitter.py         #   封装 RecursiveCharacterTextSplitter
│   └── metadata.py         #   给每个块打标签 (hash, 序号, 时间戳)
│
├── embeddings/             # 职责：文本 → 向量（封装外部 API）
│   └── nvidia_embeddings.py
│
├── vector_store/           # 职责：向量的增删查
│   ├── chroma_store.py     #   ChromaDB CRUD + 余弦检索
│   └── indexer.py          #   写入流水线编排 (load → split → store)
│
├── llm/                    # 职责：调用大模型生成答案
│   ├── nvidia_client.py    #   ChatNVIDIA 封装 (generate + stream)
│   └── streaming.py        #   流式输出收集工具
│
├── retrieval/              # 职责：根据问题找相关文档块
│   └── retriever.py        #   搜索 + 阈值过滤 + 上下文格式化
│
├── generation/             # 职责：把检索结果 + 提示词 → 最终答案
│   ├── generator.py        #   RAG 主流程编排 (retrieve → prompt → generate → cite)
│   ├── prompts.py          #   提示词模板
│   └── citations.py        #   引用提取 + 格式化 (终端/API 两套输出)
│
├── memory/                 # 职责：记住对话上下文
│   └── conversation.py     #   滑动窗口 buffer (按 session_id 隔离)
│
├── api/                    # 职责：对外提供 HTTP 接口
│   ├── server.py           #   FastAPI app 工厂 + lifespan
│   ├── routes.py           #   REST 端点定义
│   ├── models.py           #   Pydantic 请求/响应 schema
│   └── dependencies.py     #   依赖注入（懒加载单例）
│
├── cli.py                  # 职责：命令行入口 (Typer + Rich 美化)
├── __main__.py             # python -m rag_agent 入口
│
└── utils/                  # 职责：横切关注点
    ├── logger.py           #   loguru 配置 (控制台 + 文件)
    ├── retry.py            #   tenacity 指数退避重试装饰器
    └── hashing.py          #   SHA-256 文件/文本哈希
```

**目录设计原则**：

| 原则 | 体现 |
|------|------|
| **单一职责** | 每个包只做一件事。`ingestion/` 不管向量怎么存，`vector_store/` 不管文档怎么加载 |
| **依赖方向** | 上层依赖下层。`generator.py` 依赖 `retriever` + `llm`，但反过来不成立 |
| **接口隔离** | 每个模块通过构造函数注入依赖，不直接 import 具体实现 |
| **可替换性** | 换 embedding 只需改 `embeddings/`，换向量数据库只需改 `vector_store/` |

---

## 5. 关键设计决策

### 5.1 为什么用 LangChain 而不是直接调 OpenAI SDK？

```
直接调 openai 库当然可以，两行代码的事。

但 LangChain 给了我们三个东西：
  1. ChatNVIDIA / NVIDIAEmbeddings 已经封装了认证、模型名、base_url
  2. RecursiveCharacterTextSplitter 是一个成熟的分块算法
  3. Chroma wrapper 自动处理 embedding 调用，不需要手动分批

我们只在"正好能用上 LangChain 的地方"用它，
不是把整个系统绑在 LangChain 的抽象上。
核心业务逻辑（indexer, generator, retriever）都是我们自己的代码。
```

### 5.2 为什么用 ChromaDB 而不是 FAISS / Pinecone？

```
ChromaDB = 本地文件数据库，零运维。
FAISS = 纯内存，进程重启数据就没了（需要手动序列化）。
Pinecone = 云服务，需要额外费用和网络。

对于本地 RAG 场景，ChromaDB 是最简单的选择。
未来如果需要扩展到生产环境，ChromaDB → Weaviate/Pinecone 只需改 vector_store/ 一个包。
```

### 5.3 为什么用 pydantic-settings 管理配置？

```
.env 文件 → Settings 对象 → 全局 settings 单例

好处：
  - 类型安全：settings.chunk_size 一定是 int，IDE 有自动补全
  - 集中管理：所有配置项在一个类里，改一个地方全局生效
  - 环境隔离：开发/测试/生产用不同的 .env 文件
```

### 5.4 Prompt 模板为什么是英文？

```
当前 NVIDIA 的模型是 Llama 系列，对英文指令的遵循度最好。
中文提问完全没问题——模型本身支持多语言。
系统指令用英文是为了确保"只基于上下文回答""不要编造信息"
这类约束被模型严格遵循。
```

### 5.5 对话记忆为什么用滑动窗口而不是 LLM 摘要？

```
滑动窗口：
  - 零额外 API 调用
  - 实现简单，行为可预测
  - 默认保留最近 10 轮对话 (20 条消息)

LLM 摘要：
  - 每轮需要额外调用一次模型
  - 摘要可能丢失关键细节
  - 适合长对话 (>20 轮) 的场景

当前默认窗口足够覆盖大多数文档问答场景。
后续可以做成可配置的策略：window / summary / hybrid。
```

---

## 6. 外部依赖全景图

```
                        ┌──────────────────┐
                        │   NVIDIA NIM API  │
                        │ integrate.api.    │
                        │ nvidia.com/v1     │
                        └──┬────────────┬──┘
                           │            │
              ┌────────────▼──┐  ┌─────▼──────────┐
              │ Chat Model    │  │ Embedding Model │
              │ nemotron-     │  │ nv-embedqa-    │
              │ super-49b     │  │ e5-v5          │
              └───────┬───────┘  └─────┬───────────┘
                      │                │
┌─────────────────────┼────────────────┼──────────────────┐
│                RAG Agent                                │
│                     │                │                   │
│  ┌──────────────────▼────┐  ┌───────▼───────────────┐  │
│  │ langchain-nvidia-     │  │ langchain-nvidia-      │  │
│  │ ai-endpoints          │  │ ai-endpoints           │  │
│  │ (ChatNVIDIA)          │  │ (NVIDIAEmbeddings)     │  │
│  └───────────────────────┘  └─────────────────────────┘  │
│                                                          │
│  ┌───────────────────────┐  ┌─────────────────────────┐  │
│  │ ChromaDB (本地文件)    │  │ 文档加载器               │  │
│  │ data/chroma/           │  │ pdfplumber / pypdf      │  │
│  │ 余弦距离索引            │  │ python-docx / BS4       │  │
│  └───────────────────────┘  └─────────────────────────┘  │
│                                                          │
│  ┌───────────────────────┐  ┌─────────────────────────┐  │
│  │ FastAPI + uvicorn      │  │ Typer + Rich            │  │
│  │ REST + SSE 流式        │  │ CLI + 美化输出           │  │
│  └───────────────────────┘  └─────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 基础设施: pydantic-settings / loguru / tenacity    │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 7. 扩展指南

### 7.1 换 LLM 提供商

只需改两个地方：

```python
# 1. .env 改模型名
CHAT_MODEL=openai/gpt-oss-120b   # NVIDIA 上可用的任何模型

# 2. 或者完全换提供商：改 llm/nvidia_client.py，
#    把 ChatNVIDIA 换成 ChatOpenAI / ChatAnthropic
```

### 7.2 添加新文件格式

在 `ingestion/` 下加一个新 loader：

```python
# ingestion/csv_loader.py
class CSVLoader(BaseLoader):
    @property
    def supported_extensions(self):
        return [".csv"]

    def load(self, source: str) -> list[Document]:
        # 读 CSV → Document
        ...

# 然后在 indexer.py 的 _create_loader_registry() 里注册
registry.register(CSVLoader)
```

### 7.3 换向量数据库

实现 `chroma_store.py` 的接口契约，替换实现：

```python
# 当前 ChromaVectorStore 的方法签名就是接口：
class ChromaVectorStore:
    def add_documents(self, texts, metadatas) -> list[str]: ...
    def search(self, query, top_k, filter) -> list[tuple]: ...
    def delete_by_source(self, source_path) -> int: ...
    def get_source_hashes(self) -> dict: ...
    def get_collection_stats(self) -> dict: ...
    def clear_collection(self) -> None: ...

# 换 Milvus / Weaviate 只需要实现同样的方法
```

### 7.4 增加重排序 (Re-ranking)

在 `generator.py` 的 `generate()` 里加一步：

```python
# 检索 top-50 → Cohere Rerank → 取 top-5 → 送给 LLM
documents = self._retriever.retrieve(query=question, top_k=50)
reranked = self._reranker.rerank(query, documents)[:5]
```

---

## 8. 一句话总结每个文件是干什么的

| 文件 | 一句话 |
|------|--------|
| `config.py` | 所有配置从这里读 |
| `ingestion/base.py` | 定义 Document 是什么，以及加载器的注册表 |
| `ingestion/pdf_loader.py` | 读 PDF（先试 pdfplumber，不行再用 pypdf） |
| `ingestion/text_loader.py` | 读 .txt 和 .md |
| `chunking/splitter.py` | 把长文档切成小块（1000 字/块，重叠 200 字） |
| `embeddings/nvidia_embeddings.py` | 文本转向量 |
| `vector_store/chroma_store.py` | ChromaDB 的增删查 |
| `vector_store/indexer.py` | 索引流水线（加载→分块→存储一条龙） |
| `llm/nvidia_client.py` | 调 NVIDIA 大模型生成文本 |
| `retrieval/retriever.py` | 根据问题找相关文档块 |
| `generation/generator.py` | RAG 主流程：检索→构建提示词→生成→提取引用 |
| `generation/prompts.py` | 提示词模板（"只基于上下文回答，别瞎编"） |
| `generation/citations.py` | 把检索来源格式化为可读的引用 |
| `memory/conversation.py` | 记住对话上下文（按 session 隔离） |
| `api/server.py` | FastAPI 启动入口 |
| `api/routes.py` | 定义所有 HTTP 端点 |
| `cli.py` | 命令行入口（Typer + Rich 美化） |
| `utils/logger.py` | 日志配置（loguru） |
| `utils/retry.py` | API 调用失败自动重试（指数退避） |
| `utils/hashing.py` | 计算文件 SHA-256（用于增量索引） |
