# RAG Agent

基于 NVIDIA NIM API 的通用文档问答 RAG (Retrieval-Augmented Generation) Agent。

## 功能

- 📄 **多格式文档摄入**: PDF, Word (DOCX), Markdown, TXT, HTML/网页
- 🔍 **语义检索**: 基于 NVIDIA embedding 模型的向量相似度搜索
- 🤖 **智能问答**: 基于 Llama 3.1 Nemotron Ultra 的上下文感知生成
- 💬 **对话记忆**: 支持多轮对话，滑动窗口上下文管理
- 🖥️ **双界面**: CLI 命令行 + FastAPI REST API (SSE 流式输出)
- 📊 **增量索引**: 基于内容哈希的文档变更检测，避免重复摄入

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 NVIDIA_API_KEY
```

### 3. 摄入文档

```bash
# 摄入单个文件
python -m rag_agent index data/sample.pdf

# 摄入整个目录（自动识别所有支持的文件）
python -m rag_agent index ./documents/

# 强制重建索引（忽略缓存，全部重新处理）
python -m rag_agent index ./documents/ --force
```

### 4. 提问

```bash
# 单次提问
python -m rag_agent query "文档的主要内容是什么？"

# 交互式对话
python -m rag_agent chat
```

### 5. 启动 API 服务

```bash
python -m rag_agent serve
# API 文档: http://127.0.0.1:8000/docs
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/ingest` | 上传并摄入文档 |
| GET | `/api/v1/documents` | 列出已索引文档 |
| DELETE | `/api/v1/documents/{path}` | 删除文档索引 |
| POST | `/api/v1/query` | 问答（非流式） |
| POST | `/api/v1/query/stream` | 问答（SSE 流式） |
| GET | `/api/v1/health` | 健康检查 |

## 项目结构

```
src/rag_agent/
├── config.py          # 配置管理
├── cli.py             # CLI 入口
├── ingestion/         # 文档加载与分块
├── embeddings/        # NVIDIA embedding
├── vector_store/      # ChromaDB 向量存储
├── llm/               # NVIDIA LLM 客户端
├── retrieval/         # 语义检索
├── generation/        # 生成与引用
├── memory/            # 对话记忆
├── api/               # FastAPI 服务
└── utils/             # 工具函数
```

## License

MIT
