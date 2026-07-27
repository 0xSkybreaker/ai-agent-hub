# AI Agent Hub

AI Agent 学习笔记 + 实战项目仓库 — 从面试备战到动手实践。

```
ai-agent-hub/
├── notes/                     ← 📝 学习笔记（8 大面试维度）
├── projects/                  ← 🛠️ 实战项目（逐步增加）
└── README.md
```

## 快速导航

### 📝 学习笔记 → [notes/](notes/)

| # | 主题 | 一句话 |
|---|------|--------|
| 0 | [备战攻略](notes/00-备战攻略.md) | 8 周通关路线图 + 面试技巧 |
| 1 | [Agent 架构基础](notes/01-architecture.md) | ReAct / Plan-Execute / 多 Agent 协作 |
| 2 | [工具调用](notes/02-tool-use.md) | Function Calling / 结构化输出 / 错误处理 |
| 3 | [记忆与上下文](notes/03-memory.md) | 短期记忆 / 长期记忆 / RAG |
| 4 | [规划与推理](notes/04-planning.md) | CoT / Tree-of-Thoughts / Self-Reflection |
| 5 | [评估与可靠性](notes/05-evaluation.md) | Hallucination 控制 / 安全 / 基准测试 |
| 6 | [工程实践](notes/06-engineering.md) | 框架对比 / 可观测性 / 流式 / HITL |
| 7 | [系统设计](notes/07-system-design.md) | 场景设计题 / 架构决策 |
| 8 | [动手 Coding](notes/08-coding.md) | ReAct 实现 / Memory 实现 / 工具调度 |

### 🛠️ 实战项目 → [projects/](projects/)

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| 01 | [mini-agent](projects/01-mini-agent/) | ✅ 完成 | 最小 ReAct Agent，零框架依赖 |
| 02 | [rag-agent](projects/02-rag-agent/) | ✅ 已有 | 基于 NVIDIA NIM 的 RAG 文档问答 |
| 03 | [agentic-rag](projects/03-agentic-rag/) | ✅ 已有 | ReAct Agent 自主驱动 RAG 检索与合成 |
| … | 更多 | 🔜 | 持续更新 |

## 学习路线

```
第 1-2 周：通读笔记 → 画出 Agent 架构全景图
第 3-4 周：做 01-mini-agent → 理解 ReAct 循环的本质
第 5-6 周：做 02/03 → 理解 RAG，再将 Agent 循环叠加到 RAG 上
第 7-8 周：整理面试故事 → 刷高频题 → 白板练习
```

详细计划见 → [备战攻略](notes/00-备战攻略.md)

## 技术栈

- **LLM**: NVIDIA NIM (Llama 3.1 Nemotron) / OpenAI / Anthropic
- **向量存储**: ChromaDB
- **后端**: Python / FastAPI / Typer
- **嵌入**: NVIDIA Embedding API
