# 实战项目

每个项目一个文件夹，按编号命名 `0X-project-name/`。

## 项目列表

| # | 项目 | 说明 | 涉及知识点 |
|---|------|------|-----------|
| 01 | [mini-agent](01-mini-agent/) | ✅ 最小 ReAct Agent，零框架 | Agent 循环、工具注册、错误处理 |
| 02 | rag-agent | 文档 RAG 问答（NVIDIA NIM） | RAG、向量检索、对话记忆、流式输出 |
| 03 | agentic-rag | 把 RAG 升级为 Agent | Agent 决策、多工具调度、自我纠错 |

## 新增项目规范

```
projects/0X-project-name/
├── README.md        ← 项目说明、如何运行、架构图
├── src/             ← 源代码
├── tests/           ← 测试
└── requirements.txt ← 依赖
```
