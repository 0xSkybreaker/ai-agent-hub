# 01-mini-agent — 最小 ReAct Agent

**零框架依赖，从零实现一个能自主决策、使用工具的 AI Agent。**

用最少的代码（~350 行核心逻辑）展示 Agent 的本质：不是魔法，就是一个循环。

## 为什么做这个项目

面试官让你"手写一个 Agent"，你写出来的就是它。

这个项目回答三个核心问题：
1. **Agent 和普通 LLM 调用的区别是什么？** — Agent 会在循环中自主决定"下一步做什么"
2. **工具调用到底怎么实现的？** — 不是魔法，是 OpenAI function calling + 结果回传
3. **没有 LangChain 能写 Agent 吗？** — 能，而且更清晰

## 架构

```
┌─────────────────────────────────────────────────┐
│                  ReActAgent                      │
│                                                 │
│   ┌──────────┐     ┌──────────┐     ┌───────┐  │
│   │   LLM    │ ──→ │  Parse   │ ──→ │ Tool  │  │
│   │  (思考)   │ ←── │  (解析)   │ ←── │ (执行) │  │
│   └──────────┘     └──────────┘     └───────┘  │
│        │                                    │    │
│        └──── 循环，直到任务完成 ────────────┘    │
│                                                 │
│   每轮： Thought → Action → Observation → ...   │
└─────────────────────────────────────────────────┘
```

### 文件结构

```
src/mini_agent/
├── __main__.py    ← 入口：python -m mini_agent
├── config.py      ← 配置（pydantic-settings）
├── llm.py         ← LLM 客户端（原生 openai SDK）
├── tools.py       ← 工具注册表 + 5 个内置工具
├── agent.py       ← ReActAgent 核心循环 ★
└── cli.py         ← CLI 界面（typer + rich）
```

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_BASE_URL 和 LLM_API_KEY
```

支持任何 OpenAI 兼容的 API：
- **NVIDIA NIM**: `LLM_BASE_URL=https://integrate.api.nvidia.com/v1`
- **OpenAI**: `LLM_BASE_URL=https://api.openai.com/v1`
- **本地 vLLM/Ollama**: `LLM_BASE_URL=http://localhost:8000/v1`

### 3. 运行

```bash
# 单次任务
python -m mini_agent run "算一下 (30 + 45) * 6 / 2"

# 列出所有工具
python -m mini_agent tools

# 交互式对话
python -m mini_agent chat
```

## 内置工具

| 工具 | 用途 | 示例 |
|------|------|------|
| `calculator` | 安全执行数学表达式 | `calculator("(3+5)*2")` |
| `get_current_time` | 获取当前 UTC 时间 | `get_current_time()` |
| `read_file` | 读取文本文件内容 | `read_file("src/main.py")` |
| `list_directory` | 列出目录内容 | `list_directory(".")` |
| `text_stats` | 统计字/词/行数 | `text_stats("hello world")` |

## 核心代码解读

### Agent 循环（agent.py 核心）

```python
def run(self, task: str) -> AgentResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for i in range(max_steps):
        # 1. 调用 LLM，让它决定下一步
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools.get_schemas(),  # ← 这是关键
            tool_choice="auto",               # ← LLM 自主决定要不要调工具
        )

        choice = response.choices[0]

        # 2. 如果 LLM 返回了文本 → 任务完成
        if not choice.message.tool_calls:
            return choice.message.content  # 最终答案

        # 3. 如果 LLM 要调工具 → 执行 + 结果回传
        tool_call = choice.message.tool_calls[0]
        result = self.tools.execute(
            tool_call.function.name,
            tool_call.function.arguments,
        )

        # 4. 把工具结果追加到消息历史 → 下一轮循环
        messages.append(assistant_tool_call_message)
        messages.append(tool_result_message)

    # 到达最大步数 → 停止
```

### 工具注册（tools.py 核心）

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict       # JSON Schema
    func: Callable         # 实际执行的 Python 函数

class ToolRegistry:
    def register(self, tool: Tool): ...
    def get_schemas(self) -> list[dict]: ...  # → 喂给 LLM
    def execute(self, name, args) -> str: ...  # → 执行工具
```

## 设计决策

| 决策 | 为什么 |
|------|--------|
| 使用原生 function calling | 比文本解析更可靠，LLM 直接返回结构化 JSON |
| 不用 LangChain | 理解底层原理是面试的核心要求 |
| 工具定义用 JSON Schema | 和 OpenAI API 原生兼容，不需要自定义格式 |
| 工具结果是纯文本 | 最简单的"回传"方式，LLM 能直接理解 |
| 单轮工具调用 | 每次只调一个工具，减少复杂度（可扩展为并行） |

## 扩展方向

这是你的最小可行 Agent。在你理解了它的原理之后，可以加：

- **并行工具调用** — 一次调用多个独立工具
- **自我纠错** — 工具报错后自动重试或换策略
- **记忆系统** — 跨对话保留重要信息（参考 02-rag-agent）
- **流式输出** — 用户能看到 Agent 的思考过程
- **工具检索** — 工具超过 20 个时自动匹配最相关的

这些扩展本质上都是在这个循环上加东西——循环才是 Agent 的灵魂。

## License

MIT
