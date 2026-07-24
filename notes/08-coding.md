# 动手 Coding

## 1. 实现简单的 ReAct Agent

```python
import json
from typing import Any

class SimpleReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm = llm          # callable: (messages) -> response_text
        self.tools = tools      # {name: callable}
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task}
        ]

        for step in range(self.max_steps):
            response = self.llm(messages)
            action = self._parse_action(response)

            if action["type"] == "finish":
                return action["answer"]

            if action["type"] == "tool_call":
                tool_result = self._execute_tool(
                    action["tool_name"],
                    action["tool_args"]
                )
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"工具结果: {tool_result}\n请继续。"
                })

        return "达到最大步数，任务未完成"

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )
        return f"""你是一个能使用工具的 Agent。

可用工具：
{tool_descriptions}

回复格式：
- 调用工具：{{"type": "tool_call", "tool_name": "...", "tool_args": {{...}}}}
- 完成任务：{{"type": "finish", "answer": "..."}}"""

    def _parse_action(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"type": "finish", "answer": response}

    def _execute_tool(self, name: str, args: dict) -> Any:
        if name not in self.tools:
            return f"错误：工具 '{name}' 不存在"
        try:
            return self.tools[name](**args)
        except Exception as e:
            return f"工具调用异常: {e}"


# 使用示例
def search(query: str) -> str:
    """搜索互联网，参数: query - 搜索关键词"""
    # 模拟搜索结果
    return f"关于 '{query}' 的搜索结果: ..."

agent = SimpleReActAgent(
    llm=my_llm_function,
    tools={"search": search},
    max_steps=5
)
result = agent.run("今天的天气怎么样？")
```

## 2. 实现带 Memory 的对话 Agent

```python
import hashlib
from typing import Optional
import numpy as np

class MemoryAgent:
    def __init__(self, llm, embedding_model, vector_store):
        self.llm = llm
        self.embed = embedding_model
        self.store = vector_store
        self.conversation_history = []  # 短期记忆

    def chat(self, user_input: str, user_id: str) -> str:
        # 1. 检索相关长期记忆
        relevant_memories = self._retrieve_memories(user_input, user_id)

        # 2. 构建上下文
        context = self._build_context(user_input, relevant_memories)

        # 3. 生成回答
        response = self.llm(context)

        # 4. 更新短期记忆
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        self._trim_history(max_turns=20)

        # 5. 提取并保存长期记忆
        self._extract_and_save_memory(user_input, response, user_id)

        return response

    def _retrieve_memories(self, query: str, user_id: str, top_k: int = 5):
        query_vec = self.embed(query)
        return self.store.search(
            query_vec,
            filter={"user_id": user_id},
            top_k=top_k
        )

    def _build_context(self, user_input: str, memories: list) -> list:
        system_msg = "你是一个有记忆的助手。"

        if memories:
            memory_text = "\n".join(f"- {m['content']}" for m in memories)
            system_msg += f"\n\n相关历史记忆：\n{memory_text}"

        return [
            {"role": "system", "content": system_msg},
            *self.conversation_history[-20:],  # 最近 20 轮
            {"role": "user", "content": user_input}
        ]

    def _trim_history(self, max_turns: int):
        if len(self.conversation_history) > max_turns * 2:
            self.conversation_history = self.conversation_history[-(max_turns * 2):]

    def _extract_and_save_memory(self, user_input: str, response: str, user_id: str):
        # 让 LLM 提取值得长期记住的信息
        extraction_prompt = f"""从以下对话中提取值得长期记住的信息，每条一行。
如果不值得记住，回复 "NONE"。

用户: {user_input}
助手: {response}

值得记住的信息："""

        extracted = self.llm([{"role": "user", "content": extraction_prompt}])

        if extracted.strip() != "NONE":
            for line in extracted.strip().split("\n"):
                if line.strip():
                    vec = self.embed(line)
                    self.store.insert(
                        vector=vec,
                        metadata={
                            "content": line,
                            "user_id": user_id,
                            "timestamp": time.time()
                        }
                    )
```

## 3. 实现工具注册与调度

```python
from functools import wraps
import inspect
from typing import get_type_hints, Any

class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, func=None, *, name: str = None, description: str = None):
        """装饰器：注册一个工具"""
        if func is None:
            return lambda f: self.register(f, name=name, description=description)

        tool_name = name or func.__name__
        hints = get_type_hints(func)

        # 解析参数 schema
        params = {}
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, str)
            params[param_name] = {
                "type": self._python_type_to_json(param_type),
                "description": self._get_param_description(func, param_name)
            }

        self._tools[tool_name] = {
            "function": func,
            "schema": {
                "name": tool_name,
                "description": description or func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": [
                        name for name, p in sig.parameters.items()
                        if p.default is inspect.Parameter.empty
                        and name not in ("self", "cls")
                    ]
                }
            }
        }
        return func

    def get_schemas(self) -> list[dict]:
        return [t["schema"] for t in self._tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        return self._tools[name]["function"](**args)

    def _python_type_to_json(self, py_type) -> str:
        mapping = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        return mapping.get(py_type, "string")

    def _get_param_description(self, func, param_name: str) -> str:
        # 可以增强：从 docstring 解析参数说明
        return f"{param_name} 参数"


# 使用示例
registry = ToolRegistry()

@registry.register(description="搜索用户数据库")
def search_users(query: str, limit: int = 10) -> list:
    """根据姓名或邮箱搜索用户"""
    # 实现...
    return []

@registry.register(description="发送邮件")
def send_email(to: str, subject: str, body: str) -> bool:
    """发送一封邮件"""
    # 实现...
    return True

# 获取所有工具 schema 传给 LLM
schemas = registry.get_schemas()

# 执行 LLM 要调用的工具
result = registry.execute("search_users", {"query": "张三", "limit": 5})
```

## 面试可能考的变体

1. 给上面的 Agent 加上 **并行工具调用** 支持
2. 实现一个 **带超时和重试** 的工具执行器
3. 给 Memory Agent 加上 **记忆衰减** 机制
4. 实现一个简单的 **多 Agent 对话** 调度器
