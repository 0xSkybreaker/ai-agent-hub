# 工具调用 (Tool Use / Function Calling)

## 工具定义

### 最佳实践

```json
{
  "name": "search_database",
  "description": "查询用户数据库。用于查找用户信息、订单记录等。不要在单次调用中传入过多条件。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "SQL SELECT 查询语句，只允许 SELECT，禁止 INSERT/UPDATE/DELETE"
      },
      "limit": {
        "type": "integer",
        "description": "返回结果的最大条数，默认 10，最大 100",
        "default": 10
      }
    },
    "required": ["query"]
  }
}
```

### 关键原则

- **描述要具体**：什么时候用、什么时候不用都要写清楚
- **参数约束明确**：类型、范围、默认值、required
- **命名清晰**：动词_名词，如 `search_docs`、`create_ticket`
- **避免歧义**：不要在 description 里写"和其他操作"

## 结构化输出

### JSON Mode vs JSON Schema

- **JSON Mode**：保证输出是合法 JSON，但不保证字段结构
- **Structured Output (JSON Schema)**：严格按 schema 输出，字段名和类型都保证

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "weather_response",
    "schema": {
      "type": "object",
      "properties": {
        "temperature": { "type": "number" },
        "condition": { "type": "string", "enum": ["sunny", "cloudy", "rainy"] },
        "humidity": { "type": "number", "minimum": 0, "maximum": 100 }
      },
      "required": ["temperature", "condition"]
    }
  }
}
```

## 工具调度策略

### 工具选择

- **全量传入**：工具少（<20）时直接全部传入 context
- **检索式**：用 embedding 匹配最相关的 N 个工具
- **分层式**：先选工具类别，再在类别内选具体工具
- **动态注册**：根据当前上下文动态决定可用工具集

### 工具执行

- **串行**：一步步调，适合依赖关系强的场景
- **并行**：独立工具调用并发执行，减少延迟
- **条件分支**：根据上一步结果决定下一步调什么

## 错误处理

```
调用失败模式：
1. 参数错误 → 返回具体错误信息 + 格式要求，让 LLM 重试
2. 超时 → 设置合理 timeout，超时后返回部分结果或重试
3. 权限不足 → 返回权限问题说明，触发升级或降级
4. 空结果 → 明确告知无结果，避免 LLM 编造数据
5. 网络/服务异常 → 重试（指数退避）+ 最终降级
```

### 重试策略

- 最多重试 2-3 次
- 每次重试时把上次错误信息带入 context
- 超过重试次数后走降级路径

## 面试常见追问

1. 一个工具的结果太大（比如 10 万条数据）怎么处理？
2. 怎么防止 LLM 调用不该调的工具？
3. 工具调用和 Structured Output 怎么结合？
4. 如何设计一个工具来让 Agent 能自我纠错？
