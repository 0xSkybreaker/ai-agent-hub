# 工程实践

## 框架对比

| 框架 | 特点 | 适合场景 | 学习曲线 |
|------|------|----------|----------|
| **LangChain** | 生态最全，组件丰富 | 快速原型、复杂链 | 中高 |
| **LangGraph** | 图状态机、可控流程 | 复杂 Agent 流程 | 高 |
| **AutoGen** | 多 Agent 对话 | 多 Agent 协作 | 中 |
| **CrewAI** | 角色扮演、简单易用 | 多 Agent 工作流 | 低 |
| **Semantic Kernel** | 微软生态、.NET 友好 | 企业级应用 | 中 |
| **原生 API** | 无框架依赖、完全可控 | 追求极致可控 | 低（概念）高（实现） |
| **OpenAI Agents SDK** | 轻量级、原生支持 | 简单 Agent | 低 |

### 选型考虑

- 团队熟悉度 → 降低学习成本
- 灵活性需求 → 原生 > 框架
- 开发速度 → 框架 > 原生
- 生产稳定性 → 考虑框架成熟度和 breaking changes
- 生态兼容 → 和现有基础设施是否匹配

## 可观测性 (Observability)

### 三大支柱

```
Logging（日志）    → 发生了什么
Tracing（追踪）    → 每一步的输入输出
Metrics（指标）    → 成功率、延迟、成本
```

### 追踪方案

| 工具 | 特点 |
|------|------|
| **LangSmith** | LangChain 官方，收费 |
| **LangFuse** | 开源，自部署 |
| **Arize Phoenix** | 开源，兼容 OpenAI |
| **Weights & Biases** | ML 实验追踪 + LLM 扩展 |
| **自定义** | 结构化日志 + 可视化 |

### 关键追踪点

```
每次 LLM 调用：
  - prompt tokens / completion tokens
  - latency
  - 完整 request / response

每次工具调用：
  - 工具名称和参数
  - 执行时间
  - 返回结果（成功/失败/异常）
  - 重试次数

Agent 级别：
  - 总步数
  - 总 token 消耗
  - 任务成功/失败
  - 终止原因
```

## 流式输出 (Streaming)

### 方案

- **SSE (Server-Sent Events)**：HTTP 单向流，简单
- **WebSocket**：全双工，适合交互式场景
- **gRPC Stream**：适合微服务内网通信

### 流式内容

- **LLM Token 流**：逐 token 输出思考过程
- **中间步骤流**：每个工具调用结果即时推送
- **状态更新流**：任务进度、当前阶段

```python
# 概念示例
async for event in agent.stream("帮我订一张去北京的机票"):
    if event.type == "thinking":
        print(f"💭 {event.content}")
    elif event.type == "tool_call":
        print(f"🔧 调用 {event.tool_name}...")
    elif event.type == "tool_result":
        print(f"📊 结果: {event.content}")
    elif event.type == "final":
        print(f"✅ 完成: {event.content}")
```

## Human-in-the-Loop (HITL)

### 介入节点

```
┌──────┐    ┌──────────┐    ┌──────┐    ┌──────────┐
│ 开始  │ → │ 分析任务  │ → │ ⚠️审批 │ → │ 执行操作  │ → ...
└──────┘    └──────────┘    └──┬───┘    └──────────┘
                               │
                          人工确认/修改/拒绝
```

### 实现模式

- **审批节点**：关键操作前暂停，人工确认后继续
- **中断点**：允许人在任意步骤介入修改
- **建议模式**：Agent 给出建议，人做最终决定
- **协作模式**：人和 Agent 交替操作

### 设计原则

- 审批节点要少而精，过多会破坏自动化体验
- 审批信息要充分：展示 Agent 将做什么、为什么
- 超时策略：人工长时间不响应时怎么处理

## 面试常见追问

1. LangChain 和直接用 API 你怎么选？为什么？
2. Agent 生产环境排障流程是怎样的？
3. 怎么控制 Agent 的单次对话成本（token 消耗）？
4. 流式输出中如果出错，怎么向前端报告？
5. 设计一个支持 1000 并发 Agent 的系统架构
