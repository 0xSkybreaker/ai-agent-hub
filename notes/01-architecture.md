# Agent 架构基础

## 核心组件

- **Profile（角色设定）**：系统提示词设计，角色约束，行为边界
- **Memory（记忆）**：短期（对话窗口）、长期（外部存储）
- **Planning（规划）**：任务分解、路径选择、动态调整
- **Action（工具使用）**：工具调用、结果解析、错误处理

## 主流范式

### ReAct (Reasoning + Acting)

交替进行推理和行动，每一步先思考再执行：

```
Thought → Action → Observation → Thought → Action → ...
```

- 优点：灵活、可解释性强
- 缺点：每一步都要完整推理，token 消耗大
- 论文：ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)

### Plan-and-Execute

先生成完整计划，再逐步执行：

- 优点：适合长任务，减少中间步骤的偏离
- 缺点：计划可能不适应动态变化
- 变体：Plan-and-Solve、ReWOO

### Reflexion

在执行失败后进行反思，将经验写入长期记忆：

- 核心：self-evaluation + verbal reinforcement
- 关键：反思内容的质量决定改进效果

### 其他范式

- **Tree-of-Thoughts**：多路径探索 + 剪枝
- **Graph-of-Thoughts**：更灵活的路径组合
- **LLM Compiler**：把子任务编译成 DAG 并行执行

## 多 Agent 协作

### 协作模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 流水线（Pipeline） | A → B → C 顺序传递 | 数据加工流水线 |
| 辩论（Debate） | 多个 Agent 讨论后汇总 | 需要多视角验证 |
| 投票（Voting） | 多 Agent 独立回答后投票 | 提高准确率 |
| 分层调度（Hierarchical） | 主 Agent 分配子任务 | 复杂任务分解 |
| 混合（Hybrid） | 组合以上模式 | 真实生产场景 |

### 关键问题

- 消息协议设计：JSON / 自然语言 / 结构化
- 冲突解决：投票、主 Agent 裁决、置信度比较
- 上下文共享：共享记忆、消息广播、黑板模式

## Agent 循环设计

```
while not done and steps < max_steps:
    context = build_context(memory, tools, observation)
    response = llm(context)
    action = parse_action(response)
    observation = execute(action)
    memory.append(observation)
    done = check_termination(response, observation)
```

### 终止条件

- LLM 显式输出 FINISH
- 达到最大步数
- Token 预算耗尽
- 工具返回终止信号
- 超时

## 面试常见追问

1. ReAct 和 Plan-and-Execute 分别适合什么场景？
2. 多 Agent 之间怎么共享状态？
3. 如果 Agent 陷入循环怎么办？
4. 怎么设计一个既能自主决策又不会失控的 Agent？
