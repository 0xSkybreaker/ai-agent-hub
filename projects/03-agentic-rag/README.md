# 03-agentic-rag — Agentic RAG

**ReAct Agent + RAG = Autonomous document Q&A with iterative search and self-verification.**

Basic RAG is a one-shot pipeline: query → retrieve → generate. Agentic RAG puts an Agent in control — it decides *what* to search for, *evaluates* whether results are sufficient, *reformulates* when needed, and *synthesizes* from multiple search rounds.

## The Key Difference

| Basic RAG (02) | Agentic RAG (03) |
|---|---|
| One-shot: query → retrieve → generate | Multi-step: think → search → evaluate → reformulate → synthesize |
| Fixed retrieval strategy | Agent chooses search queries |
| No quality check | Agent evaluates relevance scores |
| Single query | Breaks complex questions into sub-searches |
| No self-verification | Agent verifies claims against context |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                AgenticRAGAgent (ReAct Loop)             │
│                                                        │
│   ┌──────────┐     ┌──────────────┐     ┌──────────┐  │
│   │   LLM    │ ──→ │ Tool Choice  │ ──→ │   Tool   │  │
│   │ (think)  │ ←── │ (function    │ ←── │ (execute)│  │
│   └──────────┘     │  calling)    │     └──────────┘  │
│                    └──────────────┘                    │
│                          │                             │
│   Tools:                │                             │
│   • search_documents ───┘                              │
│   • get_document                                       │
│   • list_documents                                     │
│                                                        │
│   Each turn: Thought → Tool call → Result → ...       │
│   Until: Agent decides it has enough to answer         │
└────────────────────────────────────────────────────────┘
```

The agent reuses rag-agent's infrastructure (ChromaDB, Retriever, EmbeddingClient)
but replaces the fixed pipeline with a ReAct decision loop.

## File Structure

```
src/agentic_rag/
├── __init__.py       ← Import strategy: adds rag-agent to sys.path
├── __main__.py       ← Entry: python -m agentic_rag
├── config.py         ← pydantic-settings
├── llm.py            ← OpenAI SDK wrapper (native function calling)
├── prompts.py        ← System prompts — teaches the agent to be "agentic"
├── tools.py          ← ★ RAG tools: search, inspect, list documents
├── agent.py          ← ★ AgenticRAGAgent — the core ReAct loop
├── cli.py            ← CLI (typer + rich)
└── api/              ← FastAPI REST + SSE streaming
    ├── server.py
    ├── routes.py
    ├── models.py
    └── dependencies.py
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env, fill in NVIDIA_API_KEY
# ChromaDB defaults to sharing 02-rag-agent's data:
#   CHROMA_PERSIST_DIR=../02-rag-agent/data/chroma
```

### 3. Index Documents (via rag-agent)

```bash
cd ../02-rag-agent
python -m rag_agent index ./data/
```

### 4. Ask Questions

```bash
# Single question
python -m agentic_rag run "What are the key benefits of RAG and how does it compare to fine-tuning?"

# Interactive chat
python -m agentic_rag chat

# List tools
python -m agentic_rag tools

# Start API
python -m agentic_rag serve
```

### 5. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/query` | Q&A with step trace |
| POST | `/api/v1/query/stream` | SSE streaming (step-by-step) |

```bash
# Non-streaming
curl -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Streaming
curl -X POST http://127.0.0.1:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Tools

| Tool | Purpose | When the Agent Uses It |
|------|---------|----------------------|
| `search_documents` | Semantic search | Primary information gathering — searches with specific queries |
| `get_document` | Read full document | After finding a promising source via search |
| `list_documents` | Show available docs | At session start, to understand knowledge coverage |

## The Agent's Decision Cycle

```
1. ANALYZE the question
   "What kind of information do I need?"

2. LIST available documents
   "What topics are covered?"

3. SEARCH with well-formulated queries
   "What specific terms match the question?"

4. EVALUATE results
   "Are these relevant? Scores above 0.3?"

5. REFORMULATE if needed
   "Let me try different keywords..."

6. DEEP-DIVE into promising documents
   "This doc looks relevant — let me read it fully."

7. SYNTHESIZE from all gathered context
   "What do all the search results tell me?"

8. VERIFY before answering
   "Does the context actually support my claims?"

→ Final answer with source citations
```

## Design Decisions

| Decision | Why |
|----------|-----|
| OpenAI SDK (not LangChain) for agent loop | Native function calling is clearer; matches 01-mini-agent |
| Reuse rag-agent infrastructure via sys.path | Don't copy code — compose. The retriever/vector store are production components |
| Tools return formatted text | Easiest for the LLM to read and reason over |
| Shared ChromaDB data directory | Index once with rag-agent, query agentically with agentic-rag |
| System prompt encodes the workflow | The LLM decides when to reformulate, when to verify — no hardcoded rules |
| Single tool call per step | Clearer trace, simpler loop, matches 01-mini-agent pattern |

## Comparison: Run the Same Question

Try running the same question through both systems:

```bash
# Basic RAG (one-shot)
cd ../02-rag-agent
python -m rag_agent query "Compare RAG with fine-tuning for LLM knowledge updates"

# Agentic RAG (multi-step)
cd ../03-agentic-rag
python -m agentic_rag run "Compare RAG with fine-tuning for LLM knowledge updates"
```

Expected difference: The agentic version may take 3-5 steps (list → search "RAG benefits" → search "fine-tuning limitations" → synthesize) while basic RAG does it in one shot. For complex, multi-part questions, the agentic approach produces more thorough answers.

## Extensions

- **Conversation memory**: Track multi-turn chat context (reuse ConversationMemory from rag-agent)
- **Verification tool**: Add a `verify_claim` tool that searches for evidence supporting/refuting specific claims
- **Query history**: The agent remembers what it has already searched for
- **Parallel tool calls**: Search multiple queries simultaneously

## License

MIT
