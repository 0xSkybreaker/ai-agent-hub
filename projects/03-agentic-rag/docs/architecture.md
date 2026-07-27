# Agentic RAG — Architecture Document

> One-pager for onboarding and architecture review.

---

## 1. One-Line Summary

**Agentic RAG** puts a ReAct Agent in control of RAG retrieval — the agent decides *what* to search, *evaluates* results, *reformulates* when needed, and *synthesizes* from multiple search rounds. It combines the agent loop from 01-mini-agent with the RAG infrastructure from 02-rag-agent.

---

## 2. The Problem with Basic RAG

Basic RAG (02-rag-agent) is a fixed pipeline:

```
User question → Embed query → Retrieve top-K → Stuff into prompt → Generate answer
```

This has limitations:

| Problem | Example |
|---------|---------|
| **Query formulation**: The user's raw question may not be the best search query | "Tell me about that thing from the paper" → poor retrieval |
| **No iteration**: If top-K results miss the answer, there's no second chance | First search misses; system says "I don't know" |
| **No evaluation**: No way to know if retrieved chunks actually contain the answer | Gets irrelevant chunks but generates anyway |
| **Single-pass**: Can't break complex questions into sub-searches | "Compare A with B" → needs two searches |
| **No verification**: No check that the generated answer is supported by context | Hallucinated claims go undetected |

---

## 3. How Agentic RAG Solves It

Agentic RAG replaces the fixed pipeline with a ReAct decision loop:

```
User question → Agent thinks → Agent acts (search) → Agent observes (results) → Agent thinks (evaluate) → Agent acts (re-search/reformulate) → Agent thinks (synthesize) → Answer
```

The key innovations:

1. **Query formulation by the LLM**: The agent thinks about *what* to search for, not just using the raw question
2. **Iterative retrieval**: Can search multiple times with different queries
3. **Quality evaluation**: Checks relevance scores before deciding to use results
4. **Reformulation**: If results are poor, tries different keywords or phrasings
5. **Multi-hop decomposition**: Breaks complex questions into sub-searches
6. **Self-verification**: Checks generated claims against retrieved context

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│            CLI (Typer+Rich)  │  FastAPI REST + SSE          │
└──────────────┬──────────────────┬──────────────────────────┘
               │                  │
┌──────────────▼──────────────────▼──────────────────────────┐
│               AgenticRAGAgent (ReAct Loop)                  │
│                                                             │
│  System Prompt: teaches LLM agentic RAG workflow            │
│                                                             │
│  while not done:                                            │
│    LLM decides: call tool OR give final answer              │
│    if tool: execute → feed result back → next iteration     │
│    if answer: return with sources                           │
│                                                             │
│  Tools:                                                     │
│    search_documents(query, top_k)                           │
│    get_document(source_path)                                │
│    list_documents()                                         │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Tool:  │ │Tool:  │ │Tool:  │
│search │ │get_doc│ │list   │
│docs   │ │       │ │docs   │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └────┬────┴────┬────┘
         │         │
    ┌────▼─────────▼────┐
    │  rag-agent infra  │
    │  (reused)         │
    │                   │
    │  Retriever        │
    │  ChromaVectorStore│
    │  EmbeddingClient  │
    └───────────────────┘
```

---

## 5. Core Data Flow

### Example: "Compare RAG with fine-tuning for LLM knowledge updates"

```
Step 1: Agent thinks
  "This is a comparison question. I need info on both RAG and fine-tuning."
  → Calls: list_documents()

Step 2: Agent observes
  "I have documents about RAG, LLMs, and model training."
  → Calls: search_documents(query="RAG benefits for knowledge update")

Step 3: Agent evaluates
  "Got 5 results about RAG, relevance 0.7-0.9. Good."
  → Calls: search_documents(query="fine-tuning limitations knowledge update")

Step 4: Agent evaluates
  "Got 3 results about fine-tuning. Now I can compare."
  → Final answer: synthesizes from both search rounds, cites sources.
```

Total: 4 steps (1 list + 2 searches + 1 answer). Compare to basic RAG where the raw question would be used as the single query.

---

## 6. Code Architecture

### Import Strategy

```
03-agentic-rag/
  src/agentic_rag/__init__.py
    → sys.path.insert(0, "../../02-rag-agent/src")
    → Makes rag_agent.* importable
```

This is simpler than restructuring the repo (which would break the narrative of each project being standalone). The `__init__.py` handles the import path magic — all other modules just do normal imports.

### Component Wiring

```
config.py (Settings from .env)
    ↓
llm.py (OpenAI SDK client)
    ↓
tools.py (ToolRegistry with RAG tools)
    │  ↓ lazily: Retriever + ChromaVectorStore + EmbeddingClient
    │  (all imported from rag_agent.*)
    ↓
agent.py (AgenticRAGAgent)
    │  ← prompts.py (AGENTIC_RAG_SYSTEM_PROMPT)
    │  ← tools.py (ToolRegistry)
    ↓
cli.py (Typer commands) / api/ (FastAPI)
```

### Why Not Reuse the Generator?

rag-agent's `Generator` class is designed for one-shot RAG — it calls `retrieve()` once, then `generate()` once. This is the wrong abstraction for an agentic system that needs:

- Multiple retrievals with different queries
- Evaluation between retrievals
- Conditional logic (if results poor → reformulate)
- Synthesis from accumulated context

Instead, the agent directly uses `Retriever` and `ChromaVectorStore` — the same lower-level components the Generator uses.

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **OpenAI SDK, not LangChain** for agent loop | Function calling is central to the ReAct pattern. LangChain adds abstraction overhead. Matching 01-mini-agent pattern keeps it learnable. |
| **System prompt encodes agentic behavior** | No hardcoded evaluation logic. The LLM reads the workflow from the prompt and decides when to reformulate, verify, etc. This is the standard "vibe-driven" agent pattern. |
| **Tools close over RAG components via lazy init** | Avoids complex DI. Tools lazily import and cache Retriever/VectorStore on first use. Testable by patching the module-level singletons. |
| **Shared data with rag-agent** | Default config points at `../02-rag-agent/data/chroma`. Users index once, query both ways. Demonstrates infrastructure reuse. |
| **Single tool call per step** | Matches 01-mini-agent pattern. Simpler loop, clearer trace. Could extend to parallel calls later. |
| **Port 8001 (vs 8000 for rag-agent)** | Both APIs can run simultaneously for comparison. |

---

## 8. Comparison: Basic vs Agentic

| Dimension | 02-rag-agent (Basic) | 03-agentic-rag (Agentic) |
|-----------|---------------------|--------------------------|
| Core loop | Fixed pipeline | ReAct decision loop |
| Retrieval | One-shot, uses raw question | Multi-step, agent formulates queries |
| Query quality | Depends on user's phrasing | Agent optimizes query for retrieval |
| Result evaluation | None — uses all top-K results | Agent checks relevance scores |
| Reformulation | Not possible | Agent tries different queries |
| Multi-hop questions | Poor — single search can't cover all parts | Good — agent searches each sub-question |
| Answer verification | None | Agent checks claims against context |
| Step trace | No visibility | Full step-by-step trace |
| API response | answer + sources | answer + steps + sources |
| Weakness for simple facts | N/A (works fine) | Overhead — may take more steps for simple questions |

---

## 9. Extension Points

1. **Conversation memory**: Track multi-turn chat. Reuse `ConversationMemory` from rag-agent. Add a session_id to agent state.
2. **Verification tool**: A `verify_claim(claim, sources)` tool that the agent can call before finalizing.
3. **Query history tracking**: Prevent the agent from searching the exact same query twice.
4. **Parallel tool calls**: Support `tool_calls[0]`, `tool_calls[1]`, ... for concurrent searches.
5. **Agentic indexing**: Agent decides *which* documents to index based on user needs.
6. **Confidence scoring**: Agent quantifies its confidence in the answer (high/medium/low).
