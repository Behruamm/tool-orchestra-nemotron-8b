# Synthetic Evaluation Suite

This folder defines a small, buyer-facing evaluation suite for multi-step tool use.
Each task requires **local_search** (RAG) and **python_sandbox** (calculator) so
we can demonstrate routing + tool execution in a verifiable way.

## Task Format

We use JSON Lines in `tasks.jsonl`, one task per line:

```
{
  "id": "ts_001",
  "domain": "tool_orchestra_docs",
  "complexity": "simple|medium",
  "task": "User request to solve",
  "required_facts": ["facts that must appear in the answer"],
  "required_tools": ["local_search", "python_sandbox"],
  "golden_tool_calls": [
    {"tool": "local_search", "args": {"query": "...", "top_k": 3}},
    {"tool": "python_sandbox", "args": {"code": "..."}}
  ],
  "notes": "Optional guidance for the evaluator"
}
```

## Scoring Intent

The evaluator (built in the next step) will score a task as **pass** when:
- The tool call sequence contains the `required_tools` in order.
- The final answer contains all `required_facts`.

## Why This Matters (Buyer-Facing)

This suite mirrors the paper’s idea of **synthetic, verifiable tool-use tasks**:
it proves the system can **find internal facts** and **compute deterministic outputs**
through tools instead of hallucinating. This is the core value for B2B buyers.
