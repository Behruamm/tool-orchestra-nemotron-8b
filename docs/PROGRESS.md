# Progress Log

## Step 1 - Buyer-Facing Narrative Audit

**Status:** Done

**What I reviewed**
- `README.md` for top-level story and positioning.
- `docs/01_architecture.md` and `docs/02_models.md` for system details and terminology.
- `docs/03_tools.md` for tool schema and routing preferences.

**Findings**
- The repo already explains the technical architecture well, but it is written for engineers.
- There is no buyer-facing story that emphasizes outcomes, measurable value, or proof.
- There is no “proof” section (evaluation or real scenario results) that a buyer can trust quickly.

**Buyer-Facing Story (Proposed Outline)**
1. Problem: Orgs need fast, cost-controlled, privacy-aware agentic automation.
2. Solution: Tool Orchestra routes work across local models, cloud models, and tools based on user prefs.
3. Differentiators: Cost optimization, privacy mode, unified tool interface, traceable routing.
4. Proof: Multi-step tool-call evaluation + real scenario demo.
5. How to try: 3-step quickstart and one command to run a demo.

**Where to place it**
- Add a buyer-facing section at the top of `README.md`.
- Create a new `docs/BUYER_GUIDE.md` for the full narrative and demo flow.
- Add a short “Evaluation Method” section that references the synthetic tool-call approach.

**Next Step**
- Design the synthetic evaluation suite (task schema + 10–30 multi-step tasks).

## Step 2 - Synthetic Evaluation Suite Design

**Status:** Done

**What I created**
- `data/synthetic/README.md` defining the task schema and scoring intent.
- `data/synthetic/tasks.jsonl` with 12 multi-step tasks that require `local_search`
  + `python_sandbox`.

**Design Notes**
- Each task includes `required_facts` and `required_tools` so we can score correctness
  and tool-use behavior separately.
- Tasks are grounded in the existing knowledge base (`data/knowledge/*`) to keep the
  evaluation reproducible and privacy-safe.
- Calculations are intentionally simple and deterministic to keep tool execution verifiable.

**Next Step**
- Implement a lightweight evaluator script to run tasks through the CLI and score tool
  sequence + required facts.

## Step 3 - Evaluator Script

**Status:** Done

**What I created**
- `scripts/evaluate_synthetic.py` to run JSONL tasks through the LangGraph pipeline
  and score:
  - Required tool sequence (subsequence match)
  - Required facts in the final response

**Outputs**
- JSON report written to `reports/synthetic_eval.json` by default
- Console summary with pass rate and per-task status

**Notes / Requirements**
- Requires `GEMINI_API_KEY` because `local_search` embeddings use Gemini.
- Requires a built vector store in `data/vectorstore/` (run
  `python -m scripts.ingest_knowledge` if missing).

**Next Step**
- Create a buyer-facing results template and add 2–3 example trajectories plus a short
  cost/latency summary.

## Step 3b - Routing Prompt Guardrails

**Status:** Done

**What I changed**
- Updated `src/orchestrator/router.py` to enforce tool-use rules in the orchestrator
  system prompt:
  - Use `local_search` for doc/knowledge-based questions.
  - Use `python_sandbox` for arithmetic or counting.
  - For multi-step tasks: `local_search` → `python_sandbox` → `finish`.

**Why**
- The initial evaluation run showed 0% pass rate because the orchestrator chose `phi4`
  and skipped tools. These guardrails force the intended tool sequence for evaluation
  tasks and improve real-world reliability.

## Step 3c - JSON Enforcement + Parsing Hardening

**Status:** Done

**What I changed**
- Updated `src/models/lm_studio.py` to pass `response_format={"type": "json_object"}`
  when requested by the router (with a safe retry if unsupported).
- Updated `src/orchestrator/router.py` to request JSON mode and remove chain-of-thought
  requirements that caused `<think>` outputs.
- Updated `src/orchestrator/parser.py` to strip `<think>` tags before JSON extraction.

**Why**
- The orchestrator sometimes returned non-JSON text, causing parse failures and
  tool-selection fallbacks. This fix enforces valid JSON for tool routing.
