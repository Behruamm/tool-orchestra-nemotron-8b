# Models and Routing

## Available Models

### Local Models (via LM Studio)

**Nemotron Orchestrator 8B**: The routing model that decides which tool to invoke next. Runs locally with zero API cost. Trained for function calling and tool selection.

**Phi-4 Mini Instruct**: Microsoft's efficient small model for general text tasks. Handles summarization, simple Q&A, translation, and basic analysis. Zero API cost, runs entirely on local hardware.

### Cloud Models

**Gemini 2.0 Flash**: Google's latest efficient cloud model. Used for complex reasoning, multi-step analysis, and tasks requiring world knowledge. Free tier available with rate limits.

**Gemini 1.5 Pro**: Higher capability model for the most demanding tasks. Higher API cost but superior reasoning quality.

## Routing Logic

The Router class in `src/orchestrator/router.py` implements the routing decision logic:

1. If privacy=True, only local tools are considered
2. If budget < 0.3, strongly prefer local Phi-4
3. If quality > 0.7, prefer Gemini for better results
4. Tool selection based on query type (search, code execution, reasoning)

## Model Interface

All models implement `BaseModelClient` with:
- `chat()` / `achat()` for sync/async completions
- `is_local` property indicating API cost
- `calculate_cost()` for token-based pricing
- Unified response format via `ModelResponse`
