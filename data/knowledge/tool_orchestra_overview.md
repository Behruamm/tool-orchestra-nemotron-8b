# Tool Orchestra Overview

Tool Orchestra is a cost-aware multi-model orchestration framework built on LangGraph. It intelligently routes tasks between local models (Phi-4 via LM Studio) and cloud models (Gemini) based on user preferences.

## Core Concept

The system treats both LLMs and traditional tools as first-class "tools" in a unified interface. An orchestrator model analyzes each query and decides which tool to invoke next, considering cost, quality, speed, and privacy preferences.

## Preference Vector

Four parameters control routing decisions:
- **budget** (0-1): 0 = minimize cost (prefer local models), 1 = maximize quality (allow cloud models)
- **privacy** (boolean): true = local models only, no external API calls
- **speed** (0-1): 0 = fastest response, 1 = most thorough analysis
- **quality** (0-1): 0 = acceptable quality, 1 = best possible quality

## Cost Optimization

The framework aims to route 60%+ of tasks through free local Phi-4 model, reserving Gemini for complex reasoning that requires cloud-tier capabilities. This significantly reduces API costs while maintaining quality.

## Architecture

The system uses a 5-node LangGraph StateGraph:
1. **orchestrator** - Entry point that analyzes queries and decides next action
2. **phi4** - Local model node for cost-efficient processing
3. **gemini** - Cloud model node for complex reasoning
4. **tools** - Executes tool calls (python_sandbox, local_search, web_search)
5. **aggregate** - Collects results and decides whether to loop or finish
