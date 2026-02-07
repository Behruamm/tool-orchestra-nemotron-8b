# Tools Reference

## Tool System Architecture

All tools extend `BaseTool` and register through a global `ToolRegistry`. Each tool provides:
- A `ToolConfig` with name, description, JSON schema, cost/latency estimates
- A `run()` method returning a standardized `ToolResult`
- Optional `arun()` for async execution

## Available Tools

### Python Sandbox
- **Name**: `python_sandbox`
- **Type**: Local (no API cost)
- **Purpose**: Execute Python code safely with restricted globals
- **Allowed modules**: math, datetime, json, random
- **Parameters**: `code` (string)

### Web Search
- **Name**: `web_search`
- **Type**: External API (Brave Search)
- **Purpose**: Search the public internet for real-time information
- **Parameters**: `query` (string), `num_results` (integer, default 5)
- **Note**: Disabled when privacy=True

### Local Search
- **Name**: `local_search`
- **Type**: Local (FAISS vector store)
- **Purpose**: RAG search against local knowledge base documents
- **Parameters**: `query` (string), `top_k` (integer, default 3)
- **Note**: Always available regardless of privacy settings

### Phi-4 (LLM as Tool)
- **Name**: `phi4`
- **Type**: Local (no API cost)
- **Purpose**: General text tasks via local Phi-4 model
- **Parameters**: `prompt`, `system_prompt`, `max_tokens`

### Gemini (LLM as Tool)
- **Name**: `gemini`
- **Type**: Cloud API
- **Purpose**: Complex reasoning via Google Gemini
- **Parameters**: `prompt`, `system_prompt`, `max_tokens`

### Finish
- **Name**: `finish`
- **Type**: Special (terminal)
- **Purpose**: End the workflow and return final response
- **Parameters**: `answer`, `confidence`, `sources`
