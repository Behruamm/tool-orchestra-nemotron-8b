# Available Tools

The following tools are available to the Orchestrator-8B model:

## 1. Web Search
- **Name**: `web_search`
- **Purpose**: Retrieve current information from the internet.
- **Provider**: Brave Search API.
- **Parameters**: `query` (string)

## 2. Python Sandbox
- **Name**: `python_sandbox`
- **Purpose**: Execute Python code for calculations, data analysis, or complex logic that requires precision.
- **Environment**: Restricted local execution.
- **Parameters**: `code` (string)

## 3. Local Search (RAG)
- **Name**: `local_search`
- **Purpose**: Search the local knowledge base (documents in `data/knowledge`).
- **Use Case**: Retrieving specific facts from uploaded documents.
- **Parameters**: `query` (string), `top_k` (integer)

## 4. Phi-4 Model
- **Name**: `phi4`
- **Purpose**: Quick, low-cost reasoning or text generation.
- **Use Case**: Summarizing search results, rephrasing queries, simple extraction.
- **Parameters**: `prompt` (string)

## 5. Gemini Model
- **Name**: `gemini`
- **Purpose**: High-capability reasoning.
- **Use Case**: Complex analysis, difficult logic, creative writing, or situations where Phi-4 fails.
- **Parameters**: `prompt` (string)

## 6. Finish
- **Name**: `finish`
- **Purpose**: Signal that the task is complete and provide the final answer to the user.
- **Parameters**: `answer` (string), `sources` (list of strings)
