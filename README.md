# Tool Orchestra (Nemotron 8B)

Intelligent research agent powered by NVIDIA's Orchestrator-8B model.

## How It Works

```
User Query → Orchestrator-8B → Tool Calls → Answer + Sources
```

The system uses a flexible, model-driven architecture where **Orchestrator-8B** acts as the central brain. It decides which tools to call, when to search, when to code, and when to finish.

## Key Features

- **Brain**: Orchestrator-8B (Local) routes and plans.
- **Specialists**:
  - **Phi-4** (Local): Fast, free model for quick tasks.
  - **Gemini** (Cloud): Powerful model for complex reasoning.
- **Tools**: Integrated web search, python sandbox, and local RAG.

## Available Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via Brave API |
| `python_sandbox` | Execute Python code locally |
| `local_search` | RAG search over local documents |
| `phi4` | Call Phi-4 for quick local reasoning |
| `gemini` | Call Gemini for complex reasoning |
| `finish` | Return final answer to user |

## Usage

```bash
# Single query
python -m src.main query "What is the capital of France?"

# Interactive chat
python -m src.main chat

# List tools
python -m src.main tools

# Show config
python -m src.main config
```

## Prerequisites

1. **LM Studio** with loaded models:
   - `nemotron-orchestrator-8b` (Brain)
   - `microsoft-phi-4-mini-instruct` (Worker)
2. **API Keys** in `.env`:
   ```
   GEMINI_API_KEY=your_key
   BRAVE_API_KEY=your_key
   ```

## Documentation

- [Architecture Overview](docs/01_architecture.md)
- [Available Tools](docs/02_tools.md)

## Project Structure

```
src/
├── orchestrator/
│   ├── loop.py          # Core execution loop
│   ├── parser.py        # JSON response parsing
│   └── actions.py       # Action schemas
├── tools/
│   ├── web_search.py    # Brave Search
│   ├── python_sandbox.py
│   ├── llm_tools/       # phi4.py, gemini.py
│   └── finish.py
├── models/
│   ├── lm_studio.py     # Local model client
│   └── gemini.py        # Cloud model client
└── main.py              # CLI
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
