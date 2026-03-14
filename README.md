# MCP-as-API-Wrapper

A thin MCP server that wraps any API. Instead of tools doing the work, the MCP server provides **3 stable tools** for discovery, authentication, and documentation. The full API evolves independently behind this stable layer.

## The Idea

Traditional MCP servers expose tools that do the work directly. Every new capability means a new tool, and API changes require server updates + client reconnects.

**This inverts that pattern.** The MCP server has exactly 3 tools that never change:

| Tool | Purpose |
|------|---------|
| `getAPIEndpoints` | Discover what the API can do |
| `getAuthToken` | Get a scoped Bearer token for API access |
| `getDocumentation` | Get the URL to download full API docs |

The client agent connects via MCP once, uses these 3 tools to bootstrap, then talks directly to the API using Bearer tokens. When the API evolves, clients just re-fetch the docs — no MCP reconnect needed.

## Architecture

```
┌─────────────┐     MCP Protocol     ┌──────────────────┐
│ Client Agent │◄───────────────────►│  Thin MCP Server  │
│  (MCP Client)│                     │   (3 tools only)  │
└──────┬───────┘                     └────────┬──────────┘
       │                                      │
       │  Direct API Calls                    │ Discovery
       │  (Bearer Token)                      │ + Auth
       │                                      │
       ▼                                      ▼
┌──────────────┐                     ┌──────────────────┐
│  API Server   │◄──── deploys ────│  Backend Agent     │
│  (Full API)   │                   │  (API Manager)    │
└──────────────┘                     └────────┬──────────┘
                                              │
                                     ┌────────▼──────────┐
                                     │  Message Queue     │
                                     │  (Agent Comms)     │
                                     └───────────────────┘
```

## Key Features

- **Zero reconnects** — API evolves, MCP stays stable
- **Decoupled auth** — API-level auth (2FA, OAuth, RBAC) independent of MCP session
- **Agent-managed API** — backend agent responds to client requests to evolve the API
- **Dynamic integration** — two agents negotiate capabilities at runtime

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Run the MCP server (stdio mode)
mcp-api-wrapper

# Run the example API
uvicorn mcp_api_wrapper.api.example_api:create_example_api --factory --port 8000
```

## Project Structure

```
src/mcp_api_wrapper/
├── server/main.py          # The 3-tool MCP server
├── api/
│   ├── registry.py         # API endpoint registry
│   └── example_api.py      # Example FastAPI app
├── auth/
│   └── token_service.py    # JWT token generation/validation
├── queue/
│   └── message_queue.py    # Agent-to-agent message queue
├── schemas.py              # Shared Pydantic schemas
└── config.py               # Settings
```

## License

MIT
