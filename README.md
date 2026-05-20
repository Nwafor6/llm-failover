# LLM Failover

Multi-LLM provider client with automatic failover and priority ordering. Supports OpenAI, Anthropic, Google Gemini, xAI/Grok, and DeepSeek with seamless switching between providers.

## Features

- **Automatic Failover**: If one provider fails, automatically switch to the next priority provider
- **Priority Ordering**: Configure provider preference order dynamically
- **Vision Support**: Filter providers by vision capability
- **Streaming**: Full streaming support with callbacks for chunks and tool execution
- **Tool Calling**: Standardized tool calling interface across all providers
- **Framework Agnostic**: Works with Django, FastAPI, or standalone Python scripts

## Installation

```bash
pip install llm-failover
```

Or install from source:

```bash
git clone https://github.com/bildbridge/llm-failover.git
cd llm-failover
pip install -e .
```

## Quick Start

### Basic Usage

```python
from llm_failover import AIClientFactory

# Initialize factory with API keys (or use environment variables)
factory = AIClientFactory(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    # Other keys optional
)

# Get a client (tries preferred provider first, falls back automatically)
client, model = factory.get_client()

# Generate a response
response = await client.create_message(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

### Using Environment Variables

Set API keys in your environment:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_GENAI_API_KEY="..."
export GROK_API_KEY="xai-..."
export DEEPSEEK_API_KEY="sk-..."
export PREFERRED_AI_PROVIDER="anthropic"  # Optional
```

Then use without passing keys:

```python
from llm_failover import AIClientFactory

factory = AIClientFactory()
client, model = factory.get_client()
```

### Reordering Providers

```python
# Change priority order (only these providers will be used)
factory.reorder_clients(["anthropic", "openai", "gemini"])

# Now Anthropic is tried first, then OpenAI, then Gemini
client, model = factory.get_client()
```

### Vision Support

```python
# Get only providers that support vision
client, model = factory.get_client(require_vision=True)

# Send image with message
response = await client.create_message(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."}
                }
            ]
        }
    ]
)
```

### Streaming with Callbacks

```python
async def on_chunk(chunk: str):
    print(chunk, end="", flush=True)

async def on_tool_start(tool_name: str, args: dict):
    print(f"\nCalling tool: {tool_name} with {args}")

async def on_tool_result(tool_name: str, result: dict):
    print(f"\nTool {tool_name} returned: {result}")

# Stream response with callbacks
client, model = factory.get_client()
response = await client.create_stream(
    messages=[{"role": "user", "content": "Count to 10"}],
    max_tokens=100,
    on_chunk=on_chunk,
    on_tool_start=on_tool_start,
    on_tool_result=on_tool_result
)
```

### Custom Tool Execution

By default, tool calls return a success stub. Override `process_tool_calls` in a subclass:

```python
from llm_failover.clients import OpenAIClient

class MyOpenAIClient(OpenAIClient):
    async def process_tool_calls(self, tool_calls):
        results = {}
        for tool_call in tool_calls:
            if tool_call["name"] == "get_weather":
                # Your custom tool execution
                location = tool_call["arguments"]["location"]
                results[tool_call["id"]] = {
                    "success": True,
                    "result": f"Weather in {location}: Sunny, 72°F"
                }
        return results

# Use custom client in factory
factory.model_priority[3]["client_class"] = MyOpenAIClient
```

## Supported Providers

| Provider | Model Default | Vision Support | Notes |
|----------|---------------|----------------|-------|
| Gemini | `gemini-1.5-flash` | ✅ | Google's Gemini via OpenAI-compatible endpoint |
| Anthropic | `claude-3-sonnet-20240229` | ✅ | Claude models with content block streaming |
| xAI/Grok | `grok-beta` | ✅ | Grok models from xAI |
| OpenAI | `gpt-4o` | ✅ | GPT models with function calling |
| DeepSeek | `deepseek-chat` | ❌ | DeepSeek models (no vision) |

## API Reference

### AIClientFactory

#### `__init__(gemini_api_key=None, anthropic_api_key=None, ...)`

Initialize factory with optional API keys. Falls back to environment variables.

#### `get_client(fallback=False, require_vision=False, **kwargs) -> Tuple[AIAgentClient, str]`

Get a client instance and model name.

- `fallback`: Skip preferred provider, use first available
- `require_vision`: Only return vision-capable providers
- `**kwargs`: Passed to client initialization

#### `reorder_clients(provider_order: List[str])`

Reorder providers by priority. Only specified providers will be used.

#### `update_model(provider: str, model: str)`

Change the model for a specific provider.

#### `list_providers() -> List[Dict]`

Get list of all configured providers with their settings.

### AIAgentClient (Base Class)

All provider clients inherit from this base class.

#### `async create_message(messages, **kwargs) -> dict`

Generate a non-streaming response.

#### `async create_stream(messages, on_chunk=None, on_tool_start=None, on_tool_result=None, **kwargs) -> dict`

Generate a streaming response with callbacks.

#### `async process_tool_calls(tool_calls) -> dict`

Process tool calls. Override in subclasses for custom tools.

## Examples

### Use in FastAPI

```python
from fastapi import FastAPI
from llm_failover import AIClientFactory

app = FastAPI()
factory = AIClientFactory()

@app.post("/chat")
async def chat(message: str):
    client, model = factory.get_client()
    response = await client.create_message(
        messages=[{"role": "user", "content": message}]
    )
    return {"response": response["content"], "model": model}
```

### Use in Django

```python
# views.py
from llm_failover import AIClientFactory
import asyncio

factory = AIClientFactory()

async def generate_response(user_message):
    client, model = factory.get_client()
    return await client.create_message(
        messages=[{"role": "user", "content": user_message}]
    )

def chat_view(request):
    message = request.POST.get("message")
    response = asyncio.run(generate_response(message))
    return JsonResponse(response)
```

### Standalone Script

```python
import asyncio
from llm_failover import AIClientFactory

async def main():
    factory = AIClientFactory()
    
    # Try vision-capable providers
    client, model = factory.get_client(require_vision=True)
    print(f"Using {model}")
    
    response = await client.create_message(
        messages=[{"role": "user", "content": "Explain quantum computing"}],
        max_tokens=500
    )
    
    print(response["content"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Development

```bash
# Clone repository
git clone https://github.com/bildbridge/llm-failover.git
cd llm-failover

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
