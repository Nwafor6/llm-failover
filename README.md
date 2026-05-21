# LLM Failover

**Simple, automatic failover across multiple LLM providers.** No vendor lock-in. No manual retry logic. Just call `chat()` or `stream()` and let the package handle the rest.

Supports OpenAI, Anthropic Claude, Google Gemini, xAI/Grok, and DeepSeek with seamless automatic switching when a provider fails.

## Features

- **Zero-Configuration Failover**: Automatically tries providers in priority order until one succeeds
- **Simple API**: Just two methods - `chat()` for non-streaming, `stream()` for streaming
- **Multi-Turn Conversations**: Built-in history management with `keep_history=True`
- **Provider Priority**: Configure which providers to try and in what order
- **Vision Support**: Automatic filtering for vision-capable providers
- **Async Callbacks**: Full async/await support for streaming with callbacks
- **Framework Agnostic**: Pure async Python - integrate with any async framework (FastAPI, Django, Flask, etc.) or use standalone
- **Error Context Propagation**: Failed attempts inform retry strategy

## Installation

```bash
pip install llm-failover
```

**Development Installation:**

```bash
git clone https://github.com/Nwafor6/llm-failover.git
cd llm-failover
pip install -e ".[dev]"
```

**Requirements:**
- Python 3.8+
- `aiohttp` - Async HTTP client
- `anthropic` - Anthropic SDK (for Claude)
- `openai` - OpenAI SDK (also used for Gemini, Grok, DeepSeek via compatible endpoints)

## Quick Start

**The simplest way to use llm-failover:**

```python
import asyncio
from llm_failover import ChatClient

async def main():
    # Initialize once
    client = ChatClient()
    
    # Chat (non-streaming) - failover happens automatically!
    response = await client.chat("What is Python?")
    print(response["content"])
    print(f"Used: {response['provider']} ({response['model']})")

asyncio.run(main())
```

**That's it!** The package automatically:
- Tries providers in order (Gemini → Anthropic → xAI → OpenAI → DeepSeek)
- Handles failures and retries with the next provider
- Returns the response with metadata about which provider succeeded

### Streaming with Callbacks

```python
from llm_failover import ChatClient

client = ChatClient()

# Define callback for real-time chunks
def on_chunk(chunk: str):
    print(chunk, end="", flush=True)

# Stream response - failover is automatic!
response = await client.stream(
    "Tell me a story",
    on_chunk=on_chunk
)

print(f"\n\nProvider: {response['provider']}")
```

### Multi-Turn Conversations

Keep conversation history automatically:

```python
client = ChatClient()

# First message - context is saved
response = await client.chat(
    "My name is Alice.",
    keep_history=True
)

# Follow-up - remembers previous context
response = await client.chat(
    "What's my name?",
    keep_history=True
)
print(response["content"])  # "Your name is Alice."

# Clear history when starting new conversation
client.clear_history()
```

### Custom Configuration

```python
# Customize provider order, system message, and defaults
client = ChatClient(
    provider_order=["xai", "anthropic", "openai"],  # Only try these 3
    system_message="You are a helpful coding assistant.",
    max_tokens=500
)

# Pass additional parameters per request
response = await client.chat(
    "How do I reverse a list in Python?",
    temperature=0.7,
    max_tokens=200  # Override default
)
```

## Supported Providers

Default provider priority order (tries each in sequence until one succeeds):

| Priority | Provider | Default Model | Vision | Notes |
|----------|----------|--------------|--------|-------|
| 1 | Gemini | `gemini-3-flash-preview` | ✅ | Google's latest Gemini via OpenAI-compatible API |
| 2 | Anthropic | `claude-3-5-sonnet-20241022` | ✅ | Claude 3.5 Sonnet with streaming support |
| 3 | xAI/Grok | `grok-4.3` | ✅ | Latest Grok model from xAI |
| 4 | OpenAI | `gpt-4o` | ✅ | GPT-4 Omni with vision and function calling |
| 5 | DeepSeek | `deepseek-chat` | ❌ | Cost-effective option (no vision support) |

**Note:** You can reorder or limit providers using `provider_order` parameter:

```python
# Only use Anthropic and OpenAI, in that order
client = ChatClient(provider_order=["anthropic", "openai"])
```

## Environment Variables

Set API keys via environment variables (recommended for production):

```bash
export GOOGLE_GENAI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="sk-ant-..."
export GROK_API_KEY="xai-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# Optional: set preferred provider (default: gemini)
export PREFERRED_AI_PROVIDER="anthropic"
```

**What does `PREFERRED_AI_PROVIDER` do?**

This variable sets which provider to **try first** when making requests. The failover system will:
1. Try the preferred provider first
2. If it fails (rate limit, API error, etc.), automatically fall back to other available providers
3. Default is `"gemini"` if not set

**Examples:**
- `PREFERRED_AI_PROVIDER="anthropic"` → tries Anthropic's Claude first, falls back to others if needed
- `PREFERRED_AI_PROVIDER="openai"` → tries OpenAI's GPT-4o first, falls back to others if needed
- Not set → defaults to Gemini first

This is a **convenience setting** to prioritize your favorite provider without hardcoding it. All providers with valid API keys remain available as fallbacks.

Then initialize without passing keys:

```python
client = ChatClient()  # Reads from environment, uses PREFERRED_AI_PROVIDER
```

## API Reference

### ChatClient (Recommended)

The high-level interface that handles all failover logic automatically.

#### `__init__(provider_order=None, system_message="", max_tokens=4096, **factory_kwargs)`

Initialize the ChatClient.

**Parameters:**
- `provider_order` (list, optional): List of provider names to try in order. Example: `["xai", "anthropic", "openai"]`
- `system_message` (str, optional): Default system message for all requests
- `max_tokens` (int, optional): Default max tokens (default: 4096)
- `**factory_kwargs`: Additional arguments passed to `AIClientFactory` (e.g., API keys, custom models)

**Example:**
```python
client = ChatClient(
    provider_order=["anthropic", "openai"],
    system_message="You are a helpful assistant.",
    anthropic_api_key="sk-ant-...",  # Or use environment variables
    max_tokens=500
)
```

#### `async chat(message=None, messages=None, keep_history=False, max_tokens=None, **kwargs)`

Generate a non-streaming response with automatic failover.

**Parameters:**
- `message` (str, optional): Simple string message (convenience parameter)
- `messages` (list, optional): Full message history in OpenAI format. Use this OR `message`, not both.
- `keep_history` (bool, optional): If True, maintains conversation history across calls (default: False)
- `max_tokens` (int, optional): Override default max_tokens for this request
- `**kwargs`: Additional parameters passed to the provider (e.g., `temperature`, `top_p`)

**Returns:** dict with keys:
- `content` (str): The generated response text
- `provider` (str): Which provider was used (e.g., "anthropic")
- `model` (str): Which model was used (e.g., "claude-3-5-sonnet-20241022")
- `attempt` (int): Which attempt succeeded (1 = first provider, 2 = second, etc.)

**Example:**
```python
# Simple message
response = await client.chat("Hello!")
print(response["content"])

# With parameters
response = await client.chat(
    "Explain quantum physics",
    temperature=0.7,
    max_tokens=200
)

# Multi-turn with history
response = await client.chat("My name is Bob", keep_history=True)
response = await client.chat("What's my name?", keep_history=True)
```

#### `async stream(message=None, messages=None, keep_history=False, on_chunk=None, on_tool_start=None, on_tool_result=None, **kwargs)`

Generate a streaming response with callbacks and automatic failover.

**Parameters:**
- `message` (str, optional): Simple string message
- `messages` (list, optional): Full message history
- `keep_history` (bool, optional): Maintain conversation history
- `on_chunk` (callable, optional): Callback for each text chunk. Can be sync or async function.
- `on_tool_start` (callable, optional): Callback when tool execution starts
- `on_tool_result` (callable, optional): Callback when tool execution completes
- `**kwargs`: Additional provider parameters

**Returns:** dict with same keys as `chat()` plus:
- `content` (str): Full accumulated response

**Example:**
```python
def on_chunk(chunk: str):
    print(chunk, end="", flush=True)

async def on_chunk_async(chunk: str):
    await some_async_operation(chunk)

response = await client.stream(
    "Tell me a story",
    on_chunk=on_chunk,  # Sync or async both work
    temperature=0.9
)
```

#### `clear_history()`

Clear the conversation history.

```python
client.clear_history()
```

#### `get_history()`

Get the current conversation history.

```python
history = client.get_history()
# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
```

#### `set_provider_order(provider_order: list)`

Change the provider priority order.

```python
client.set_provider_order(["openai", "anthropic"])
```

## Advanced: Using AIClientFactory

For advanced use cases where you need fine-grained control over client initialization and management, you can use the `AIClientFactory` directly:

```python
from llm_failover import AIClientFactory

# Initialize factory
factory = AIClientFactory(
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-...",
    gemini_model="gemini-2.0-flash-exp"  # Custom model
)

# Get a client with specific requirements
client, model = factory.get_client(
    require_vision=True,  # Only vision-capable providers
    fallback=False  # Use preferred provider
)

# Use client directly
response = await client.create_message(
    model=model,
    system_message="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Clean up resources (closes HTTP sessions)
await client.close()

# Reorder providers dynamically - changes which providers to try and in what order
# This also updates the preferred provider to the first one in the list
factory.reorder_clients(["openai", "anthropic"])

# Update a provider's model - change which model version a provider uses
# Useful for switching between different model variants (e.g., gpt-4o vs gpt-4o-mini)
factory.update_model("openai", "gpt-4o-mini")

# List all configured providers - returns list of provider names that have valid API keys
# Example return: ["anthropic", "openai", "gemini"]
providers = factory.list_providers()
```

### Custom Tool Execution

Override `process_tool_calls` to implement custom tool handling:

```python
from llm_failover.clients import OpenAIClient

class CustomOpenAIClient(OpenAIClient):
    async def process_tool_calls(self, tool_calls):
        results = {}
        for tool_call in tool_calls:
            if tool_call["name"] == "get_weather":
                location = tool_call["arguments"]["location"]
                results[tool_call["id"]] = {
                    "success": True,
                    "result": f"Weather in {location}: Sunny, 72°F"
                }
        return results

# Use custom client
factory.model_priority[3]["client_class"] = CustomOpenAIClient
```

### Vision Support

```python
# Only get providers that support vision
client, model = factory.get_client(require_vision=True)

# Send image
response = await client.create_message(
    model=model,
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

## Examples

### Basic Usage Examples

**File:** [`examples/simple_usage.py`](examples/simple_usage.py)

Two complete examples using the ChatClient API:

1. **Basic Chat** - Simple non-streaming request with automatic failover
2. **Streaming Chat** - Real-time streaming with callback function

Run with:
```bash
python examples/simple_usage.py
```

### Framework Integration

This package is pure async Python and can be integrated with any async web framework. Here are some common patterns:

#### FastAPI Example

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_failover import ChatClient

app = FastAPI()
client = ChatClient()

@app.post("/chat")
async def chat(message: str):
    response = await client.chat(message)
    return {"response": response["content"], "provider": response["provider"]}

@app.post("/stream")
async def stream_chat(message: str):
    chunks = []
    
    async def collect_chunk(chunk: str):
        chunks.append(chunk)
    
    async def stream_generator():
        await client.stream(message, on_chunk=collect_chunk)
        for chunk in chunks:
            yield chunk
    
    return StreamingResponse(stream_generator(), media_type="text/plain")
```

#### Django Async View Example

```python
from django.http import JsonResponse
from llm_failover import ChatClient

client = ChatClient()

async def chat_view(request):
    message = request.POST.get("message")
    response = await client.chat(message)
    return JsonResponse({
        "response": response["content"],
        "provider": response["provider"]
    })
```

#### Standalone Script Example

```python
import asyncio
from llm_failover import ChatClient

async def main():
    client = ChatClient()
    
    # Simple chat
    response = await client.chat("What is Python?")
    print(response["content"])
    
    # Streaming
    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)
    
    await client.stream("Tell me a story", on_chunk=on_chunk)

if __name__ == "__main__":
    asyncio.run(main())
```

For more complete examples, see [`examples/simple_usage.py`](examples/simple_usage.py).

## How It Works

When you call `chat()` or `stream()`, the package:

1. **Tries the first provider** in your priority order (default: Gemini)
2. **If it fails**, captures the error context and tries the next provider
3. **Appends error context** to the system message on retry (helps the next provider avoid the same issue)
4. **Returns the response** from whichever provider succeeds
5. **Includes metadata** so you know which provider and model were used

All of this happens automatically - you just call `chat()` or `stream()`.

## Common Use Cases

### Simple Chatbot

```python
from llm_failover import ChatClient

client = ChatClient()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    response = await client.chat(user_input, keep_history=True)
    print(f"Bot: {response['content']}")
```

### Code Review Assistant

```python
client = ChatClient(
    system_message="You are an expert code reviewer.",
    provider_order=["anthropic", "openai"],  # Claude is great for code
    max_tokens=2000
)

code = """
def factorial(n):
    if n == 0: return 1
    return n * factorial(n-1)
"""

response = await client.chat(
    f"Review this code:\n\n{code}",
    temperature=0.3  # Lower temperature for focused analysis
)
print(response['content'])
```

### Streaming Content Generator

```python
client = ChatClient(
    system_message="You are a creative storyteller.",
    provider_order=["xai", "openai"]  # Grok is great for creative content
)

def on_chunk(chunk: str):
    print(chunk, end="", flush=True)

response = await client.stream(
    "Write a short story about a time-traveling cat",
    on_chunk=on_chunk,
    temperature=0.9  # Higher temperature for creativity
)
```

## Development

```bash
# Clone repository
git clone https://github.com/Nwafor6/llm-failover.git
cd llm-failover

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run example script
python examples/simple_usage.py

# Format code
black src/ examples/

# Type checking
mypy src/
```

## Troubleshooting

**"All providers failed"** - Check that:
1. At least one API key is set correctly
2. You have credits/quota with at least one provider
3. Your network can reach the provider APIs

**Import errors** - Make sure dependencies are installed:
```bash
pip install aiohttp anthropic openai
```

**Streaming not working** - Ensure callbacks are defined:
```python
def on_chunk(chunk: str):  # Can be sync or async
    print(chunk, end="")

response = await client.stream("test", on_chunk=on_chunk)
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional provider support
- Better error handling patterns
- Performance optimizations
- More examples

Please open an issue or PR on GitHub.

## License

MIT License - see LICENSE file for details.

## Links

- **GitHub**: [github.com/Nwafor6/llm-failover](https://github.com/Nwafor6/llm-failover)
- **PyPI**: [pypi.org/project/llm-failover](https://pypi.org/project/llm-failover)
- **Issues**: [github.com/Nwafor6/llm-failover/issues](https://github.com/Nwafor6/llm-failover/issues)

---

**Made with ❤️ for reliable LLM applications**
