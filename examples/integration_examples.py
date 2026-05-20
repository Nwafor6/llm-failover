"""
Integration examples for llm-failover package.
Shows how to use the package in different frameworks.
"""

# ============================================================================
# Example 1: FastAPI Integration
# ============================================================================

"""
# File: fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_failover import AIClientFactory
import asyncio

app = FastAPI()

# Initialize factory once at startup
factory = AIClientFactory(preferred_provider="anthropic")

class ChatRequest(BaseModel):
    message: str
    provider: str | None = None
    max_tokens: int = 500

class ChatResponse(BaseModel):
    content: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Optionally reorder if specific provider requested
        if request.provider:
            factory.reorder_clients([request.provider])
        
        # Get client with automatic failover
        client, model = factory.get_client()
        
        # Make request
        response = await client.create_message(
            messages=[{"role": "user", "content": request.message}],
            max_tokens=request.max_tokens
        )
        
        await client.close()
        
        return ChatResponse(
            content=response.get("content", ""),
            model=model
        )
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def list_providers():
    return {"providers": factory.list_providers()}

# Run with: uvicorn fastapi_app:app --reload
"""


# ============================================================================
# Example 2: Django Integration (Views)
# ============================================================================

"""
# File: views.py (Django)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from llm_failover import AIClientFactory
import json
import asyncio

# Initialize factory (could also be done in apps.py ready() method)
factory = AIClientFactory()

@csrf_exempt
def chat_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        data = json.loads(request.body)
        message = data.get("message", "")
        
        # Run async function in sync context
        response = asyncio.run(_chat_async(message))
        
        return JsonResponse({
            "content": response["content"],
            "model": response["model"]
        })
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

async def _chat_async(message: str):
    client, model = factory.get_client()
    
    response = await client.create_message(
        messages=[{"role": "user", "content": message}],
        max_tokens=500
    )
    
    await client.close()
    
    return {
        "content": response.get("content", ""),
        "model": model
    }

def providers_view(request):
    providers = factory.list_providers()
    return JsonResponse({"providers": providers})
"""


# ============================================================================
# Example 3: Update BildBridge Agents
# ============================================================================

"""
# File: agents/job_creation_agent/main.py (UPDATED)

# OLD:
# from agents.clients.factory import AIClientFactory
# from config.lib import CONFIG

# NEW:
from llm_failover import AIClientFactory

class JobCreationAgent:
    def __init__(self):
        # Initialize factory with production models
        self.factory = AIClientFactory()
        
        # Update to production-specific model versions
        self.factory.update_model("gemini", "gemini-3.1-pro-preview")
        self.factory.update_model("anthropic", "claude-haiku-4-5-20251001")
        self.factory.update_model("xai", "grok-4-1-fast-reasoning")
        self.factory.update_model("openai", "gpt-5.2-2025-12-11")
    
    async def generate_job_description(self, company_id: int, role: str):
        # Get client with automatic failover
        client, model = self.factory.get_client()
        
        # Your existing logic here...
        response = await client.create_stream(
            messages=[...],
            tools=[...],
            on_chunk=self.handle_chunk,
            on_tool_start=self.handle_tool_start,
            on_tool_result=self.handle_tool_result
        )
        
        await client.close()
        return response
"""
