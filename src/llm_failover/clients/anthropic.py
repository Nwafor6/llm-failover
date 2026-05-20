"""Anthropic client implementation."""
import asyncio
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import anthropic

from ..base import AIAgentClient

logger = logging.getLogger(__name__)


def safe_json_serializer(obj):
    """Basic JSON serializer for common non-serializable types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, UUID):
        return str(obj)
    return str(obj)


class AnthropicClient(AIAgentClient):
    """Anthropic API client with streaming and tool calling support."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (optional, will use ANTHROPIC_API_KEY env var if not provided)
        """
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)

    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate AI response using Anthropic."""
        response = await self.create_message(
            model=kwargs.get("model", "claude-3-sonnet-20240229"),
            system_message=kwargs.get("system_message", "You are a helpful assistant."),
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=kwargs.get("max_tokens", 1000),
            **kwargs
        )
        return {"response": response}

    @staticmethod
    def _normalize_tool_choice(tool_choice):
        """Convert OpenAI-style tool_choice strings to Anthropic dict format."""
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": "auto"}

    async def create_stream(
        self,
        model: str,
        system_message: str,
        user_message: str = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Stream responses with support for multi-turn tool calling."""
        try:
            tools = kwargs.get("tools", [])
            conversations = kwargs.get("conversations", [])
            
            on_chunk = kwargs.get("on_chunk")
            on_tool_start = kwargs.get("on_tool_start")
            on_tool_result = kwargs.get("on_tool_result")
            
            if conversations:
                anthropic_messages = await self.organize_message(conversations)
            else:
                anthropic_messages = [{"role": "user", "content": [{"type": "text", "text": user_message}]}]

            max_tool_rounds = kwargs.get("max_tool_rounds", 5)
            current_round = 0
            full_response = ""
            
            while current_round < max_tool_rounds:
                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_message,
                        messages=anthropic_messages,
                        tools=tools,
                        tool_choice=self._normalize_tool_choice(kwargs.get("tool_choice")),
                        stream=True,
                        temperature=kwargs.get("temperature", 0.3)
                    ),
                )
                
                chunk_response = ""
                tool_calls = []
                current_tool_call = None
                current_tool_input = ""
                
                for chunk in stream:
                    # Text handling
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        content_text = chunk.delta.text
                        chunk_response += content_text
                        full_response += content_text
                        if on_chunk:
                            if asyncio.iscoroutinefunction(on_chunk):
                                await on_chunk(content_text)
                            else:
                                on_chunk(content_text)
                    
                    # Tool call start
                    elif hasattr(chunk, "content_block") and chunk.type == "content_block_start":
                        if chunk.content_block.type == "tool_use":
                            current_tool_call = {
                                "id": chunk.content_block.id,
                                "name": chunk.content_block.name,
                                "input": {}
                            }
                    
                    # Tool input delta
                    elif chunk.type == "content_block_delta" and hasattr(chunk.delta, "partial_json"):
                        if current_tool_call:
                            current_tool_input += chunk.delta.partial_json

                    # Tool call stop
                    elif chunk.type == "content_block_stop" and current_tool_call:
                        try:
                            current_tool_call["input"] = json.loads(current_tool_input) if current_tool_input else {}
                            tool_calls.append(current_tool_call)
                        except json.JSONDecodeError:
                            current_tool_call["input"] = {}
                            tool_calls.append(current_tool_call)
                        current_tool_call = None
                        current_tool_input = ""
                
                if not tool_calls:
                    return full_response or "No response received."

                # Process Tool Calls
                assistant_content = []
                if chunk_response:
                    assistant_content.append({"type": "text", "text": chunk_response})
                
                for tc in tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"]
                    })
                    if on_tool_start:
                        if asyncio.iscoroutinefunction(on_tool_start):
                            await on_tool_start(tc["name"])
                        else:
                            on_tool_start(tc["name"])
                
                anthropic_messages.append({"role": "assistant", "content": assistant_content})
                
                # Execute tools - convert to standard format for process_tool_calls
                standardized_calls = [
                    {
                        "id": tc["id"],
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}
                    }
                    for tc in tool_calls
                ]
                
                tool_results = await self.process_tool_calls(standardized_calls, **kwargs)
                
                tool_result_content = []
                for tc in tool_calls:
                    # Map back from standardized ID to Anthropic format
                    res = tool_results.get(tc["id"], {"success": False, "message": "No result"})
                    
                    if on_tool_result:
                        if asyncio.iscoroutinefunction(on_tool_result):
                            await on_tool_result(tc["name"], res)
                        else:
                            on_tool_result(tc["name"], res)

                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(res, default=safe_json_serializer)
                    })
                
                anthropic_messages.append({"role": "user", "content": tool_result_content})
                current_round += 1
                
            return full_response

        except Exception as e:
            logger.error(f"Error in Anthropic create_stream: {str(e)}")
            raise

    async def create_message(
        self,
        model: str,
        system_message: str,
        messages: list,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Non-streaming message creation with tool support."""
        try:
            anthropic_messages = await self.organize_message(messages)
            tools = kwargs.get("tools", [])
            max_tool_rounds = kwargs.get("max_tool_rounds", 5)
            current_round = 0

            while current_round < max_tool_rounds:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_message,
                        messages=anthropic_messages,
                        tools=tools if tools else [],
                    ),
                )

                tool_calls = []
                text_content = ""

                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
                    elif block.type == "text":
                        text_content += block.text

                if not tool_calls:
                    return text_content

                # Prepare assistant message for history
                assistant_blocks = []
                for block in response.content:
                    if block.type == "text":
                        assistant_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_blocks.append({
                            "type": "tool_use", 
                            "id": block.id, 
                            "name": block.name, 
                            "input": block.input
                        })
                
                anthropic_messages.append({"role": "assistant", "content": assistant_blocks})

                # Convert to standard format for processing
                standardized_calls = [
                    {
                        "id": tc["id"],
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}
                    }
                    for tc in tool_calls
                ]

                results = await self.process_tool_calls(standardized_calls, **kwargs)
                
                tool_result_content = []
                for tc in tool_calls:
                    res = results.get(tc["id"], {"success": False})
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(res, default=safe_json_serializer)
                    })

                    if res.get("end_loop", False):
                        max_tool_rounds = 0

                anthropic_messages.append({"role": "user", "content": tool_result_content})
                current_round += 1

            return text_content

        except Exception as e:
            logger.error(f"Error in Anthropic create_message: {str(e)}")
            raise

    async def organize_message(self, messages: list) -> list:
        """Standardizes input message formats into Anthropic's role/content blocks."""
        anthropic_messages = []
        for msg in messages:
            formatted_content = []
            content_input = msg.get("content", [])
            
            if isinstance(content_input, str):
                formatted_content = [{"type": "text", "text": content_input}]
            else:
                for block in content_input:
                    if block["type"] == "text":
                        formatted_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        formatted_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block["media_type"],
                                "data": block["base64"],
                            },
                        })
            
            anthropic_messages.append({"role": msg["role"], "content": formatted_content})
        
        return anthropic_messages
