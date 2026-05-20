"""Gemini client implementation using OpenAI-compatible API."""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import openai

from ..base import AIAgentClient

logger = logging.getLogger(__name__)


class GeminiClient(AIAgentClient):
    """Gemini API client via OpenAI-compatible endpoint."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google GenAI API key (optional, will use GOOGLE_GENAI_API_KEY env var if not provided)
        """
        super().__init__()
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("GOOGLE_GENAI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate AI response using Gemini."""
        response = await self.create_stream(
            model=kwargs.get("model", "gemini-3-flash-preview"),
            system_message=kwargs.get("system_message", "You are a helpful assistant."),
            user_message=prompt,
            max_tokens=kwargs.get("max_tokens", 2048),
            **kwargs
        )
        return {"response": response}

    async def create_stream(
        self,
        model: str,
        system_message: str,
        user_message: str,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Stream responses with multi-turn tool calling support and exponential backoff."""
        try:
            openai_messages = [{"role": "system", "content": system_message}]
            
            tools = kwargs.get("tools", [])
            conversations = kwargs.get("conversations", [])
            on_chunk = kwargs.get("on_chunk")
            on_tool_start = kwargs.get("on_tool_start")
            on_tool_result = kwargs.get("on_tool_result")

            if conversations:
                openai_messages.extend(await self.organize_message(conversations))
            else:
                openai_messages.append({"role": "user", "content": user_message})

            max_tool_rounds = kwargs.get("max_tool_rounds", 5)
            current_round = 0
            full_response = ""
            max_retries = kwargs.get("max_retries", 3)
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    while current_round < max_tool_rounds:
                        stream = await self.client.chat.completions.create(
                            model=self._normalize_model(model),
                            messages=openai_messages,
                            max_tokens=max_tokens,
                            stream=True,
                            temperature=kwargs.get("temperature", 0.0),
                            tools=tools if tools else None,
                        )
                        
                        chunk_response = ""
                        tool_calls = []
                        current_tool_call = None
                        
                        async for chunk in stream:
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta

                            # Handle Content
                            if delta.content:
                                content_text = delta.content
                                chunk_response += content_text
                                full_response += content_text
                                if on_chunk:
                                    if asyncio.iscoroutinefunction(on_chunk):
                                        await on_chunk(content_text)
                                    else:
                                        on_chunk(content_text)
                            
                            # Handle Tool Call Deltas
                            if delta.tool_calls:
                                for tc_delta in delta.tool_calls:
                                    if tc_delta.index is not None:
                                        while len(tool_calls) <= tc_delta.index:
                                            tool_calls.append({
                                                "id": "",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""},
                                                "extra_content": None
                                            })
                                        current_tool_call = tool_calls[tc_delta.index]
                                    elif not current_tool_call and tool_calls:
                                        current_tool_call = tool_calls[-1]
                                    elif not current_tool_call:
                                        current_tool_call = {
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                            "extra_content": None
                                        }
                                        tool_calls.append(current_tool_call)

                                    if current_tool_call:
                                        if tc_delta.id:
                                            current_tool_call["id"] = tc_delta.id
                                        if tc_delta.function:
                                            if tc_delta.function.name:
                                                current_tool_call["function"]["name"] = tc_delta.function.name
                                            if tc_delta.function.arguments:
                                                current_tool_call["function"]["arguments"] += tc_delta.function.arguments
                                        if hasattr(tc_delta, 'extra_content') and tc_delta.extra_content:
                                            current_tool_call["extra_content"] = tc_delta.extra_content
                
                        if not tool_calls:
                            return full_response

                        # Notify tool execution start
                        for tc in tool_calls:
                            tool_name = tc["function"]["name"]
                            if on_tool_start:
                                if asyncio.iscoroutinefunction(on_tool_start):
                                    await on_tool_start(tool_name)
                                else:
                                    on_tool_start(tool_name)

                        # Process Tool Calls
                        assistant_msg = {"role": "assistant", "content": chunk_response, "tool_calls": []}
                        standardized_calls = []

                        for tc in tool_calls:
                            # Build the assistant history message
                            formatted_tc = {
                                "id": tc["id"],
                                "type": tc["type"],
                                "function": tc["function"]
                            }
                            if tc.get("extra_content"):
                                formatted_tc["extra_content"] = tc["extra_content"]
                            assistant_msg["tool_calls"].append(formatted_tc)

                            # Standardize for the execution logic
                            standardized_calls.append({
                                "id": tc["id"],
                                "function": tc["function"]
                            })

                        openai_messages.append(assistant_msg)

                        # Execute via base class
                        tool_results = await self.process_tool_calls(standardized_calls, **kwargs)
                        
                        for tc in standardized_calls:
                            tc_id = tc["id"]
                            result = tool_results.get(tc_id, {"success": False, "message": "No result"})
                            
                            if on_tool_result:
                                if asyncio.iscoroutinefunction(on_tool_result):
                                    await on_tool_result(tc["function"]["name"], result)
                                else:
                                    on_tool_result(tc["function"]["name"], result)
                            
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(result)
                            })
                        
                        current_round += 1
                    
                    return full_response

                except openai.APIError as e:
                    if e.response and e.response.status_code in [408, 429, 500, 502, 503, 504]:
                        logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        raise

            raise ValueError("Max retries exceeded for Gemini client.")

        except Exception as e:
            logger.error(f"Unexpected error in Gemini create_stream: {str(e)}")
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
            openai_messages = [{"role": "system", "content": system_message}]
            openai_messages.extend(await self.organize_message(messages))
            
            tools = kwargs.get("tools", [])
            max_tool_rounds = kwargs.get("max_tool_rounds", 5)
            current_round = 0

            while current_round < max_tool_rounds:
                response = await self.client.chat.completions.create(
                    model=self._normalize_model(model),
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    stream=False,
                    temperature=kwargs.get("temperature", 0.3),
                    tools=tools if tools else None,
                )

                message = response.choices[0].message

                if not message.tool_calls:
                    return message.content or ""

                # Add assistant call to history
                history_tool_calls = []
                for tc in message.tool_calls:
                    if hasattr(tc, "model_dump"):
                        history_tool_calls.append(tc.model_dump(exclude_none=True))
                    else:
                        history_tool_calls.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                openai_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": history_tool_calls,
                })

                # Standardize and execute
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]

                results = await self.process_tool_calls(tool_calls, **kwargs)

                for tc in tool_calls:
                    res = results.get(tc["id"], {"success": False})
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(res)
                    })

                    if res.get("end_loop", False):
                        max_tool_rounds = 0

                current_round += 1

            return message.content or "Maximum tool call rounds reached."

        except Exception as e:
            logger.error(f"Error in Gemini create_message: {str(e)}")
            raise

    async def organize_message(self, messages: list) -> list:
        """Standardizes input messages into OpenAI format."""
        formatted_messages = []
        for msg in messages:
            content = []
            msg_content = msg.get("content", [])
            
            if isinstance(msg_content, str):
                content = msg_content
            else:
                for block in msg_content:
                    if block["type"] == "text":
                        content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block['media_type']};base64,{block['base64']}",
                                "detail": "high",
                            },
                        })
            formatted_messages.append({"role": msg["role"], "content": content})
        return formatted_messages

    async def close(self):
        """Close the Gemini client."""
        await self.client.close()

    def _normalize_model(self, model: str) -> str:
        """Normalize model name for Gemini OpenAI-compatible endpoint."""
        if not model.startswith("models/"):
            # Only force -latest on older 1.5 models; new models like gemini-3-* use as-is
            if model.startswith("gemini-1.5-flash") and "latest" not in model and not model.endswith(("-001", "-002")):
                model = "gemini-1.5-flash-latest"
            return f"models/{model}" if not model.startswith("gemini") else model
        return model
