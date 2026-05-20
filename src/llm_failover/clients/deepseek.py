"""DeepSeek client implementation."""
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import openai

from ..base import AIAgentClient

logger = logging.getLogger(__name__)


class DeepseekClient(AIAgentClient):
    """DeepSeek API client."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (optional, will use DEEPSEEK_API_KEY env var if not provided)
        """
        super().__init__()
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate AI response using DeepSeek."""
        response = await self.create_stream(
            model=kwargs.get("model", "deepseek-chat"),
            system_message=kwargs.get("system_message", "You are a helpful assistant."),
            user_message=prompt,
            max_tokens=kwargs.get("max_tokens", 1000),
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
        """Stream responses with support for multi-turn tool calling."""
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

            max_tool_rounds = kwargs.get("max_tool_rounds", 10)
            current_round = 0
            full_response = ""
            
            while current_round < max_tool_rounds:
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_tokens=kwargs.get("max_tokens", 8192),
                    stream=True,
                    tools=tools if tools else [],
                )
                
                chunk_response = ""
                tool_calls = []
                current_tool_call = None
                
                logger.info(f"Streaming response started (DeepSeek) - Round {current_round + 1}")
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    
                    # Handle regular content
                    if delta.content:
                        content_text = delta.content
                        chunk_response += content_text
                        full_response += content_text
                        if on_chunk:
                            if asyncio.iscoroutinefunction(on_chunk):
                                await on_chunk(content_text)
                            else:
                                on_chunk(content_text)
                    
                    # Handle tool calls in streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            # Handle case when index is None
                            if len(tool_calls) == 0:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                                current_tool_call = tool_calls[0]
                            elif tool_call_delta.index is not None:
                                while len(tool_calls) <= tool_call_delta.index:
                                    tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                current_tool_call = tool_calls[tool_call_delta.index]
                                
                            if tool_call_delta.id:
                                current_tool_call["id"] = tool_call_delta.id
                            
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    current_tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments
                
                # Check if we have tool calls to process
                if tool_calls:
                    logger.info(f"Tool calls detected in stream: {len(tool_calls)} calls")
                    
                    # Notify tool execution start
                    if on_tool_start:
                        for tc in tool_calls:
                            tool_name = tc["function"]["name"]
                            if asyncio.iscoroutinefunction(on_tool_start):
                                await on_tool_start(tool_name)
                            else:
                                on_tool_start(tool_name)
                    
                    # Add assistant message with tool calls to conversation
                    openai_messages.append({
                        "role": "assistant",
                        "content": chunk_response,
                        "tool_calls": tool_calls
                    })
                    
                    # Convert to our standard format
                    standardized_tool_calls = [
                        {
                            "id": tc["id"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in tool_calls
                    ]
                    
                    # Process tool calls
                    tool_results = await self.process_tool_calls(standardized_tool_calls, **kwargs)
                    
                    # Add tool results to conversation
                    for tool_call in tool_calls:
                        tool_call_id = tool_call["id"]
                        tool_name = tool_call["function"]["name"]
                        result = tool_results.get(tool_call_id, {"success": False, "message": "No result"})

                        if on_tool_result:
                            if asyncio.iscoroutinefunction(on_tool_result):
                                await on_tool_result(tool_name, result)
                            else:
                                on_tool_result(tool_name, result)

                        logger.info(f"Tool end_loop flag: {result.get('end_loop')}")
                        if result.get("end_loop", False):
                            max_tool_rounds = 0

                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result)
                        })
                    
                    current_round += 1
                else:
                    # No tool calls, we're done
                    return full_response or "No response received."
            
            # If we've exhausted tool rounds, return the accumulated response
            return full_response or "Maximum tool call rounds reached."
            
        except openai.APIError as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek create_stream: {str(e)}")
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
            max_tool_rounds = kwargs.get("max_tool_rounds", 10)
            current_round = 0

            while current_round < max_tool_rounds:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    stream=False,
                    tools=tools if tools else [],
                )

                message = response.choices[0].message
                if not message.tool_calls:
                    return message.content or ""

                openai_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }
                        for tc in message.tool_calls
                    ]
                })

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

            return message.content or ""

        except Exception as e:
            logger.error(f"Error in DeepSeek create_message: {str(e)}")
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
        """Close the DeepSeek client."""
        await self.client.close()
