"""OpenAI client implementation."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import openai

from ..base import AIAgentClient

logger = logging.getLogger(__name__)


class OpenAIClient(AIAgentClient):
    """OpenAI API client with streaming and tool calling support."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)
        """
        super().__init__()
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate AI response using OpenAI."""
        response = await self.create_stream(
            model=kwargs.get("model", "gpt-4o"),
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

            max_tool_rounds = kwargs.get("max_tool_rounds", 5)
            current_round = 0
            full_response = ""
            
            while current_round < max_tool_rounds:
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_completion_tokens=max_tokens,
                    stream=True,
                    temperature=kwargs.get("temperature", 0.0),
                    tools=tools if tools else None,
                    tool_choice=kwargs.get("tool_choice", "auto"),
                )
                
                chunk_response = ""
                tool_calls = []
                
                logger.info(f"Streaming response started (OpenAI) - Round {current_round + 1}")
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                        
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        content_text = delta.content
                        chunk_response += content_text
                        full_response += content_text
                        if on_chunk:
                            if asyncio.iscoroutinefunction(on_chunk):
                                await on_chunk(content_text)
                            else:
                                on_chunk(content_text)
                    
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            if tc_delta.index is not None:
                                while len(tool_calls) <= tc_delta.index:
                                    tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                target = tool_calls[tc_delta.index]
                                if tc_delta.id:
                                    target["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        target["function"]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        target["function"]["arguments"] += tc_delta.function.arguments
                
                if not tool_calls:
                    return full_response

                # Notify tool execution start
                if on_tool_start:
                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        if asyncio.iscoroutinefunction(on_tool_start):
                            await on_tool_start(tool_name)
                        else:
                            on_tool_start(tool_name)

                # Prepare assistant message for context history
                openai_messages.append({
                    "role": "assistant",
                    "content": chunk_response,
                    "tool_calls": tool_calls
                })
                
                # Execute tools via base class
                tool_results = await self.process_tool_calls(tool_calls, **kwargs)
                
                for tc in tool_calls:
                    tc_id = tc["id"]
                    result = tool_results.get(tc_id, {"success": False, "message": "No result"})
                    
                    # Notify tool execution result
                    if on_tool_result:
                        if asyncio.iscoroutinefunction(on_tool_result):
                            await on_tool_result(tc["function"]["name"], result)
                        else:
                            on_tool_result(tc["function"]["name"], result)
                    
                    # Logic to break the loop if a tool specifies 'end_loop'
                    if isinstance(result, dict) and result.get("end_loop"):
                        max_tool_rounds = 0

                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(result)
                    })
                
                current_round += 1
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in OpenAI create_stream: {str(e)}")
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
                    model=model,
                    messages=openai_messages,
                    max_completion_tokens=max_tokens,
                    stream=False,
                    temperature=kwargs.get("temperature", 0.3),
                    tools=tools if tools else None,
                )

                message = response.choices[0].message
                if not message.tool_calls:
                    return message.content or ""

                # Build tool calls for assistant history
                formatted_tc = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]

                openai_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": formatted_tc
                })

                # Process results
                results = await self.process_tool_calls(formatted_tc, **kwargs)

                for tc in formatted_tc:
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
            logger.error(f"Error in OpenAI create_message: {str(e)}")
            raise

    async def organize_message(self, messages: list) -> list:
        """Standardizes input messages into OpenAI format."""
        formatted_messages = []
        for msg in messages:
            content = []
            msg_content = msg.get("content", [])
            
            # Handle both raw strings and list of blocks
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
        """Close the OpenAI client."""
        await self.client.close()
