"""Base AI Agent Client interface."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AIAgentClient(ABC):
    """
    Base AI Agent Client providing common interfaces for all LLM providers.
    
    Subclasses must implement:
    - generate_response: Simple prompt-response
    - create_stream: Streaming response with tool support
    - create_message: Non-streaming response
    """

    def __init__(self):
        """Initialize the client. Override to add provider-specific setup."""
        pass

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate AI response from a simple prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing the response
        """
        pass

    @abstractmethod
    async def create_stream(
        self,
        model: str,
        system_message: str,
        user_message: str,
        max_tokens: int,
        **kwargs
    ) -> str:
        """
        Create streaming response with optional tool calling.
        
        Args:
            model: Model identifier
            system_message: System prompt
            user_message: User message (can be None if conversations provided)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters including:
                - conversations: List of conversation history
                - tools: List of tool definitions
                - on_chunk: Callback for each text chunk
                - on_tool_start: Callback when tool execution starts
                - on_tool_result: Callback when tool execution completes
                
        Returns:
            The complete response text
        """
        pass

    @abstractmethod
    async def create_message(
        self,
        model: str,
        system_message: str,
        messages: list,
        max_tokens: int,
        **kwargs
    ) -> str:
        """
        Create non-streaming response.
        
        Args:
            model: Model identifier
            system_message: System prompt
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The response text
        """
        pass

    async def organize_message(self, messages: list) -> list:
        """
        Standardize input messages into provider format.
        Override this for provider-specific message formatting.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted messages for the provider
        """
        return messages

    async def process_tool_calls(
        self,
        tool_calls: List[Dict],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process tool calls from the LLM response.
        
        This is a stub implementation that returns success for all tools.
        Override this method to implement actual tool execution logic.
        
        Args:
            tool_calls: List of tool calls from the LLM
            **kwargs: Additional context for tool execution
            
        Returns:
            Dict mapping tool call IDs to results
        """
        results = {}
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", tool_call.get("name", "unknown"))
            tool_name = tool_call.get("function", {}).get("name") or tool_call.get("name")
            
            logger.info(f"Stub: Tool call {tool_name} (override process_tool_calls to implement)")
            
            results[tool_call_id] = {
                "success": True,
                "message": f"Tool {tool_name} executed (stub implementation)",
                "data": {}
            }
        
        return results

    async def close(self):
        """Close the client and cleanup resources. Override if needed."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
