"""
High-level ChatClient for LLM Failover.

Provides simple chat() and stream() methods with automatic failover across multiple providers.
"""
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from .factory import AIClientFactory

logger = logging.getLogger(__name__)


class ChatClient:
    """
    High-level chat interface with automatic failover across multiple LLM providers.

    This is the main class users should interact with. It handles all failover logic
    internally and provides simple chat() and stream() methods.

    Example:
        ```python
        from llm_failover import ChatClient

        # Initialize once
        client = ChatClient()

        # Non-streaming chat
        response = await client.chat("What is Python?")
        print(response["content"])

        # Streaming chat
        async def on_chunk(chunk):
            print(chunk, end="", flush=True)

        await client.stream("Tell me a story", on_chunk=on_chunk)
        ```
    """

    def __init__(
        self,
        provider_order: Optional[List[str]] = None,
        system_message: str = "You are a helpful AI assistant.",
        max_tokens: int = 4096,
        **factory_kwargs
    ):
        """
        Initialize ChatClient with failover configuration.

        Args:
            provider_order: List of provider names in priority order.
                          Default: ["gemini", "anthropic", "xai", "openai", "deepseek"]
            system_message: Default system message for all chats
            max_tokens: Default max tokens for responses
            **factory_kwargs: Additional arguments passed to AIClientFactory
                            (e.g., gemini_api_key, anthropic_model, etc.)
        """
        self.factory = AIClientFactory(**factory_kwargs)

        # Set provider priority order
        if provider_order:
            self.factory.reorder_clients(provider_order)

        self.default_system_message = system_message
        self.default_max_tokens = max_tokens
        self.conversation_history = []

    async def chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        keep_history: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming chat with automatic failover.

        Args:
            message: User message to send (ignored if messages provided)
            system_message: System message (uses default if not provided)
            messages: Full conversation history (overrides message parameter)
            max_tokens: Maximum tokens for response (uses default if not provided)
            keep_history: If True, maintains conversation history for multi-turn chats
            **kwargs: Additional parameters passed to the underlying AI client
                     (e.g., temperature, tools, etc.)

        Returns:
            Dict with:
                - content: The AI response text
                - provider: Which provider was used
                - model: Which model was used
                - attempt: Which attempt succeeded (1-indexed)

        Raises:
            Exception: If all providers fail
        """
        system_msg = system_message or self.default_system_message
        max_tok = max_tokens or self.default_max_tokens

        # Build messages list
        if messages:
            msg_list = messages
        elif keep_history:
            self.conversation_history.append({"role": "user", "content": message})
            msg_list = self.conversation_history
        else:
            msg_list = [{"role": "user", "content": message}]

        # Try each provider in priority order
        max_attempts = len(self.factory.model_priority)
        error_context = None

        for attempt in range(max_attempts):
            try:
                config = self.factory.model_priority[attempt]
                provider = config["provider"]
                model = config["model"]
                api_key = self.factory.api_keys.get(provider)

                if not api_key:
                    logger.warning(f"No API key for {provider}, skipping...")
                    continue

                logger.info(f"Attempt {attempt + 1}/{max_attempts}: Trying {provider} ({model})")

                # Append error context to system message if retrying
                current_system = system_msg
                if error_context:
                    current_system += f"\n\nNote: Previous attempt failed with: {error_context}"

                # Initialize client
                client = config["client_class"](api_key=api_key)

                # Make request
                response = await client.create_message(
                    model=model,
                    system_message=current_system,
                    messages=msg_list,
                    max_tokens=max_tok,
                    **kwargs
                )

                # Close client
                if hasattr(client, 'close'):
                    await client.close()

                # Update conversation history if keeping it
                if keep_history:
                    self.conversation_history.append({"role": "assistant", "content": response})

                # Success!
                logger.info(f"✓ Success with {provider}")
                return {
                    "content": response,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt + 1
                }

            except Exception as e:
                error_context = str(e)
                logger.warning(f"✗ {provider} failed: {error_context}")

                if attempt == max_attempts - 1:
                    raise Exception(
                        f"All {max_attempts} providers failed. Last error: {error_context}"
                    )

    async def stream(
        self,
        message: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, dict], None]] = None,
        keep_history: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Streaming chat with automatic failover.

        Args:
            message: User message to send
            system_message: System message (uses default if not provided)
            max_tokens: Maximum tokens for response (uses default if not provided)
            on_chunk: Callback function called for each text chunk: func(chunk: str)
            on_tool_start: Callback when tool execution starts: func(tool_name: str, args: dict)
            on_tool_result: Callback when tool execution completes: func(tool_name: str, result: dict)
            keep_history: If True, maintains conversation history
            **kwargs: Additional parameters passed to the underlying AI client

        Returns:
            Dict with:
                - content: The complete accumulated response text
                - provider: Which provider was used
                - model: Which model was used
                - attempt: Which attempt succeeded (1-indexed)

        Raises:
            Exception: If all providers fail
        """
        system_msg = system_message or self.default_system_message
        max_tok = max_tokens or self.default_max_tokens

        # Update conversation history if keeping it
        if keep_history:
            self.conversation_history.append({"role": "user", "content": message})

        # Try each provider in priority order
        max_attempts = len(self.factory.model_priority)
        error_context = None
        accumulated_text = ""

        for attempt in range(max_attempts):
            try:
                config = self.factory.model_priority[attempt]
                provider = config["provider"]
                model = config["model"]
                api_key = self.factory.api_keys.get(provider)

                if not api_key:
                    logger.warning(f"No API key for {provider}, skipping...")
                    continue

                logger.info(f"Attempt {attempt + 1}/{max_attempts}: Trying {provider} ({model})")

                # Append error context to system message if retrying
                current_system = system_msg
                if error_context:
                    current_system += f"\n\nNote: Previous attempt failed with: {error_context}"

                # Initialize client
                client = config["client_class"](api_key=api_key)

                # Internal callback to accumulate text
                accumulated_text = ""

                async def internal_on_chunk(chunk: str):
                    nonlocal accumulated_text
                    accumulated_text += chunk
                    if on_chunk:
                        if inspect.iscoroutinefunction(on_chunk):
                            await on_chunk(chunk)
                        else:
                            on_chunk(chunk)

                # Make streaming request
                await client.create_stream(
                    model=model,
                    system_message=current_system,
                    user_message=message,
                    max_tokens=max_tok,
                    on_chunk=internal_on_chunk,
                    on_tool_start=on_tool_start,
                    on_tool_result=on_tool_result,
                    conversations=self.conversation_history if keep_history else None,
                    **kwargs
                )

                # Close client
                if hasattr(client, 'close'):
                    await client.close()

                # Update conversation history if keeping it
                if keep_history:
                    self.conversation_history.append({"role": "assistant", "content": accumulated_text})

                # Success!
                logger.info(f"✓ Success with {provider}")
                return {
                    "content": accumulated_text,
                    "provider": provider,
                    "model": model,
                    "attempt": attempt + 1
                }

            except Exception as e:
                error_context = str(e)
                logger.warning(f"✗ {provider} failed: {error_context}")
                accumulated_text = ""  # Reset on failure

                if attempt == max_attempts - 1:
                    raise Exception(
                        f"All {max_attempts} providers failed. Last error: {error_context}"
                    )

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def set_provider_order(self, provider_order: List[str]):
        """
        Change provider priority order.

        Args:
            provider_order: List of provider names in desired priority order
        """
        self.factory.reorder_clients(provider_order)

