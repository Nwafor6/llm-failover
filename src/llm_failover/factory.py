"""AI Client Factory for managing multiple LLM providers with automatic failover."""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import AIAgentClient
from .clients import (
    AnthropicClient,
    DeepseekClient,
    GeminiClient,
    GrokClient,
    OpenAIClient,
)

logger = logging.getLogger(__name__)


class AIClientFactory:
    """
    Factory for creating and managing multiple AI client providers with automatic failover.
    
    Supports:
    - Multiple providers (Gemini, Anthropic, xAI/Grok, OpenAI, DeepSeek)
    - Automatic failover on initialization errors
    - Dynamic priority reordering
    - Vision support filtering
    - Customizable model selection per provider
    
    Example:
        ```python
        # Use default models
        factory = AIClientFactory()
        client, model = factory.get_client()
        
        # Specify custom models
        factory = AIClientFactory(
            gemini_model="gemini-2.0-flash-exp",
            anthropic_model="claude-3-5-sonnet-20241022",
            openai_model="gpt-4o-mini"
        )
        
        # Reorder priorities
        factory.reorder_clients(["anthropic", "openai"])
        ```
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        gemini_model: Optional[str] = None,
        anthropic_model: Optional[str] = None,
        grok_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        deepseek_model: Optional[str] = None,
    ):
        """
        Initialize the AI Client Factory.
        
        Args:
            gemini_api_key: Google GenAI API key (defaults to GOOGLE_GENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            grok_api_key: xAI/Grok API key (defaults to GROK_API_KEY env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            deepseek_api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            preferred_provider: Preferred provider name (defaults to PREFERRED_AI_PROVIDER env var or "gemini")
            gemini_model: Gemini model name (defaults to "gemini-2.0-flash-exp")
            anthropic_model: Anthropic model name (defaults to "claude-3-5-sonnet-20241022")
            grok_model: Grok model name (defaults to "grok-2-1212")
            openai_model: OpenAI model name (defaults to "gpt-4o")
            deepseek_model: DeepSeek model name (defaults to "deepseek-chat")
        """
        self.api_keys = {
            "gemini": gemini_api_key or os.getenv("GOOGLE_GENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "xai": grok_api_key or os.getenv("GROK_API_KEY"),
            "openai": openai_api_key or os.getenv("OPENAI_API_KEY"),
            "deepseek": deepseek_api_key or os.getenv("DEEPSEEK_API_KEY"),
        }
        
        self.model_priority = [
            {
                "provider": "gemini",
                "model": gemini_model or "gemini-3-flash-preview",
                "client_class": GeminiClient,
                "supports_vision": True,
            },
            {
                "provider": "anthropic",
                "model": anthropic_model or "claude-3-5-sonnet-20241022",
                "client_class": AnthropicClient,
                "supports_vision": True,
            },
            {
                "provider": "xai",
                "model": grok_model or "grok-4.3",
                "client_class": GrokClient,
                "supports_vision": True,
            },
            {
                "provider": "openai",
                "model": openai_model or "gpt-4o",
                "client_class": OpenAIClient,
                "supports_vision": True,
            },
            {
                "provider": "deepseek",
                "model": deepseek_model or "deepseek-chat",
                "client_class": DeepseekClient,
                "supports_vision": False,
            },
        ]
        
        self.preferred_provider = preferred_provider or os.getenv("PREFERRED_AI_PROVIDER", "gemini")

    def reorder_clients(self, provider_order: List[str]) -> None:
        """
        Reorder the clients based on the provided list of provider names.
        Only providers in the list will be kept, in the specified order.
        The first provider in the list becomes the new preferred provider.
        
        Args:
            provider_order: List of provider names in desired order (e.g., ["anthropic", "openai"])
        
        Example:
            ```python
            factory = AIClientFactory()
            factory.reorder_clients(["xai", "anthropic", "openai"])
            # Now xai is the preferred provider
            ```
        """
        new_priority = []
        for provider in provider_order:
            for config in self.model_priority:
                if config["provider"] == provider:
                    new_priority.append(config)
                    break
        
        if not new_priority:
            logger.warning("No matching providers found in reorder_clients")
            return
        
        self.model_priority = new_priority
        # Update preferred provider to the first in the new order
        self.preferred_provider = provider_order[0]
        logger.info(f"Reordered AI clients to: {', '.join(provider_order)}")

    def get_client(
        self,
        fallback: bool = False,
        require_vision: bool = False,
        **client_kwargs
    ) -> Tuple[AIAgentClient, str]:
        """
        Get an AI client based on priority or fallback logic.
        
        Args:
            fallback: If True, skip preferred provider check and use first available
            require_vision: If True, only return providers with vision support
            **client_kwargs: Additional keyword arguments to pass to client initialization
        
        Returns:
            Tuple of (client_instance, model_name)
        
        Raises:
            ValueError: If no providers are available
        
        Example:
            ```python
            factory = AIClientFactory()
            
            # Get preferred client
            client, model = factory.get_client()
            
            # Get first available with vision
            client, model = factory.get_client(require_vision=True)
            
            # Force fallback mode
            client, model = factory.get_client(fallback=True)
            ```
        """
        candidates = self.model_priority
        if require_vision:
            candidates = [c for c in self.model_priority if c["supports_vision"]]

        if not candidates:
            raise ValueError(
                "No AI providers available" + (" with vision support" if require_vision else "")
            )

        # Try preferred provider first if not in fallback mode
        if not fallback:
            for config in candidates:
                if config["provider"] == self.preferred_provider:
                    try:
                        api_key = self.api_keys.get(config["provider"])
                        client = config["client_class"](api_key=api_key, **client_kwargs)
                        logger.info(f"Initialized preferred provider: {config['provider']}")
                        return client, config["model"]
                    except Exception as e:
                        logger.warning(
                            f"Failed to initialize preferred provider {config['provider']}: {str(e)}"
                        )
                        break

        # Try all candidates in order
        for config in candidates:
            try:
                api_key = self.api_keys.get(config["provider"])
                client = config["client_class"](api_key=api_key, **client_kwargs)
                logger.info(f"Initialized AI client: {config['provider']}")
                return client, config["model"]
            except Exception as e:
                logger.warning(f"Failed to initialize {config['provider']} client: {str(e)}")
                continue

        raise ValueError(
            "All AI providers failed to initialize" + (" with vision support" if require_vision else "")
        )

    def update_model(self, provider: str, model: str) -> None:
        """
        Update the model for a specific provider.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name to use (e.g., "gpt-4", "claude-3-opus-20240229")
        
        Example:
            ```python
            factory = AIClientFactory()
            factory.update_model("openai", "gpt-4-turbo")
            factory.update_model("anthropic", "claude-3-opus-20240229")
            ```
        """
        for config in self.model_priority:
            if config["provider"] == provider:
                old_model = config["model"]
                config["model"] = model
                logger.info(f"Updated {provider} model from {old_model} to {model}")
                return
        
        logger.warning(f"Provider {provider} not found in model_priority")

    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List all configured providers with their settings.
        
        Returns:
            List of provider configurations
        
        Example:
            ```python
            factory = AIClientFactory()
            providers = factory.list_providers()
            for p in providers:
                print(f"{p['provider']}: {p['model']} (vision: {p['supports_vision']})")
            ```
        """
        return [
            {
                "provider": config["provider"],
                "model": config["model"],
                "supports_vision": config["supports_vision"],
                "has_api_key": bool(self.api_keys.get(config["provider"])),
            }
            for config in self.model_priority
        ]
