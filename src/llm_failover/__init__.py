"""
LLM Failover - Multi-provider AI client orchestration with automatic failover.

Provides a factory pattern for managing multiple LLM providers (OpenAI, Anthropic,
Gemini, Grok, DeepSeek) with priority ordering and automatic fallback when providers fail.
"""

from .base import AIAgentClient
from .factory import AIClientFactory

__version__ = "0.1.0"
__all__ = ["AIClientFactory", "AIAgentClient"]
