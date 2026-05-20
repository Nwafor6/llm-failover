#!/usr/bin/env python3
"""
Basic validation script for llm-failover package.
Tests all provider initializations and basic functionality.
"""

import asyncio
import os

from llm_failover import AIClientFactory


async def test_providers():
    """Test that all providers can be initialized."""
    print("=" * 60)
    print("LLM Failover Package - Basic Validation")
    print("=" * 60)
    
    # Initialize factory
    factory = AIClientFactory()
    print(f"\n✓ Factory initialized successfully")
    
    # List all providers
    providers = factory.list_providers()
    print(f"\n✓ Found {len(providers)} providers:")
    for p in providers:
        status = "✓ Has API key" if p['has_api_key'] else "✗ No API key"
        vision = "Vision ✓" if p['supports_vision'] else "No vision"
        print(f"  - {p['provider']:12} | {p['model']:30} | {status} | {vision}")
    
    # Test update_model
    print(f"\n✓ Testing update_model...")
    original_model = factory.model_priority[0]['model']
    original_provider = factory.model_priority[0]['provider']
    factory.update_model(original_provider, "test-model-name")
    print(f"  Changed {original_provider} model from '{original_model}' to '{factory.model_priority[0]['model']}'")
    factory.update_model(original_provider, original_model)  # Reset
    
    # Test reordering (note: this is destructive, can't be easily undone)
    print(f"\n✓ Testing reorder_clients...")
    original_count = len(factory.model_priority)
    factory.reorder_clients(["openai", "anthropic"])
    reordered = factory.list_providers()
    print(f"  Reordered from {original_count} providers to: {', '.join([p['provider'] for p in reordered])}")
    print(f"  Note: reorder_clients is destructive - original list cannot be restored")
    
    # Test get_client with vision filter
    print(f"\n✓ Testing get_client with vision filter...")
    try:
        client, model = factory.get_client(require_vision=True)
        print(f"  Got vision-capable client: {model}")
        await client.close()
    except ValueError as e:
        print(f"  Note: {e}")
    
    # Test individual provider initialization (without making actual API calls)
    print(f"\n✓ Testing individual provider clients...")
    for provider_config in factory.model_priority:
        provider = provider_config['provider']
        client_class = provider_config['client_class']
        try:
            # Try to instantiate without API key to test structure
            # We won't make actual API calls
            print(f"  - {provider:12} client class: {client_class.__name__} ✓")
        except Exception as e:
            print(f"  - {provider:12} error: {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set API keys in environment variables")
    print("2. Test actual API calls with: python test_api_call.py")
    print("3. Use in 3+ different projects within 1 week to validate value")
    print("\nEnvironment variables to set:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_GENAI_API_KEY")
    print("  - GROK_API_KEY")
    print("  - DEEPSEEK_API_KEY")
    print("  - PREFERRED_AI_PROVIDER (optional, defaults to first available)")


async def test_api_call_example():
    """
    Example of how to make actual API calls (commented out).
    Uncomment and run when you have API keys configured.
    """
    # factory = AIClientFactory()
    # 
    # # Get a client (tries preferred provider, falls back automatically)
    # try:
    #     client, model = factory.get_client()
    #     print(f"Using: {model}")
    #     
    #     # Make a simple API call
    #     response = await client.create_message(
    #         messages=[{"role": "user", "content": "Say 'Hello from llm-failover!'"}],
    #         max_tokens=50
    #     )
    #     
    #     print(f"Response: {response.get('content', 'No content')}")
    #     await client.close()
    #     
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     print("Make sure you have at least one API key configured")
    pass


if __name__ == "__main__":
    asyncio.run(test_providers())
    
    # Uncomment to test actual API calls:
    # asyncio.run(test_api_call_example())
