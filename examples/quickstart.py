#!/usr/bin/env python3
"""
Quick start example for llm-failover package.
Run this after setting up your API keys to test actual LLM calls.

Setup:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    # Or set others: GOOGLE_GENAI_API_KEY, GROK_API_KEY, DEEPSEEK_API_KEY
    
    python examples/quickstart.py
"""

import asyncio

from llm_failover import AIClientFactory


async def simple_chat():
    """Basic chat example with automatic failover."""
    print("=" * 60)
    print("Example 1: Simple Chat")
    print("=" * 60)
    
    factory = AIClientFactory()
    
    try:
        # Get first available client (tries in priority order)
        client, model = factory.get_client()
        print(f"✓ Using: {model}\n")
        
        # Make a simple chat request
        response = await client.create_message(
            messages=[
                {"role": "user", "content": "Say hello and tell me which AI model you are in one sentence."}
            ],
            max_tokens=100,
            model=model, 
            system_message="Respond in a friendly tone and include the model name in your response."
        )
        
        print(f"Response: {response}\n")
        
        await client.close()
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have at least one API key configured.\n")


async def streaming_chat():
    """Streaming example with real-time output."""
    print("=" * 60)
    print("Example 2: Streaming Chat")
    print("=" * 60)
    
    factory = AIClientFactory()
    
    try:
        client, model = factory.get_client()
        print(f"✓ Using: {model}\n")
        print("Response: ", end="", flush=True)
        
        # Stream response with callback
        response = await client.create_stream(
            model=model,
            system_message="Respond in a friendly tone and include the model name in your response.",
            user_message="Tell me a story about a magical adventure in a fantasy world.",
            max_tokens=3000,
            on_chunk=lambda chunk: print(chunk, end="", flush=True),
        )
        
        print("\n")  # New line after streaming
        await client.close()
        
    except ValueError as e:
        print(f"❌ Error: {e}\n")


async def provider_reordering():
    """Example of changing provider priority."""
    print("=" * 60)
    print("Example 3: Provider Reordering")
    print("=" * 60)
    
    factory = AIClientFactory()
    
    # Show original order
    providers = factory.list_providers()
    print(f"Original order: {', '.join([p['provider'] for p in providers])}\n")
    
    # Reorder to prefer Anthropic
    factory.reorder_clients(["xai", "openai", "gemini"])
    
    try:
        client, model = factory.get_client()
        print(f"✓ After reordering, using: {model}\n")
        
        response = await client.create_message(
            model=model,
            system_message="You are a helpful AI assistant.",
            messages=[{"role": "user", "content": "Just say 'Hello from' followed by your model name."}],
            max_tokens=50
        )
        
        print(f"Response: {response}\n")
        await client.close()
        
    except ValueError as e:
        print(f"❌ Error: {e}\n")


async def vision_example():
    """Example of using vision-capable models."""
    print("=" * 60)
    print("Example 4: Vision Support")
    print("=" * 60)
    
    factory = AIClientFactory()
    
    try:
        # Get vision-capable client
        client, model = factory.get_client(require_vision=True)
        print(f"✓ Using vision-capable model: {model}\n")
        
        # Example with base64 image (you would provide actual image data)
        # For this demo, just show that it accepts the format
        print("Note: Vision example requires actual image data.")
        print("Format: {'role': 'user', 'content': [")
        print("    {'type': 'text', 'text': 'What is in this image?'},")
        print("    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}")
        print("]}\n")
        
        await client.close()
        
    except ValueError as e:
        print(f"❌ Error: {e}\n")


async def multi_turn_conversation():
    """Example of a multi-turn conversation."""
    print("=" * 60)
    print("Example 5: Multi-turn Conversation")
    print("=" * 60)
    
    factory = AIClientFactory()
    
    try:
        client, model = factory.get_client()
        print(f"✓ Using: {model}\n")
        
        # Conversation history
        messages = [
            {"role": "user", "content": "What is 15 + 27?"}
        ]
        
        # First turn
        response1 = await client.create_message(
            model=model,
            system_message="You are a helpful math assistant.",
            messages=messages,
            max_tokens=5000
        )
        print(f"User: {messages[0]['content']}")
        print(f"Assistant: {response1}\n")
        
        # Add to history
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "Now multiply that by 2."})
        
        # Second turn
        response2 = await client.create_message(
            model=model,
            system_message="You are a helpful math assistant.",
            messages=messages,
            max_tokens=5000
        )
        print(f"User: {messages[2]['content']}")
        print(f"Assistant: {response2}\n")
        
        await client.close()
        
    except ValueError as e:
        print(f"❌ Error: {e}\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Failover Package - Quick Start Examples")
    print("=" * 60 + "\n")
    
    # await simple_chat()
    # await asyncio.sleep(1)  # Brief pause between examples
    
    # await streaming_chat()
    # await asyncio.sleep(1)
    
    # await provider_reordering()
    # await asyncio.sleep(1)
    
    # await vision_example()
    # await asyncio.sleep(1)
    
    await multi_turn_conversation()
    
    # print("=" * 60)
    # print("All examples complete!")
    # print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
