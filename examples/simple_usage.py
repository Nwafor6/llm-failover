#!/usr/bin/env python3
"""
Simple Usage Example - The easiest way to use llm-failover!

This shows the new ChatClient API with automatic failover.
Just call chat() or stream() - failover happens automatically!
"""

import asyncio
from llm_failover import ChatClient


async def example_1_basic_chat():
    """Example 1: Simple non-streaming chat."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Chat")
    print("=" * 60 + "\n")
    
    # Initialize once
    client = ChatClient()
    
    # Just call chat() - failover is automatic!
    response = await client.chat("What is Python? Answer in one sentence.")
    
    print(f"Provider: {response['provider']}")
    print(f"Model: {response['model']}")
    print(f"Response: {response['content']}\n")


async def example_2_streaming_chat():
    """Example 2: Streaming chat with callback."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Chat")
    print("=" * 60 + "\n")
    
    client = ChatClient()
    
    # Define callback for streaming chunks
    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)
    
    print("Response: ", end="", flush=True)
    
    # Stream with callback - failover is automatic!
    response = await client.stream(
        "Tell me a 2-sentence story about a robot.",
        on_chunk=on_chunk
    )
    
    print(f"\n\nProvider: {response['provider']}")
    print(f"Model: {response['model']}\n")


async def example_3_custom_settings():
    """Example 3: Custom provider order and settings."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Settings")
    print("=" * 60 + "\n")
    
    # Customize provider order and system message
    client = ChatClient(
        provider_order=["xai", "anthropic", "openai"],
        system_message="You are a helpful coding assistant.",
        max_tokens=200
    )
    
    response = await client.chat(
        "How do I reverse a list in Python?",
        temperature=0.7  # Pass any kwargs to the AI
    )
    
    print(f"Provider: {response['provider']}")
    print(f"Response: {response['content']}\n")


async def example_4_multi_turn_conversation():
    """Example 4: Multi-turn conversation with history."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-turn Conversation")
    print("=" * 60 + "\n")
    
    client = ChatClient()
    
    # First turn (keep_history=True maintains context)
    print("User: My name is Alice.")
    response = await client.chat(
        "My name is Alice.",
        keep_history=True
    )
    print(f"AI: {response['content']}\n")
    
    # Second turn (remembers previous context)
    print("User: What's my name?")
    response = await client.chat(
        "What's my name?",
        keep_history=True
    )
    print(f"AI: {response['content']}\n")
    
    # Third turn
    print("User: What was the first thing I told you?")
    response = await client.chat(
        "What was the first thing I told you?",
        keep_history=True
    )
    print(f"AI: {response['content']}\n")
    
    # Clear history when done
    client.clear_history()


async def example_5_custom_messages():
    """Example 5: Pass custom message history."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Message History")
    print("=" * 60 + "\n")
    
    client = ChatClient()
    
    # Build your own conversation history
    conversation = [
        {"role": "user", "content": "I love cooking Italian food."},
        {"role": "assistant", "content": "That's wonderful! Italian cuisine is delicious."},
        {"role": "user", "content": "What dish should I make tonight?"}
    ]
    
    response = await client.chat(
        message="",  # Message ignored when messages provided
        messages=conversation
    )
    
    print(f"Response: {response['content']}\n")


async def example_6_batch_processing():
    """Example 6: Process multiple questions efficiently."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60 + "\n")
    
    client = ChatClient(max_tokens=100)
    
    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "What does API stand for?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
        response = await client.chat(question)
        print(f"   → {response['content']} (via {response['provider']})\n")


async def example_7_async_streaming():
    """Example 7: Async streaming callback."""
    print("\n" + "=" * 60)
    print("Example 7: Async Streaming Callback")
    print("=" * 60 + "\n")
    
    client = ChatClient()
    
    # Async callback (useful for async I/O operations)
    async def async_on_chunk(chunk: str):
        print(chunk, end="", flush=True)
        # Could do async operations here like websocket.send(chunk)
    
    print("Response: ", end="", flush=True)
    
    response = await client.stream(
        "Count from 1 to 5.",
        on_chunk=async_on_chunk
    )
    
    print(f"\n\nCompleted!\n")


async def main():
    """Run all examples."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Simple Usage - LLM Failover" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    
    await example_1_basic_chat()
    await example_2_streaming_chat()
    await example_3_custom_settings()
    await example_4_multi_turn_conversation()
    await example_5_custom_messages()
    await example_6_batch_processing()
    await example_7_async_streaming()
    
    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
