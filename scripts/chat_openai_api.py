"""
MiniMind Chat Client - OpenAI-Compatible API Interface

Provides an interactive command-line chat client for MiniMind models using the
OpenAI API format. Connects to a locally-running MiniMind API server and
maintains conversation history for multi-turn dialogue.

Features:
- Streaming and non-streaming response modes
- Configurable conversation history length
- Interactive CLI with keyboard interrupt handling
- OpenAI-compatible API integration

Requirements:
- openai>=1.0.0 Python package
- Running MiniMind API server (scripts/serve_openai_api.py)

Usage:
    python scripts/chat_openai_api.py
"""

from openai import OpenAI
from typing import List, Dict, Any
import sys


def create_chat_client(base_url: str = "http://127.0.0.1:8998/v1", api_key: str = "ollama") -> OpenAI:
    """
    Create OpenAI client configured for local MiniMind server.
    
    Args:
        base_url: Local API server URL (default: http://127.0.0.1:8998/v1)
        api_key: API key (dummy value for local server)
        
    Returns:
        Configured OpenAI client instance
    """
    return OpenAI(api_key=api_key, base_url=base_url)


def get_user_input() -> str:
    """Prompt user for input and return their query."""
    try:
        return input("\n[Q]: ")
    except (KeyboardInterrupt, EOFError):
        print("\n\nGoodbye!")
        sys.exit(0)


def print_response(response: Any, stream: bool = True) -> str:
    """
    Print assistant response and return complete text.
    
    Args:
        response: API response object from OpenAI client
        stream: Whether to stream tokens or print complete response
        
    Returns:
        Complete assistant response as string
    """
    if not stream:
        # Non-streaming mode: print complete response at once
        assistant_res = response.choices[0].message.content
        print(f"\n[A]: {assistant_res}")
        return assistant_res
    
    # Streaming mode: print tokens as they arrive
    print("\n[A]: ", end='', flush=True)
    assistant_res = ''
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        assistant_res += token
        
    return assistant_res


def run_interactive_chat(
    client: OpenAI,
    model_name: str = "minimind",
    history_messages_num: int = 2,
    stream: bool = True
) -> None:
    """
    Run interactive chat session with conversation history management.
    
    Maintains rolling conversation history, keeping only the most recent
    history_messages_num messages. Setting to 0 disables history for independent
    QA per turn. For best results, use an even number to maintain Q+A pairs.
    
    Args:
        client: OpenAI client instance
        model_name: Model name to use for completion
        history_messages_num: Number of recent messages to retain in history
                             (0 = no history, even number recommended for Q+A pairs)
        stream: Enable token-by-token streaming for responses
    """
    
    conversation_history: List[Dict[str, Any]] = []
    
    print("="*60)
    print("MiniMind Chat Client")
    print(f"Model: {model_name} | Streaming: {stream} | History: {history_messages_num} messages")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    while True:
        # Get user query
        query = get_user_input()
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": query})
        
        # Prepare messages (with or without history)
        messages = conversation_history[-history_messages_num:] if history_messages_num > 0 else [
            conversation_history[-1]  # Only current query, no history
        ]
        
        try:
            # Generate response from API
            response = client.chat.completions.create( 
                model=model_name,
                messages=messages, #type: ignore 
                stream=stream
            )
            
            # Print and collect assistant response
            assistant_res = print_response(response, stream)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": assistant_res})
            
            print("\n")  # Blank line for next turn
            
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Please ensure the API server is running at the configured URL.")


def main() -> None:
    """
    Main entry point for chat client.
    
    Configuration (edit as needed):
    - API server: http://127.0.0.1:8998/v1
    - API key: "ollama" (dummy value for local server)
    - Model: "minimind"
    - History: 2 messages (1 Q+A pair)
    - Streaming: Enabled for better UX
    """
    
    # Create client
    client = create_chat_client(
        base_url="http://127.0.0.1:8998/v1",
        api_key="ollama"
    )
    
    # Run chat session
    run_interactive_chat(
        client=client,
        model_name="minimind",
        history_messages_num=2,  # Recommend even number for Q+A pairs
        stream=True
    )


if __name__ == "__main__":
    main()