#!/usr/bin/env python3
"""
Quick test to verify gpt-4o-mini is working correctly.
"""

import os
from src.agent.conversational_agent import ConversationalMinecraftAgent

def test_model_configuration():
    """Test that the model configuration is working."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  No OpenAI API key found - skipping model test")
        print("   Set OPENAI_API_KEY environment variable to test")
        return
    
    # Test default model (should be gpt-4o-mini)
    agent = ConversationalMinecraftAgent(api_key)
    print(f"✅ Default model: {agent.model}")
    
    # Test custom model
    agent_custom = ConversationalMinecraftAgent(api_key, model="o1-mini")
    print(f"✅ Custom model: {agent_custom.model}")
    
    # Test if OpenAI client is configured
    if agent.can_chat:
        print("✅ OpenAI client configured successfully")
        print("✅ Ready for conversational mode with gpt-4o-mini")
    else:
        print("❌ OpenAI client not configured")
    
    print("\n🎮 To start the conversational agent:")
    print("   python minecraft_chat.py")
    print(f"   (Will use {agent.model} by default)")

if __name__ == "__main__":
    test_model_configuration()
