#!/usr/bin/env python3
"""
Test the enhanced agent with screenshot functionality
"""
import sys
sys.path.insert(0, 'src')

def test_agent_with_screenshots():
    print("🎮 Testing SIMA Agent with Screenshot Functionality...")
    print("=" * 60)
    
    from agent.conversational_agent import ConversationalMinecraftAgent
    
    # Create agent with custom screenshot folder
    screenshot_folder = "agent_screenshots"
    agent = ConversationalMinecraftAgent(
        auto_focus=False, 
        save_screenshots=True, 
        screenshots_folder=screenshot_folder
    )
    
    print("\n📊 Testing status display...")
    agent._show_status()
    
    print("\n🔍 Testing vision question...")
    response = agent._parse_basic_command("What do you see?")
    print(f"Agent Response: {response}")

if __name__ == "__main__":
    test_agent_with_screenshots()
