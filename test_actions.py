#!/usr/bin/env python3
"""
Test the action execution functionality.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_action_execution():
    """Test that actions are properly executed."""
    print("🧪 Testing Action Execution")
    print("=" * 30)
    
    try:
        from agent.conversational_agent import ConversationalMinecraftAgent
        
        # Create agent (without OpenAI for testing)
        agent = ConversationalMinecraftAgent(openai_api_key=None)
        print("✅ Agent created successfully")
        
        print("\nTesting basic commands...")
        
        # Test movement commands
        test_commands = [
            "go forward",
            "move left", 
            "look right",
            "jump",
            "mine this block"
        ]
        
        for cmd in test_commands:
            print(f"\n🎯 Testing command: '{cmd}'")
            response = agent.process_instruction(cmd)
            print(f"📝 Response: {response}")
            time.sleep(1)  # Brief pause between commands
        
        print(f"\n✅ Action execution test complete!")
        print(f"\nIf Minecraft is open and focused, you should have seen:")
        print(f"  • Character moving forward and left")
        print(f"  • Camera turning right")
        print(f"  • Character jumping")
        print(f"  • Mining action (left click hold)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dependencies are installed:")
        print("  pip install pyautogui pynput mss pillow")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("⚠️  WARNING: This will send keyboard/mouse inputs!")
    print("Make sure Minecraft is open and focused, or close it to test safely.")
    print("Press Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
        test_action_execution()
    except KeyboardInterrupt:
        print("\n🛑 Test cancelled by user")
