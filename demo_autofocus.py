#!/usr/bin/env python3
"""
Quick demo of the enhanced auto-focus Minecraft agent.
Run this with Minecraft open to see automatic window focusing in action!
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_auto_focus():
    """Demo the auto-focus functionality."""
    print("🎮 Auto-Focus Minecraft Agent Demo")
    print("=" * 40)
    print("This demo shows automatic window focusing before each command.")
    print("Make sure Minecraft is open (windowed mode recommended)!")
    print()
    
    try:
        from agent.conversational_agent import ConversationalMinecraftAgent
        
        # Create agent with auto-focus enabled
        agent = ConversationalMinecraftAgent(openai_api_key=None, auto_focus=True)
        print("✅ Agent created with auto-focus enabled")
        
        # Demo commands that will trigger auto-focus
        demo_commands = [
            "go forward",
            "look left", 
            "jump",
            "look right",
            "go backward"
        ]
        
        print("\n🎯 Running demo commands with auto-focus...")
        print("   Watch how the agent automatically focuses Minecraft!")
        print()
        
        for i, cmd in enumerate(demo_commands, 1):
            print(f"📝 Command {i}: '{cmd}'")
            response = agent.process_instruction(cmd)
            print(f"🤖 Response: {response}")
            print()
            
            # Brief pause between commands
            import time
            time.sleep(2)
        
        print("🎉 Auto-focus demo complete!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Automatic Minecraft window detection")
        print("   • Fallback focus methods (screen center click)")
        print("   • Focus before each command execution")
        print("   • No manual window switching needed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dependencies are installed:")
        print("  pip install pyautogui pynput mss pillow")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("⚠️  This demo will send keyboard/mouse inputs to Minecraft!")
    print("Make sure Minecraft is open, or close it to test safely.")
    print()
    
    try:
        input("Press Enter to start demo, or Ctrl+C to cancel...")
        demo_auto_focus()
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled by user")
