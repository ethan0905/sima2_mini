#!/usr/bin/env python3
"""
Conversational Minecraft Agent CLI

Entry point for the chat-based Minecraft assistant that can understand
natural language instructions and control the real Minecraft game.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point for the conversational agent."""
    parser = argparse.ArgumentParser(
        description="Conversational Minecraft Agent - Chat with an AI that can play Minecraft!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minecraft_chat.py                    # Start with basic commands
  python minecraft_chat.py --openai-key sk-... # Start with full AI chat (gpt-4o-mini)
  python minecraft_chat.py --model o1-mini   # Use o1-mini model
  
  OPENAI_API_KEY=sk-... python minecraft_chat.py  # Use environment variable

Instructions:
  1. Make sure Minecraft is running and focused
  2. Start this script  
  3. Type natural language instructions
  4. Watch the agent control Minecraft for you!

Example conversations:
  You: "Go forward and mine that tree"
  Agent: "I'll move forward and break those wood blocks for you!"
  
  You: "Build a small house here"  
  Agent: "I'll help you build a house! What materials should I use?"
  
  You: "Look around for animals"
  Agent: "I'll turn the camera to scan for nearby animals!"
        """
    )
    
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key for enhanced chat (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini", 
        help="OpenAI model to use (gpt-4o-mini, gpt-4o, o1-mini, etc.)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if required dependencies are installed"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true", 
        help="Show commands to install required dependencies"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.install_deps:
        show_installation_commands()
        return
    
    # Get OpenAI API key
    openai_api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    
    # Import and start the agent
    try:
        from agent.conversational_agent import start_conversational_agent
        start_conversational_agent(openai_api_key, args.model)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you're running from the project root directory:")
        print("   cd /path/to/sima2_mini")
        print("   python minecraft_chat.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies for Conversational Minecraft Agent...")
    print("=" * 60)
    
    # Core dependencies (should be installed)
    core_deps = {
        "numpy": "numpy",
        "torch": "torch", 
        "loguru": "loguru"
    }
    
    # Optional dependencies for full functionality
    optional_deps = {
        "OpenAI API": "openai",
        "Screen Capture": "mss", 
        "Image Processing": "PIL",
        "Keyboard Control": "pynput",
        "Mouse Control": "pyautogui"
    }
    
    missing_core = []
    missing_optional = []
    
    # Check core dependencies
    print("📦 Core Dependencies:")
    for name, module in core_deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}: Available")
        except ImportError:
            print(f"  ❌ {name}: Missing")
            missing_core.append(module)
    
    print("\n🎯 Minecraft Control Dependencies:")
    for name, module in optional_deps.items():
        try:
            __import__(module) 
            print(f"  ✅ {name}: Available")
        except ImportError:
            print(f"  ❌ {name}: Missing")
            missing_optional.append(module)
    
    # Summary
    print("\n" + "=" * 60)
    if not missing_core and not missing_optional:
        print("🎉 All dependencies are installed! You're ready to go!")
    elif not missing_core:
        print("✅ Core dependencies OK")
        print("⚠️  Some optional dependencies missing - agent will have limited functionality")
        print("   Run --install-deps for installation commands")
    else:
        print("❌ Missing core dependencies - please install them first")
        print("   Run --install-deps for installation commands")
    
    # Check environment
    print(f"\n🔑 OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")


def show_installation_commands():
    """Show commands to install dependencies."""
    print("📦 Installation Commands for Conversational Minecraft Agent")
    print("=" * 65)
    
    print("\n1️⃣  Core Dependencies (Required):")
    print("   pip install numpy torch loguru")
    
    print("\n2️⃣  Minecraft Control Dependencies (Required for real control):")  
    print("   pip install pyautogui pynput mss pillow")
    
    print("\n3️⃣  AI Chat Dependencies (Optional but recommended):")
    print("   pip install openai")
    
    print("\n4️⃣  All at once:")
    print("   pip install numpy torch loguru pyautogui pynput mss pillow openai")
    
    print("\n🔑 OpenAI API Key Setup (for enhanced chat):")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("   # Or add to your ~/.bashrc or ~/.zshrc")
    
    print("\n💡 Quick Start After Installation:")
    print("   1. Open Minecraft and load a world")
    print("   2. python minecraft_chat.py")
    print("   3. Type: 'go forward and mine some wood'")
    print("   4. Watch the magic happen! ✨")
    
    print("\n⚠️  Safety Notes:")
    print("   • PyAutoGUI has a failsafe: move mouse to screen corner to stop")
    print("   • Start with simple commands to test the controls")  
    print("   • Make sure Minecraft is the focused window")


if __name__ == "__main__":
    main()
