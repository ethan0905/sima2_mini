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
  python minecraft_chat.py                    # Auto-focus mode with screenshots (default)
  python minecraft_chat.py --manual-focus     # Manual focus mode (asks you to focus)
  python minecraft_chat.py --no-auto-focus    # Disable auto-focus (not recommended)
  python minecraft_chat.py --no-screenshots   # Disable screenshot saving
  python minecraft_chat.py --screenshots-folder my_pics  # Save to custom folder
  python minecraft_chat.py --openai-key sk-... # Start with full AI chat (gpt-4o-mini)
  
  OPENAI_API_KEY=sk-... python minecraft_chat.py  # Use environment variable

Instructions:
  1. Open Minecraft and load a world
  2. Start this script  
  3. The agent will try to focus Minecraft automatically
  4. If that fails, you'll be prompted to click Minecraft manually
  5. Type natural language instructions or vision questions
  6. Watch the agent control Minecraft for you!
  7. Screenshots are saved automatically for analysis debugging

CRITICAL SETUP:
  • Set Minecraft to WINDOWED mode (not fullscreen)
  • In Minecraft: Options → Controls → Set 'Pause on Lost Focus: OFF'
  • Position Minecraft and terminal windows side-by-side

Enhanced vision capabilities:
  You: "What do you see?"
  Agent: "I can see: health is excellent, looking at stone/dirt, holding wood tool..."
  
  You: "Is it night or day?"  
  Agent: "It's currently night time in the game."
  
  You: "What am I holding?"
  Agent: "You appear to be holding: wood item/tool"
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
    
    parser.add_argument(
        "--auto-focus",
        action="store_true",
        default=True,
        help="Automatically focus Minecraft before each command (default: True)"
    )
    
    parser.add_argument(
        "--no-auto-focus",
        action="store_true",
        help="Disable automatic Minecraft focusing"
    )
    
    parser.add_argument(
        "--manual-focus",
        action="store_true",
        help="Use manual focus mode (ask user to focus Minecraft before each command)"
    )
    
    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        default=True,
        help="Save screenshots during vision analysis (default: True)"
    )
    
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Disable screenshot saving"
    )
    
    parser.add_argument(
        "--screenshots-folder",
        default="screenshots",
        help="Folder to save screenshots in (default: screenshots)"
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
    
    # Determine auto-focus setting
    if args.manual_focus:
        auto_focus = False  # Manual mode - always ask user to focus
    else:
        auto_focus = not args.no_auto_focus  # Auto mode (default) or disabled
    
    # Determine screenshot settings
    save_screenshots = not args.no_screenshots  # Default True unless disabled
    screenshots_folder = args.screenshots_folder
    
    # Import and start the agent
    try:
        from agent.conversational_agent import start_conversational_agent
        start_conversational_agent(openai_api_key, args.model, auto_focus, save_screenshots, screenshots_folder)
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
    print("📦 Installation Commands for SIMA Minecraft Agent")
    print("=" * 65)
    
    print("\n1️⃣  Core Dependencies (Required):")
    print("   pip install numpy torch loguru")
    
    print("\n2️⃣  Vision & Intelligence (Recommended - enables smart adaptive behavior):")  
    print("   pip install opencv-python numpy mss pillow")
    
    print("\n3️⃣  Minecraft Control Dependencies (Required for real control):")  
    print("   pip install pyautogui pynput")
    
    print("\n4️⃣  AI Chat Dependencies (Optional but recommended):")
    print("   pip install openai")
    
    print("\n5️⃣  All at once (Full SIMA Agent):")
    print("   pip install -r requirements.txt")
    
    print("\n🔑 OpenAI API Key Setup (for enhanced chat):")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("   # Or add to your ~/.bashrc or ~/.zshrc")
    
    print("\n💡 Quick Start After Installation:")
    print("   1. Open Minecraft and load a world")
    print("   2. python minecraft_chat.py")
    print("   3. Type: 'find food' or 'what do you see?'")
    print("   4. Watch the intelligent agent work! ✨")
    
    print("\n🎯 SIMA Features with Full Installation:")
    print("   • 👁️  Computer vision - agent can see the game")
    print("   • 🧠 Intelligent planning - adapts actions to situation")
    print("   • 💬 Natural conversation - explains what it sees")
    print("   • 📊 Experience learning - gets better over time")
    print("   • ⚙️  Adaptive control - plans based on health, hunger, threats")
    
    print("\n⚠️  Safety Notes:")
    print("   • PyAutoGUI has a failsafe: move mouse to screen corner to stop")
    print("   • Start with simple commands to test the controls")  
    print("   • Make sure Minecraft is the focused window")
    print("   • Set Minecraft 'Pause on Lost Focus: OFF' (Options → Controls)")


if __name__ == "__main__":
    main()
