#!/usr/bin/env python3
"""
Quick setup verification and instructions for Minecraft control.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking Dependencies...")
    print("-" * 30)
    
    missing_deps = []
    
    # Check core control libraries
    try:
        import pyautogui
        print("✅ pyautogui - OK")
    except ImportError:
        print("❌ pyautogui - MISSING")
        missing_deps.append("pyautogui")
    
    try:
        import pynput
        print("✅ pynput - OK")
    except ImportError:
        print("❌ pynput - MISSING")
        missing_deps.append("pynput")
    
    try:
        import mss
        print("✅ mss - OK")
    except ImportError:
        print("❌ mss - MISSING")
        missing_deps.append("mss")
    
    try:
        from PIL import Image
        print("✅ pillow - OK")
    except ImportError:
        print("❌ pillow - MISSING")
        missing_deps.append("pillow")
    
    # Check optional AI libraries
    try:
        import openai
        print("✅ openai - OK (enhanced chat available)")
    except ImportError:
        print("⚠️  openai - MISSING (basic mode only)")
    
    try:
        import pygetwindow
        print("✅ pygetwindow - OK (better window detection)")
    except ImportError:
        print("⚠️  pygetwindow - MISSING (fallback focus methods)")
    
    print()
    
    if missing_deps:
        print("❌ Missing required dependencies!")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    else:
        print("✅ All required dependencies installed!")
        return True

def show_minecraft_setup():
    """Show Minecraft setup instructions."""
    print("\n🎮 Minecraft Setup Instructions")
    print("=" * 35)
    print()
    print("1. 📱 CRITICAL SETTING:")
    print("   • Open Minecraft")
    print("   • Go to: Options → Controls")  
    print("   • Find: 'Pause on Lost Focus'")
    print("   • Set to: OFF")
    print("   This prevents pause menu when switching to terminal!")
    print()
    print("2. 🖥️  RECOMMENDED SETTINGS:")
    print("   • Use Windowed mode (not fullscreen)")
    print("   • Position Minecraft and terminal side-by-side")
    print("   • Make sure Minecraft window is visible")
    print()
    print("3. 🎯 FOCUS COMMANDS:")
    print("   • 'focus minecraft' - Brings Minecraft to front")
    print("   • 'unpause' - Closes pause menu if it opens")
    print("   • 'close menu' - Same as unpause")

def show_usage_examples():
    """Show usage examples."""
    print("\n💬 Example Commands")
    print("=" * 20)
    print()
    print("Basic Movement:")
    print("  • 'go forward'")
    print("  • 'turn left'")
    print("  • 'jump'")
    print()
    print("Actions:")
    print("  • 'mine this block'")
    print("  • 'place a block'")
    print("  • 'look around'")
    print()
    print("Focus Management:")
    print("  • 'focus minecraft'")
    print("  • 'unpause'")
    print()
    print("Complex Tasks:")
    print("  • 'build a 3x3 platform'")
    print("  • 'mine that tree'")
    print("  • 'find some animals'")

def main():
    """Main setup check."""
    print("🤖 SIMA Minecraft Agent - Setup Check")
    print("=" * 40)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Show Minecraft setup regardless
    show_minecraft_setup()
    
    # Show usage examples
    show_usage_examples()
    
    print(f"\n🚀 {'Ready to start!' if deps_ok else 'Install missing dependencies first'}")
    print()
    print("Start the agent with:")
    print("  python minecraft_chat.py")

if __name__ == "__main__":
    main()
