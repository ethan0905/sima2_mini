#!/usr/bin/env python3
"""
Simple and reliable Minecraft focus utility.
This creates a dedicated script just for focusing Minecraft.
"""

import subprocess
import sys
import time

def focus_minecraft_macos():
    """Focus Minecraft on macOS using AppleScript that targets Java process."""
    try:
        # Working AppleScript that targets the Java process (which runs Minecraft)
        script = '''
        tell application "System Events"
            if exists process "java" then
                tell process "java"
                    set frontmost to true
                    try
                        tell (first window whose name contains "Minecraft") to perform action "AXRaise"
                    end try
                end tell
                return "success"
            else
                return "java_not_found"
            end if
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if "success" in output:
                print("✅ Successfully focused Minecraft (Java process)")
                return True
            elif "java_not_found" in output:
                print("❌ Java process not found - is Minecraft running?")
                return False
        
        print(f"⚠️  AppleScript result: {result.stdout}")
        print(f"⚠️  AppleScript error: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Error focusing Minecraft: {e}")
        return False

def verify_minecraft_focus():
    """Verify if Minecraft is currently focused."""
    try:
        script = '''
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
            if frontApp is "java" then
                return "minecraft_focused"
            else
                return frontApp
            end if
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if "minecraft_focused" in output:
                print("✅ Verification: Minecraft (Java) is focused")
                return True
            else:
                print(f"⚠️  Verification: Current focus is '{output}', not Minecraft")
                return False
                
        return False
        
    except Exception as e:
        print(f"❌ Error verifying focus: {e}")
        return False

def main():
    """Test the focus functionality."""
    print("🎯 Testing Minecraft Focus...")
    
    if sys.platform == "darwin":  # macOS
        success = focus_minecraft_macos()
    else:
        print("❌ This focus script currently only supports macOS")
        print("💡 On other platforms, manually click the Minecraft window")
        success = False
    
    if success:
        # Verify if Minecraft is actually focused
        time.sleep(1)  # Wait a moment for the focus to take effect
        verify_minecraft_focus()
        
        print("🎮 Minecraft should now be focused!")
        print("💡 You can now use the chat agent commands")
    else:
        print("❌ Could not focus Minecraft automatically")
        print("💡 Please manually click on the Minecraft window")

if __name__ == "__main__":
    main()
