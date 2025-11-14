#!/usr/bin/env python3
"""
Test script for screenshot functionality
"""
import sys
import os
sys.path.insert(0, 'src')

def test_screenshot_functionality():
    print("📸 Testing Screenshot Functionality...")
    print("=" * 50)
    
    # Test with custom screenshots folder
    test_folder = "test_screenshots"
    
    try:
        from vision.minecraft_vision import MinecraftVision
        
        print(f"1. Creating vision system with screenshots in '{test_folder}'...")
        vision = MinecraftVision(save_screenshots=True, screenshots_folder=test_folder)
        
        print("2. Capturing and saving screenshot...")
        screenshot = vision.capture_minecraft_screen()
        
        if screenshot is not None:
            print("   ✅ Screenshot captured successfully")
            
            # Check if folder was created and files exist
            if os.path.exists(test_folder):
                files = os.listdir(test_folder)
                print(f"   📁 Found {len(files)} files in {test_folder}")
                for file in files:
                    print(f"      - {file}")
                    
                # Test annotated screenshot
                print("3. Testing annotated screenshot...")
                game_state = vision.analyze_current_situation(save_annotated=True, user_request="Test screenshot functionality")
                
                # List files again
                files_after = os.listdir(test_folder)
                print(f"   📁 Now {len(files_after)} files in {test_folder}")
                for file in files_after:
                    if file not in files:
                        print(f"      + {file} (NEW)")
                        
            else:
                print(f"   ❌ Screenshot folder {test_folder} not created")
        else:
            print("   ❌ Failed to capture screenshot")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Screenshot functionality test complete!")

if __name__ == "__main__":
    test_screenshot_functionality()
