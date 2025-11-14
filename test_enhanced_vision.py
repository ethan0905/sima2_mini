#!/usr/bin/env python3
"""
Test the enhanced vision system with window focusing and precise capture
"""
import sys
sys.path.insert(0, 'src')

def test_enhanced_vision():
    print("🔍 Testing Enhanced Vision System with Window Focusing...")
    print("=" * 70)
    
    # Test imports
    print("1. Testing imports...")
    try:
        from vision.minecraft_vision import MinecraftVision
        print("  ✅ MinecraftVision imported")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return
    
    # Test vision system with screenshot saving
    print("\n2. Creating vision system with screenshot saving...")
    try:
        vision = MinecraftVision(save_screenshots=True, screenshots_folder="test_screenshots")
        print("  ✅ Vision system created with screenshot saving enabled")
    except Exception as e:
        print(f"  ❌ Failed to create vision system: {e}")
        return
    
    # Test window focusing and capture
    print("\n3. Testing focused window capture...")
    try:
        # Test with focusing enabled
        screenshot = vision.capture_minecraft_screen(save_with_timestamp=True, force_focus=True)
        
        if screenshot is not None:
            print(f"  ✅ Focused screenshot captured: {screenshot.shape}")
            print(f"  📊 Image stats: min={screenshot.min()}, max={screenshot.max()}")
        else:
            print("  ❌ Failed to capture focused screenshot")
            
    except Exception as e:
        print(f"  ❌ Focused capture error: {e}")
    
    # Test vision analysis with the new system
    print("\n4. Testing complete vision analysis...")
    try:
        game_state = vision.analyze_current_situation()
        
        print(f"  📊 Health: {game_state.health:.1f}%")
        print(f"  🍖 Hunger: {game_state.hunger:.1f}%") 
        print(f"  🎯 Looking at: {game_state.current_block}")
        print(f"  🤲 Holding: {game_state.item_in_hand}")
        print(f"  👀 Entities: {', '.join(game_state.nearby_entities) if game_state.nearby_entities else 'None'}")
        print(f"  🌅 Time: {game_state.time_of_day}")
        
        situation = vision.get_situation_description()
        print(f"  💬 Situation: {situation}")
        
    except Exception as e:
        print(f"  ❌ Vision analysis failed: {e}")
    
    print("\n" + "=" * 70)
    print("🎯 Enhanced vision test complete!")
    
    # Check what screenshots were saved
    import os
    if os.path.exists("test_screenshots"):
        files = os.listdir("test_screenshots")
        print(f"\n📁 Screenshots saved: {len(files)} files")
        for f in files[-3:]:  # Show last 3 files
            print(f"   📸 {f}")

if __name__ == "__main__":
    test_enhanced_vision()
