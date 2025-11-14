#!/usr/bin/env python3
"""
Debug script to test the vision system and see why the agent might be "blind"
"""
import sys
sys.path.insert(0, 'src')

def test_vision_system():
    print("🔍 Testing SIMA Vision System...")
    print("=" * 50)
    
    # Test imports
    print("1. Testing imports...")
    try:
        import cv2
        print("  ✅ OpenCV available")
    except ImportError:
        print("  ❌ OpenCV missing - this could be the issue!")
        return
    
    try:
        import numpy as np
        print("  ✅ NumPy available")
    except ImportError:
        print("  ❌ NumPy missing")
        return
    
    try:
        import mss
        print("  ✅ MSS (screen capture) available")
    except ImportError:
        print("  ❌ MSS missing - this could be the issue!")
        return
    
    # Test vision system creation
    print("\n2. Testing vision system creation...")
    try:
        from vision.minecraft_vision import MinecraftVision
        vision = MinecraftVision()
        print("  ✅ MinecraftVision created successfully")
    except Exception as e:
        print(f"  ❌ Failed to create MinecraftVision: {e}")
        return
    
    # Test screen capture
    print("\n3. Testing screen capture...")
    try:
        screenshot = vision.capture_minecraft_screen()
        if screenshot is not None:
            print(f"  ✅ Screenshot captured: {screenshot.shape}")
            print(f"  📊 Image stats: min={screenshot.min()}, max={screenshot.max()}, mean={screenshot.mean():.1f}")
        else:
            print("  ❌ Screenshot capture returned None")
            return
    except Exception as e:
        print(f"  ❌ Screen capture failed: {e}")
        return
    
    # Test vision analysis
    print("\n4. Testing vision analysis...")
    try:
        game_state = vision.analyze_current_situation()
        print(f"  📊 Health: {game_state.health}%")
        print(f"  🍖 Hunger: {game_state.hunger}%")
        print(f"  🎯 Current block: {game_state.current_block}")
        print(f"  👀 Entities: {game_state.nearby_entities}")
        print(f"  🌅 Time: {game_state.time_of_day}")
        
        situation = vision.get_situation_description()
        print(f"  💬 Situation: {situation}")
        
    except Exception as e:
        print(f"  ❌ Vision analysis failed: {e}")
        return
    
    # Test with agent
    print("\n5. Testing agent integration...")
    try:
        from agent.conversational_agent import ConversationalMinecraftAgent
        agent = ConversationalMinecraftAgent(auto_focus=False)
        
        if agent.enable_vision:
            print("  ✅ Agent has vision enabled")
            if agent.vision_system:
                print("  ✅ Vision system is initialized")
            else:
                print("  ❌ Vision system is None")
        else:
            print("  ❌ Agent vision is disabled")
            
    except Exception as e:
        print(f"  ❌ Agent integration test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Vision system test complete!")

if __name__ == "__main__":
    test_vision_system()
