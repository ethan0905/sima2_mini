#!/usr/bin/env python3
"""
Quick test script to demonstrate Minecraft integration.

This script shows how to:
1. Create a MinecraftEnv with different configurations
2. Generate Minecraft-specific tasks 
3. Run a simple episode with Minecraft rewards
4. Display the results

Usage:
    python test_minecraft_integration.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config.config import SIMAConfig, MinecraftConfig, EnvironmentConfig  
from main import create_components
from tasks.task_setter import TaskSetter
from experience.types import Transition, Episode


def test_minecraft_environment():
    """Test creating and using the Minecraft environment."""
    print("🎮 Testing Minecraft Environment Integration")
    print("=" * 50)
    
    # Test MineRL configuration (if available)
    print("\n1. Testing MineRL configuration...")
    config_minerl = SIMAConfig(
        environment=EnvironmentConfig(env_type='minecraft'),
        minecraft=MinecraftConfig(use_minerl=True, env_id="MineRLTreechop-v0")
    )
    
    try:
        agent, task_setter, reward_model, buffer, storage = create_components(config_minerl)
        print("   ✓ MineRL environment created successfully")
    except Exception as e:
        print(f"   ⚠️  MineRL not available: {e}")
    
    # Test raw control configuration
    print("\n2. Testing raw control configuration...")
    config_raw = SIMAConfig(
        environment=EnvironmentConfig(env_type='minecraft'),
        minecraft=MinecraftConfig(use_minerl=False, max_episode_steps=100)
    )
    
    try:
        agent, task_setter, reward_model, buffer, storage = create_components(config_raw)
        print("   ✓ Raw control environment created successfully")
        
        # Test the environment interface
        env = agent.environment
        print(f"   Environment type: {type(env).__name__}")
        print(f"   Action space: {env.action_space}")
        
    except Exception as e:
        print(f"   ❌ Raw control failed: {e}")
        return False
    
    return True


def test_minecraft_tasks():
    """Test Minecraft-specific task generation."""
    print("\n3. Testing Minecraft task generation...")
    
    task_setter = TaskSetter()
    minecraft_tasks = []
    
    # Generate several tasks and look for Minecraft ones
    for _ in range(20):
        task = task_setter.propose_task()
        if any(keyword in task.id for keyword in ['collect_wood', 'look_around', 'build_pillar', 
                                                  'craft_tools', 'survive_night', 'find_diamonds']):
            minecraft_tasks.append(task)
    
    print(f"   Found {len(minecraft_tasks)} Minecraft-specific tasks:")
    for task in minecraft_tasks[:5]:  # Show first 5
        print(f"   • {task.id}: {task.description} (reward: {task.estimated_reward:.1f})")
    
    return len(minecraft_tasks) > 0


def test_minecraft_rewards():
    """Test Minecraft-specific reward calculation."""
    print("\n4. Testing Minecraft reward model...")
    
    from reward.reward_model import SimpleRewardModel
    from tasks.task_schema import Task
    
    reward_model = SimpleRewardModel()
    
    # Create a mock Minecraft task
    task = Task(
        id="collect_wood_001",
        description="Find and collect wood blocks by punching trees",
        estimated_reward=6.0,
        max_steps=200
    )
    
    # Create mock Minecraft episode with inventory changes
    initial_info = {
        "minerl_info": {
            "inventory": {},
            "life": 20,
            "pov": {"position": [0, 64, 0]}
        }
    }
    
    final_info = {
        "minerl_info": {
            "inventory": {"log": 5, "cobblestone": 2},
            "life": 20,
            "pov": {"position": [10, 64, 15]}
        }
    }
    
    # Create transitions
    transitions = [
        Transition(
            observation={"pixels": None},
            action={"type": "move", "direction": "forward"},
            reward=0.0,
            next_observation={"pixels": None},
            done=False,
            info=initial_info
        ),
        Transition(
            observation={"pixels": None},
            action={"type": "attack"},
            reward=0.0,
            next_observation={"pixels": None},
            done=True,
            info=final_info
        )
    ]
    
    episode = Episode(
        task_id=task.id,
        transitions=transitions,
        success=True,
        reached_goal=True
    )
    
    # Score the episode
    final_reward = reward_model.score_episode(task, episode)
    print(f"   Mock episode reward: {final_reward:.2f}")
    
    # Test progress calculation
    progress_reward = reward_model._calculate_minecraft_progress(initial_info, final_info)
    print(f"   Progress reward: {progress_reward:.2f}")
    print(f"   Inventory gains: 5 logs, 2 cobblestone")
    print(f"   Distance moved: ~18.0 blocks")
    
    return final_reward > 0


def main():
    """Run all integration tests."""
    print("🚀 SIMA Minecraft Integration Test")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_minecraft_environment),
        ("Task Generation", test_minecraft_tasks), 
        ("Reward Calculation", test_minecraft_rewards)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Minecraft integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next steps:")
    print("   1. Install MineRL: pip install minerl")
    print("   2. Run training: python -m src.main --env minecraft --mode train")
    print("   3. Try specific tasks: python -m src.main --env minecraft --mode play-once --task-id collect_wood")


if __name__ == "__main__":
    main()
