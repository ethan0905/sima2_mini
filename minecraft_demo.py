#!/usr/bin/env python3
"""
🎮 SIMA Minecraft Integration Demo

This script demonstrates the complete Minecraft integration for the SIMA agent.
It shows off all the key components working together:

1. Environment setup (with fallback to raw control)
2. Task generation and management
3. Episode execution and logging  
4. Reward calculation with Minecraft-specific logic
5. Training integration

Usage:
    python minecraft_demo.py

Features:
- Automatic fallback when MineRL is not available
- Real-time logging of agent actions
- Minecraft-specific task proposals
- Progress tracking and reward calculation
"""

import sys
from pathlib import Path

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.config import SIMAConfig, MinecraftConfig, EnvironmentConfig, AgentConfig
from main import create_components
from tasks.task_schema import Task
import numpy as np


def demonstrate_minecraft_integration():
    """Main demo showing all Minecraft features."""
    
    print("🎮 SIMA Minecraft Integration Demo")
    print("=" * 50)
    
    # 1. Setup and Configuration
    print("\n1️⃣ Setting up Minecraft environment...")
    
    config = SIMAConfig()
    config.environment.env_type = 'minecraft'
    config.minecraft.use_minerl = True  # Will fallback to raw control if not available
    config.minecraft.max_episode_steps = 100
    config.agent.encoder_type = 'minecraft'
    
    # Create all components
    agent, task_setter, reward_model, buffer, storage = create_components(config)
    
    print(f"   ✅ Environment: {type(agent.env).__name__}")
    print(f"   ✅ Vision Encoder: {type(agent.encoder).__name__} (dim: {agent.encoder.get_feature_dim()})")
    print(f"   ✅ Policy: {type(agent.policy).__name__}")
    print(f"   ✅ Reward Model: {type(agent.reward_model).__name__}")
    
    # 2. Task Generation
    print("\n2️⃣ Generating Minecraft-specific tasks...")
    
    minecraft_tasks = []
    for i in range(15):
        task = task_setter.propose_task()
        if any(keyword in task.id for keyword in 
               ['collect_wood', 'build_pillar', 'craft_tools', 'survive_night', 'look_around', 'find_diamonds']):
            minecraft_tasks.append(task)
    
    print(f"   Found {len(minecraft_tasks)} Minecraft tasks:")
    for i, task in enumerate(minecraft_tasks[:4], 1):
        print(f"   {i}. {task.id}: {task.description}")
        print(f"      Reward: {task.estimated_reward:.1f}, Steps: {task.max_steps}")
    
    # 3. Environment Testing
    print("\n3️⃣ Testing environment interactions...")
    
    # Reset environment
    obs = agent.env.reset()
    print(f"   Initial observation shape: {obs['pixels'].shape}")
    print(f"   Info keys: {list(obs['info'].keys())}")
    
    # Test a few actions
    actions_to_test = [
        {"type": "move", "direction": "forward"},
        {"type": "look", "yaw": 15, "pitch": -10},
        {"type": "attack", "duration": 0.1}
    ]
    
    print("   Testing actions:")
    for i, action in enumerate(actions_to_test):
        obs, reward, done, info = agent.env.step(action)
        print(f"   Step {i+1}: {action} → reward={reward:.3f}, done={done}")
    
    # 4. Task Execution Demo
    print("\n4️⃣ Running a complete episode...")
    
    # Pick an interesting task
    demo_task = next((t for t in minecraft_tasks if 'collect_wood' in t.id), minecraft_tasks[0])
    print(f"   Task: {demo_task.id}")
    print(f"   Description: {demo_task.description}")
    
    # Run episode
    episode = agent.run_episode(demo_task)
    print(f"   ✅ Episode completed!")
    print(f"   Steps taken: {episode.length}")
    print(f"   Final reward: {episode.final_reward:.2f}")
    print(f"   Success: {episode.success}")
    
    # Show some actions
    if episode.transitions:
        print(f"   Sample actions taken:")
        for i, transition in enumerate(episode.transitions[:3]):
            action_type = transition.action.get('type', 'unknown')
            reward = transition.reward
            print(f"     Step {i+1}: {action_type} (reward: {reward:.3f})")
    
    # 5. Reward System Demo
    print("\n5️⃣ Demonstrating Minecraft reward calculation...")
    
    # Mock some Minecraft progress
    initial_info = {
        'minerl_info': {
            'inventory': {},
            'life': 20,
            'pov': {'position': [0, 64, 0]}
        }
    }
    
    final_info = {
        'minerl_info': {
            'inventory': {'log': 4, 'cobblestone': 2, 'coal': 1},
            'life': 19,
            'pov': {'position': [12, 64, 8]}
        }
    }
    
    progress_reward = reward_model._calculate_minecraft_progress(initial_info, final_info)
    
    print(f"   Mock scenario: Collected materials and explored")
    print(f"   Items gained: 4 logs, 2 cobblestone, 1 coal")
    print(f"   Health: 20 → 19 (minor damage)")
    print(f"   Distance: ~14.4 blocks traveled")
    print(f"   Progress reward: {progress_reward:.2f} points")
    
    print("\n6️⃣ Task statistics and learning...")
    
    # Show task performance tracking
    task_stats = task_setter.get_task_statistics()
    if task_stats:
        print(f"   Tracking {len(task_stats)} task types:")
        for task_id, stats in list(task_stats.items())[:3]:
            print(f"   • {task_id}: {stats['num_attempts']} attempts, "
                  f"{stats['success_rate']:.1%} success")
    
    # Final Summary
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("✅ Environment setup and fallback handling")
    print("✅ Minecraft-specific task generation")
    print("✅ Vision encoding for Minecraft observations")  
    print("✅ Action translation and environment interaction")
    print("✅ Minecraft progress detection and rewards")
    print("✅ Episode execution and experience storage")
    
    print(f"\n💡 Next steps:")
    print(f"   • Install MineRL: pip install minerl")
    print(f"   • Train the agent: python -m src.main --env minecraft --mode train")
    print(f"   • Try specific tasks: python -m src.main --env minecraft --mode play-once --task-id collect_wood_001")
    
    print(f"\n🔗 The SIMA agent is now ready for Minecraft research!")


if __name__ == "__main__":
    try:
        demonstrate_minecraft_integration()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
