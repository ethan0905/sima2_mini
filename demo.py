#!/usr/bin/env python3
"""
Quick demo script for the SIMA-like agent.

This script demonstrates basic functionality without requiring
full installation or external dependencies.
"""

import sys
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_basic_functionality():
    """Demo basic functionality of all components."""
    print("🎮 SIMA-like Agent Demo")
    print("=" * 40)
    
    # Test environment
    print("\n1. Testing Environment...")
    from src.env.dummy_env import DummyGameEnv
    
    env = DummyGameEnv(grid_size=4, max_steps=20)
    obs = env.reset()
    print(f"   Environment initialized: {env.grid_size}x{env.grid_size} grid")
    print(f"   Agent at {env.agent_pos}, goal at {env.goal_pos}")
    
    # Test a few moves
    for move in ["right", "down", "right", "down"]:
        action = {"move": move}
        obs, reward, done, info = env.step(action)
        print(f"   Move {move}: agent at {env.agent_pos}, reward {reward:.1f}")
        if done:
            print(f"   Episode finished! Goal reached: {info.get('goal_reached', False)}")
            break
    
    # Test encoder
    print("\n2. Testing Observation Encoder...")
    from src.env.vision import DummyVisionEncoder
    
    encoder = DummyVisionEncoder(feature_dim=16)
    features = encoder.encode(obs)
    print(f"   Encoded {len(obs['pixels'])} pixel values to {len(features)} features")
    
    # Test policy
    print("\n3. Testing Policy...")
    from src.agent.policy import RandomPolicy
    
    policy = RandomPolicy()
    action = policy.act(features)
    print(f"   Random policy selected action: {action['move']}")
    
    # Test task
    print("\n4. Testing Task System...")
    from src.tasks.task_schema import Task
    from src.tasks.task_setter import TaskSetter
    from src.experience.buffer import ReplayBuffer
    
    buffer = ReplayBuffer()
    task_setter = TaskSetter(buffer)
    task = task_setter.propose_task()
    print(f"   Generated task: {task.id} - {task.description}")
    print(f"   Estimated reward: {task.estimated_reward}, Max steps: {task.max_steps}")
    
    # Test reward model
    print("\n5. Testing Reward Model...")
    from src.reward.reward_model import SimpleRewardModel
    
    reward_model = SimpleRewardModel()
    estimated_reward = reward_model.estimate_task_reward(task)
    print(f"   Reward model estimate: {estimated_reward:.2f}")
    
    # Test full episode
    print("\n6. Testing Full Episode...")
    from src.agent.agent import Agent
    
    # Reset environment for clean test
    env = DummyGameEnv(grid_size=4, max_steps=15)
    agent = Agent(env, policy, reward_model, encoder)
    
    episode = agent.run_episode(task)
    print(f"   Episode completed in {episode.length} steps")
    print(f"   Success: {episode.success}, Final reward: {episode.final_reward:.2f}")
    print(f"   Goal reached: {episode.reached_goal}")
    
    # Test experience storage
    print("\n7. Testing Experience Storage...")
    import tempfile
    from src.experience.storage import EpisodeStorage
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = EpisodeStorage(Path(temp_dir))
        storage.append_to_log(episode)
        
        loaded_episodes = storage.load_all_episodes()
        print(f"   Saved and loaded {len(loaded_episodes)} episode(s)")
    
    print("\n✅ All components working correctly!")
    print("\n📊 Demo Statistics:")
    print(f"   - Environment: {type(env).__name__}")
    print(f"   - Policy: {type(policy).__name__}")
    print(f"   - Episode length: {episode.length} steps")
    print(f"   - Final reward: {episode.final_reward:.2f}")
    print(f"   - Task success: {episode.success}")


def demo_mini_training():
    """Demo a mini training loop."""
    print("\n" + "=" * 40)
    print("🚀 Mini Training Demo")
    print("=" * 40)
    
    from src.env.dummy_env import DummyGameEnv
    from src.env.vision import DummyVisionEncoder
    from src.agent.policy import RandomPolicy
    from src.agent.agent import Agent
    from src.tasks.task_setter import TaskSetter
    from src.reward.reward_model import SimpleRewardModel
    from src.experience.buffer import ReplayBuffer
    
    # Setup components
    env = DummyGameEnv(grid_size=3, max_steps=12)  # Smaller for demo
    encoder = DummyVisionEncoder(feature_dim=8)
    policy = RandomPolicy()
    reward_model = SimpleRewardModel()
    agent = Agent(env, policy, reward_model, encoder)
    
    buffer = ReplayBuffer()
    task_setter = TaskSetter(buffer)
    
    print("\nRunning 3 generations with 2 episodes each...\n")
    
    # Mini training loop
    for generation in range(3):
        print(f"Generation {generation + 1}:")
        
        generation_episodes = []
        for episode_idx in range(2):
            # Get task and run episode
            task = task_setter.propose_task()
            episode = agent.run_episode(task)
            
            # Store in buffer
            buffer.add_episode(episode)
            generation_episodes.append(episode)
            
            # Update task setter
            task_setter.update_from_episode(episode)
            
            # Log results
            success_icon = "✅" if episode.success else "❌"
            print(f"  Episode {episode_idx + 1}: {task.id[:15]}... "
                  f"{success_icon} {episode.final_reward:.1f} reward, {episode.length} steps")
        
        # Calculate generation stats
        success_rate = sum(ep.success for ep in generation_episodes) / len(generation_episodes)
        avg_reward = sum(ep.final_reward for ep in generation_episodes) / len(generation_episodes)
        
        print(f"  📈 Generation stats: {success_rate:.0%} success, {avg_reward:.1f} avg reward")
        print()
    
    # Final statistics
    all_episodes = list(buffer)
    total_success_rate = sum(ep.success for ep in all_episodes) / len(all_episodes)
    total_avg_reward = sum(ep.final_reward for ep in all_episodes) / len(all_episodes)
    
    print(f"🏁 Training Complete!")
    print(f"   Total episodes: {len(all_episodes)}")
    print(f"   Overall success rate: {total_success_rate:.1%}")
    print(f"   Overall avg reward: {total_avg_reward:.2f}")
    
    # Show task statistics
    task_stats = buffer.get_task_statistics()
    print(f"   Unique tasks attempted: {len(task_stats)}")
    
    print("\n" + "=" * 40)


def main():
    """Run the demo."""
    try:
        demo_basic_functionality()
        demo_mini_training()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  1. Install the package: pip install -e .")
        print("  2. Run training: python -m src.main --mode train")
        print("  3. Try a single episode: python -m src.main --mode play-once")
        print("  4. Inspect results: python -m src.main --mode inspect-buffer")
        print("  5. For Minecraft control: python minecraft_chat.py")
        print("     (Make sure Minecraft is running and 'Pause on Lost Focus: OFF')")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
