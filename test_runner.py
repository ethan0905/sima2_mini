#!/usr/bin/env python3
"""
Simple test runner for the SIMA agent.
"""

import sys
from pathlib import Path

# Add the project root to path so we can import the src package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running Basic Tests")
    print("=" * 30)
    
    # Test 1: Environment
    print("1. Testing Environment... ", end="")
    try:
        from src.env.dummy_env import DummyGameEnv
        env = DummyGameEnv(grid_size=3, max_steps=10)
        obs = env.reset()
        action = {"move": "right"}
        next_obs, reward, done, info = env.step(action)
        assert "pixels" in obs
        assert isinstance(reward, (int, float))
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # Test 2: Encoder
    print("2. Testing Encoder... ", end="")
    try:
        from src.env.vision import DummyVisionEncoder
        encoder = DummyVisionEncoder(feature_dim=16)
        features = encoder.encode(obs)
        assert features.shape[0] == 16
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # Test 3: Policy
    print("3. Testing Policy... ", end="")
    try:
        from src.agent.policy import RandomPolicy
        policy = RandomPolicy()
        action = policy.act(features)
        assert isinstance(action, dict)
        assert "move" in action
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # Test 4: Task System
    print("4. Testing Task System... ", end="")
    try:
        from src.tasks.task_schema import Task
        from src.tasks.task_setter import TaskSetter
        from src.experience.buffer import ReplayBuffer
        
        buffer = ReplayBuffer()
        task_setter = TaskSetter(buffer)
        task = task_setter.propose_task()
        
        assert len(task.id) > 0
        assert task.estimated_reward > 0
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # Test 5: Agent Integration
    print("5. Testing Agent Integration... ", end="")
    try:
        from src.agent.agent import Agent
        from src.reward.reward_model import SimpleRewardModel
        
        reward_model = SimpleRewardModel()
        agent = Agent(env, policy, reward_model, encoder)
        episode = agent.run_episode(task)
        
        assert episode.length > 0
        assert isinstance(episode.final_reward, (int, float))
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # Test 6: Experience Storage
    print("6. Testing Experience Storage... ", end="")
    try:
        import tempfile
        from src.experience.storage import EpisodeStorage
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = EpisodeStorage(Path(temp_dir))
            storage.append_to_log(episode)
            
            loaded = storage.load_all_episodes()
            assert len(loaded) == 1
            assert loaded[0].task_id == episode.task_id
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    return True


def run_integration_test():
    """Run a mini integration test."""
    print("\n🔗 Running Integration Test")
    print("=" * 30)
    
    try:
        # Import all components
        from src.env.dummy_env import DummyGameEnv
        from src.env.vision import DummyVisionEncoder
        from src.agent.policy import RandomPolicy
        from src.agent.agent import Agent
        from src.tasks.task_setter import TaskSetter
        from src.reward.reward_model import SimpleRewardModel
        from src.experience.buffer import ReplayBuffer
        
        # Create components
        env = DummyGameEnv(grid_size=3, max_steps=15)
        encoder = DummyVisionEncoder(feature_dim=8)
        policy = RandomPolicy()
        reward_model = SimpleRewardModel()
        agent = Agent(env, policy, reward_model, encoder)
        
        buffer = ReplayBuffer()
        task_setter = TaskSetter(buffer)
        
        print("Components created successfully")
        
        # Run mini training loop
        print("Running 2 episodes...")
        
        for i in range(2):
            task = task_setter.propose_task()
            episode = agent.run_episode(task)
            buffer.add_episode(episode)
            task_setter.update_from_episode(episode)
            
            success_icon = "✅" if episode.success else "❌"
            print(f"  Episode {i+1}: {task.id[:15]} {success_icon} "
                  f"reward={episode.final_reward:.1f}, steps={episode.length}")
        
        # Check results
        assert len(buffer) == 2
        stats = buffer.get_task_statistics()
        assert len(stats) > 0
        
        print("\n🎉 Integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 SIMA Agent Test Suite")
    print("=" * 40)
    
    success = True
    
    # Run basic tests
    if not run_basic_tests():
        success = False
    
    # Run integration test
    if not run_integration_test():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! The SIMA agent is working correctly.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
