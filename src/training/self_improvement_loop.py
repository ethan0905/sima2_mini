from __future__ import annotations

from typing import Optional

from ..agent.agent import Agent
from ..experience.buffer import ReplayBuffer
from ..experience.storage import EpisodeStorage
from ..reward.reward_model import RewardModel
from ..tasks.task_setter import TaskSetter
from ..utils.logging_utils import get_logger
from ..utils.metrics import TrainingMetrics, generate_training_report

__all__ = ["self_improvement_cycle"]

logger = get_logger(__name__)


def self_improvement_cycle(
    agent: Agent,
    task_setter: TaskSetter,
    reward_model: RewardModel,
    experience_buffer: ReplayBuffer,
    storage: EpisodeStorage,
    num_generations: int,
    episodes_per_generation: int,
    learning_frequency: int = 1,
    evaluation_frequency: int = 5,
    metrics_path: Optional[str] = None
) -> TrainingMetrics:
    """
    High-level SIMA-like self-improvement cycle.
    
    This function implements the core self-improvement loop that mirrors the
    conceptual diagram:
    
    For each generation:
        1. TaskSetter proposes tasks (with estimated rewards)
        2. Agent attempts each task in the GameEnv, producing Episodes
        3. RewardModel scores Episodes -> final_reward
        4. Episodes are stored in the Self-Generated Experience buffer and on disk
        5. Agent and RewardModel optionally update from sampled experience
    
    This creates a virtuous cycle where:
    - Failed tasks inform better task generation
    - Experience accumulates to improve policies
    - Reward models learn to better evaluate performance
    - No human demonstrations required
    
    Args:
        agent: Agent to train
        task_setter: Generator of tasks
        reward_model: Evaluator of episodes  
        experience_buffer: In-memory experience storage
        storage: Persistent experience storage
        num_generations: Number of training generations
        episodes_per_generation: Episodes per generation
        learning_frequency: How often to update agent/reward model (in generations)
        evaluation_frequency: How often to run evaluation episodes
        metrics_path: Optional path to save training metrics
        
    Returns:
        TrainingMetrics: Collected training metrics
    """
    logger.info(f"Starting self-improvement cycle for {num_generations} generations")
    logger.info(f"Episodes per generation: {episodes_per_generation}")
    
    # Initialize metrics collection
    metrics = TrainingMetrics()
    
    for generation in range(num_generations):
        logger.info(f"=== Generation {generation + 1}/{num_generations} ===")
        
        # Phase 1: Task Proposal and Episode Collection
        generation_episodes = []
        generation_rewards = []
        generation_successes = []
        
        for episode_idx in range(episodes_per_generation):
            # TaskSetter proposes a task with estimated reward
            task = task_setter.propose_task()
            
            logger.info(f"Episode {episode_idx + 1}: Attempting task '{task.id}' - {task.description}")
            logger.info(f"  Estimated reward: {task.estimated_reward:.2f}, Max steps: {task.max_steps}")
            
            # Agent attempts the task, producing an episode
            episode = agent.run_episode(task)
            
            # Log episode outcome
            logger.info(f"  Completed in {episode.length} steps")
            logger.info(f"  Final reward: {episode.final_reward:.2f}, Success: {episode.success}")
            logger.info(f"  Goal reached: {episode.reached_goal}, Truncated: {episode.was_truncated}")
            
            generation_episodes.append(episode)
            generation_rewards.append(episode.final_reward)
            generation_successes.append(episode.success)
            
            # Add to metrics
            metrics.add_episode(
                reward=episode.final_reward,
                success=episode.success,
                length=episode.length,
                task_id=episode.task_id
            )
            
            # Update TaskSetter with episode outcome
            task_setter.update_from_episode(episode)
        
        # Phase 2: Experience Storage
        for episode in generation_episodes:
            # Add to in-memory buffer
            experience_buffer.add_episode(episode)
            
            # Persist to disk
            storage.append_to_log(episode)
            
        logger.info(f"Added {len(generation_episodes)} episodes to experience buffer (total: {len(experience_buffer)})")
        
        # Phase 3: Learning Updates
        if generation % learning_frequency == 0 and len(experience_buffer) > 0:
            logger.info("Performing learning updates...")
            
            # Sample experience for learning
            learning_sample_size = min(50, len(experience_buffer))
            learning_episodes = experience_buffer.sample_episodes(learning_sample_size)
            
            # Update reward model from experience
            reward_model.update_from_experience(learning_episodes)
            
            # Update agent policy from experience
            agent.improve_from_episodes(learning_episodes)
            
            logger.info(f"Updated models using {len(learning_episodes)} episodes")
        
        # Phase 4: Progress Evaluation
        if generation % evaluation_frequency == 0:
            logger.info("Running evaluation...")
            
            # Evaluate on a representative task
            eval_task = task_setter.propose_task()
            eval_results = agent.evaluate_on_task(eval_task, num_episodes=3)
            
            logger.info(f"Evaluation results on '{eval_task.id}':")
            logger.info(f"  Success rate: {eval_results['success_rate']:.2%}")
            logger.info(f"  Avg reward: {eval_results['avg_reward']:.2f}")
            logger.info(f"  Avg steps: {eval_results['avg_steps']:.1f}")
            
        # Phase 5: Progress Reporting
        gen_avg_reward = sum(generation_rewards) / len(generation_rewards)
        gen_success_rate = sum(generation_successes) / len(generation_successes)
        gen_avg_length = sum(ep.length for ep in generation_episodes) / len(generation_episodes)
        
        metrics.add_generation(gen_avg_reward, gen_success_rate, gen_avg_length)
        
        _log_generation_summary(generation, generation_episodes, experience_buffer, 
                               task_setter, reward_model)
    
    # Save metrics if path provided
    if metrics_path:
        metrics.save_to_file(metrics_path)
        
        # Generate and log training report
        report = generate_training_report(metrics)
        logger.info(f"\n{report}")
    
    logger.info("=== Self-improvement cycle completed ===")
    _log_final_summary(experience_buffer, storage, task_setter, reward_model)
    
    return metrics


def _log_generation_summary(generation: int, 
                          episodes: list, 
                          buffer: ReplayBuffer,
                          task_setter: TaskSetter,
                          reward_model: RewardModel) -> None:
    """Log summary statistics for a generation."""
    
    # Episode statistics
    success_rate = sum(ep.success for ep in episodes) / len(episodes)
    avg_reward = sum(ep.final_reward for ep in episodes) / len(episodes)
    avg_steps = sum(ep.length for ep in episodes) / len(episodes)
    
    logger.info(f"Generation {generation + 1} Summary:")
    logger.info(f"  Episodes: {len(episodes)}")
    logger.info(f"  Success rate: {success_rate:.2%}")
    logger.info(f"  Avg final reward: {avg_reward:.2f}")
    logger.info(f"  Avg episode length: {avg_steps:.1f}")
    
    # Buffer statistics
    buffer_stats = buffer.get_task_statistics()
    if buffer_stats:
        logger.info(f"  Buffer: {len(buffer)} episodes across {len(buffer_stats)} task types")
    
    # Task setter statistics  
    task_stats = task_setter.get_task_statistics()
    if task_stats:
        most_attempted = max(task_stats.items(), key=lambda x: x[1]["num_attempts"])
        logger.info(f"  Most attempted task: {most_attempted[0]} ({most_attempted[1]['num_attempts']} attempts)")
    
    # Reward model statistics
    if hasattr(reward_model, 'get_model_statistics'):
        rm_stats = reward_model.get_model_statistics()
        if rm_stats:
            avg_completion_rate = sum(s["completion_rate"] for s in rm_stats.values()) / len(rm_stats)
            logger.info(f"  Avg task completion rate: {avg_completion_rate:.2%}")


def _log_final_summary(buffer: ReplayBuffer, 
                      storage: EpisodeStorage,
                      task_setter: TaskSetter, 
                      reward_model: RewardModel) -> None:
    """Log final summary statistics."""
    
    logger.info("Final Training Summary:")
    
    # Overall buffer statistics
    buffer_stats = buffer.get_task_statistics() 
    total_episodes = len(buffer)
    
    if buffer_stats:
        overall_success_rate = sum(
            stats["successful_episodes"] for stats in buffer_stats.values()
        ) / sum(stats["total_episodes"] for stats in buffer_stats.values())
        
        overall_avg_reward = sum(
            stats["total_reward"] for stats in buffer_stats.values()
        ) / sum(stats["total_episodes"] for stats in buffer_stats.values())
        
        logger.info(f"  Total episodes: {total_episodes}")
        logger.info(f"  Unique tasks: {len(buffer_stats)}")
        logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
        logger.info(f"  Overall avg reward: {overall_avg_reward:.2f}")
        
        # Best performing task
        best_task = max(buffer_stats.items(), key=lambda x: x[1]["success_rate"])
        logger.info(f"  Best task: {best_task[0]} ({best_task[1]['success_rate']:.2%} success)")
        
        # Most challenging task
        worst_task = min(buffer_stats.items(), key=lambda x: x[1]["success_rate"])
        logger.info(f"  Most challenging: {worst_task[0]} ({worst_task[1]['success_rate']:.2%} success)")
    
    # Storage statistics
    storage_stats = storage.get_storage_stats()
    logger.info(f"  Storage: {storage_stats['total_episodes']} episodes saved")
    logger.info(f"  Storage size: {storage_stats['storage_size_bytes'] / 1024:.1f} KB")
    
    # Task setter final stats
    task_stats = task_setter.get_task_statistics()
    if task_stats:
        logger.info(f"  Task setter tracked {len(task_stats)} task types")
        
    logger.info("Self-improvement cycle completed successfully!")


# TODO: Advanced training features would include:
# 
# def hierarchical_curriculum_cycle(...):
#     """
#     Enhanced cycle with hierarchical curriculum learning.
#     
#     - Start with simple tasks, gradually increase complexity
#     - Decompose complex tasks into simpler subtasks
#     - Track skill dependencies and prerequisites
#     """
#     pass
#     
# def distributed_self_improvement_cycle(...):
#     """
#     Distributed version for training on multiple environments.
#     
#     - Run multiple agent instances in parallel
#     - Shared experience buffer across workers
#     - Coordinated task distribution and load balancing
#     """
#     pass
#     
# def human_feedback_cycle(...):
#     """
#     Integration with human feedback for reward learning.
#     
#     - Present episode pairs for human preference annotation
#     - Update reward model based on human judgments
#     - Active learning to select most informative comparisons
#     """
#     pass
