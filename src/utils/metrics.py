"""
Training metrics collection and visualization utilities.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics across generations."""
    
    generation_rewards: List[float] = field(default_factory=list)
    generation_success_rates: List[float] = field(default_factory=list)
    generation_episode_lengths: List[float] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_success: List[bool] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    task_distribution: Dict[str, int] = field(default_factory=dict)
    task_success_rates: Dict[str, float] = field(default_factory=dict)
    
    def add_generation(self, 
                      avg_reward: float,
                      success_rate: float, 
                      avg_length: float) -> None:
        """Add generation-level metrics."""
        self.generation_rewards.append(avg_reward)
        self.generation_success_rates.append(success_rate)
        self.generation_episode_lengths.append(avg_length)
    
    def add_episode(self,
                   reward: float,
                   success: bool,
                   length: int,
                   task_id: str) -> None:
        """Add episode-level metrics."""
        self.episode_rewards.append(reward)
        self.episode_success.append(success)
        self.episode_lengths.append(length)
        
        # Update task distribution
        self.task_distribution[task_id] = self.task_distribution.get(task_id, 0) + 1
        
        # Update task success rates
        if task_id not in self.task_success_rates:
            self.task_success_rates[task_id] = 0.0
            
        # Simple running average (in practice, would be more sophisticated)
        current_count = self.task_distribution[task_id]
        current_successes = int(self.task_success_rates[task_id] * (current_count - 1))
        if success:
            current_successes += 1
        self.task_success_rates[task_id] = current_successes / current_count
    
    def get_summary(self) -> Dict:
        """Get a summary of all metrics."""
        if not self.episode_rewards:
            return {"status": "no_data"}
            
        return {
            "total_episodes": len(self.episode_rewards),
            "total_generations": len(self.generation_rewards),
            "overall_success_rate": np.mean(self.episode_success) if self.episode_success else 0.0,
            "overall_avg_reward": np.mean(self.episode_rewards),
            "overall_avg_length": np.mean(self.episode_lengths),
            "reward_std": np.std(self.episode_rewards),
            "best_episode_reward": max(self.episode_rewards),
            "worst_episode_reward": min(self.episode_rewards),
            "most_common_task": max(self.task_distribution, key=self.task_distribution.get) if self.task_distribution else None,
            "task_count": len(self.task_distribution),
            "best_task": max(self.task_success_rates, key=self.task_success_rates.get) if self.task_success_rates else None,
            "hardest_task": min(self.task_success_rates, key=self.task_success_rates.get) if self.task_success_rates else None
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "generation_rewards": self.generation_rewards,
            "generation_success_rates": self.generation_success_rates,
            "generation_episode_lengths": self.generation_episode_lengths,
            "episode_rewards": self.episode_rewards,
            "episode_success": self.episode_success,
            "episode_lengths": self.episode_lengths,
            "task_distribution": self.task_distribution,
            "task_success_rates": self.task_success_rates,
            "summary": self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Metrics saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Metrics file not found: {filepath}")
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.generation_rewards = data.get("generation_rewards", [])
        metrics.generation_success_rates = data.get("generation_success_rates", [])
        metrics.generation_episode_lengths = data.get("generation_episode_lengths", [])
        metrics.episode_rewards = data.get("episode_rewards", [])
        metrics.episode_success = data.get("episode_success", [])
        metrics.episode_lengths = data.get("episode_lengths", [])
        metrics.task_distribution = data.get("task_distribution", {})
        metrics.task_success_rates = data.get("task_success_rates", {})
        
        logger.info(f"Metrics loaded from {filepath}")
        return metrics


def create_ascii_plot(values: List[float], 
                     title: str = "Plot",
                     width: int = 50,
                     height: int = 10) -> str:
    """
    Create a simple ASCII plot for terminal display.
    
    Args:
        values: List of values to plot
        title: Plot title  
        width: Plot width in characters
        height: Plot height in characters
        
    Returns:
        Multi-line string containing the ASCII plot
    """
    if not values:
        return f"{title}: No data"
    
    # Normalize values to fit in plot area
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        # All values are the same
        normalized = [height // 2] * len(values)
    else:
        normalized = [
            int((val - min_val) / (max_val - min_val) * (height - 1))
            for val in values
        ]
    
    # Create plot grid
    lines = []
    
    # Title
    lines.append(f"{title}")
    lines.append(f"Range: {min_val:.2f} to {max_val:.2f}")
    lines.append("")
    
    # Plot area (from top to bottom)
    for row in range(height - 1, -1, -1):
        line = ""
        
        # Y-axis label
        if row == height - 1:
            line += f"{max_val:6.2f} ┤"
        elif row == 0:
            line += f"{min_val:6.2f} ┤"
        elif row == height // 2:
            line += f"{(max_val + min_val) / 2:6.2f} ┤"
        else:
            line += "       ┤"
        
        # Plot data points
        for i, norm_val in enumerate(normalized):
            if norm_val == row:
                line += "●"
            elif norm_val > row:
                line += "│"
            else:
                line += " "
        
        lines.append(line)
    
    # X-axis
    x_axis = "       └" + "─" * len(normalized)
    lines.append(x_axis)
    
    # X-axis labels
    if len(values) <= width:
        x_labels = "        " + "".join([f"{i%10}" for i in range(len(values))])
        lines.append(x_labels)
    else:
        # Show start, middle, end
        x_labels = f"        0{' ' * (len(normalized) - 3)}{len(values) - 1}"
        lines.append(x_labels)
    
    return "\n".join(lines)


def generate_training_report(metrics: TrainingMetrics,
                           experiment_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive training report.
    
    Args:
        metrics: Training metrics to report on
        experiment_path: Optional path to experiment directory
        
    Returns:
        Multi-line string report
    """
    if not metrics.episode_rewards:
        return "🤔 No training data available yet."
    
    summary = metrics.get_summary()
    
    report_lines = [
        "🎮 SIMA Agent Training Report",
        "=" * 50,
        "",
        "📊 Overall Performance:",
        f"  Total Episodes: {summary['total_episodes']}",
        f"  Total Generations: {summary['total_generations']}",
        f"  Success Rate: {summary['overall_success_rate']:.1%}",
        f"  Average Reward: {summary['overall_avg_reward']:.2f} ± {summary['reward_std']:.2f}",
        f"  Average Length: {summary['overall_avg_length']:.1f} steps",
        "",
        f"📈 Best Episode: {summary['best_episode_reward']:.2f} reward",
        f"📉 Worst Episode: {summary['worst_episode_reward']:.2f} reward",
        "",
    ]
    
    # Generation progress plot
    if len(metrics.generation_rewards) > 1:
        report_lines.extend([
            "📊 Reward Progress by Generation:",
            create_ascii_plot(
                metrics.generation_rewards,
                "Average Reward per Generation",
                width=50,
                height=8
            ),
            ""
        ])
    
    # Task analysis
    if metrics.task_distribution:
        report_lines.extend([
            "🎯 Task Analysis:",
            f"  Unique Tasks: {summary['task_count']}",
            f"  Most Attempted: {summary['most_common_task']} ({metrics.task_distribution.get(summary['most_common_task'], 0)} times)",
            f"  Best Task: {summary['best_task']} ({metrics.task_success_rates.get(summary['best_task'], 0):.1%} success)",
            f"  Hardest Task: {summary['hardest_task']} ({metrics.task_success_rates.get(summary['hardest_task'], 0):.1%} success)",
            ""
        ])
        
        # Top tasks by attempts
        sorted_tasks = sorted(metrics.task_distribution.items(), key=lambda x: x[1], reverse=True)
        report_lines.append("  Task Distribution:")
        for task_id, count in sorted_tasks[:5]:  # Top 5
            success_rate = metrics.task_success_rates.get(task_id, 0)
            report_lines.append(f"    {task_id[:30]:30} {count:3d} attempts ({success_rate:.1%} success)")
    
    if experiment_path:
        report_lines.extend([
            "",
            f"💾 Experiment Path: {experiment_path}",
            ""
        ])
    
    report_lines.extend([
        "=" * 50,
        "🚀 Ready for next training session!",
        ""
    ])
    
    return "\n".join(report_lines)


# TODO: Advanced visualization features:
# - Integration with matplotlib for rich plots
# - TensorBoard logging integration  
# - Real-time training dashboards
# - Hyperparameter tracking
# - Model architecture visualization
# - Task difficulty analysis
# - Learning curve analysis with confidence intervals
