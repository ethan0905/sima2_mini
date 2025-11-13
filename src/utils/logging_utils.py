from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

__all__ = ["get_logger", "setup_logging", "log_episode_summary", "log_training_metrics"]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up structured logging for the SIMA agent training.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file path to log to
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_episode_summary(logger: logging.Logger, 
                       episode_data: dict,
                       task_info: dict) -> None:
    """
    Log a structured summary of an episode.
    
    Args:
        logger: Logger instance to use
        episode_data: Dictionary containing episode information
        task_info: Dictionary containing task information
    """
    logger.info(f"Episode Summary:")
    logger.info(f"  Task: {task_info.get('id', 'unknown')} - {task_info.get('description', 'N/A')}")
    logger.info(f"  Duration: {episode_data.get('length', 0)} steps")
    logger.info(f"  Success: {episode_data.get('success', False)}")
    logger.info(f"  Final Reward: {episode_data.get('final_reward', 0.0):.2f}")
    logger.info(f"  Estimated Reward: {task_info.get('estimated_reward', 0.0):.2f}")
    logger.info(f"  Goal Reached: {episode_data.get('goal_reached', False)}")
    
    # Log any additional metadata
    metadata = episode_data.get('metadata', {})
    if metadata:
        logger.info(f"  Metadata: {metadata}")


def log_training_metrics(logger: logging.Logger,
                        generation: int,
                        metrics: dict) -> None:
    """
    Log training metrics for a generation.
    
    Args:
        logger: Logger instance to use
        generation: Current generation number
        metrics: Dictionary containing training metrics
    """
    logger.info(f"Generation {generation} Metrics:")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            if 0 < value < 1:  # Likely a rate/percentage
                logger.info(f"  {metric_name}: {value:.2%}")
            else:
                logger.info(f"  {metric_name}: {value:.3f}")
        else:
            logger.info(f"  {metric_name}: {value}")


class StructuredLogger:
    """
    A structured logger that maintains consistent formatting for key events
    in the SIMA training process.
    
    This logger provides methods for logging:
    - Task proposals and outcomes
    - Episode completions and evaluations
    - Learning updates and progress
    - System performance metrics
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        
    def log_task_proposal(self, task_id: str, description: str, estimated_reward: float) -> None:
        """Log a task proposal from the TaskSetter."""
        self.logger.info(f"TASK_PROPOSAL | {task_id} | {estimated_reward:.2f} | {description}")
        
    def log_episode_start(self, task_id: str, agent_info: dict = None) -> None:
        """Log the start of an episode."""
        agent_str = f" | Agent: {agent_info}" if agent_info else ""
        self.logger.info(f"EPISODE_START | {task_id}{agent_str}")
        
    def log_episode_end(self, task_id: str, success: bool, reward: float, steps: int) -> None:
        """Log the completion of an episode.""" 
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(f"EPISODE_END | {task_id} | {status} | {reward:.2f} | {steps} steps")
        
    def log_learning_update(self, component: str, episodes_used: int, update_info: dict = None) -> None:
        """Log a learning update for a component."""
        info_str = f" | {update_info}" if update_info else ""
        self.logger.info(f"LEARNING_UPDATE | {component} | {episodes_used} episodes{info_str}")
        
    def log_evaluation_results(self, task_type: str, metrics: dict) -> None:
        """Log evaluation results."""
        metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        self.logger.info(f"EVALUATION | {task_type} | {metrics_str}")
        
    def log_milestone(self, milestone: str, details: dict = None) -> None:
        """Log a training milestone."""
        details_str = f" | {details}" if details else ""
        self.logger.info(f"MILESTONE | {milestone}{details_str}")


# Performance logging utilities

class PerformanceTracker:
    """
    Tracks and logs performance metrics over time.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history = {}
        
    def record_metric(self, metric_name: str, value: float, step: int) -> None:
        """Record a metric value at a specific step."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
            
        self.metrics_history[metric_name].append((step, value))
        
    def log_metric_summary(self, metric_name: str, window_size: int = 10) -> None:
        """Log a summary of recent metric values."""
        if metric_name not in self.metrics_history:
            return
            
        recent_values = [v for _, v in self.metrics_history[metric_name][-window_size:]]
        
        if recent_values:
            avg_value = sum(recent_values) / len(recent_values)
            min_value = min(recent_values)
            max_value = max(recent_values)
            
            self.logger.info(f"METRIC_SUMMARY | {metric_name} | "
                           f"avg: {avg_value:.3f} | min: {min_value:.3f} | max: {max_value:.3f} | "
                           f"samples: {len(recent_values)}")
            
    def get_metric_trend(self, metric_name: str, window_size: int = 20) -> Optional[str]:
        """Determine if a metric is trending up, down, or stable."""
        if metric_name not in self.metrics_history:
            return None
            
        values = [v for _, v in self.metrics_history[metric_name][-window_size:]]
        
        if len(values) < 5:
            return "insufficient_data"
            
        # Simple trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        relative_change = (second_avg - first_avg) / (first_avg + 1e-8)
        
        if relative_change > 0.05:
            return "improving"
        elif relative_change < -0.05:
            return "declining"
        else:
            return "stable"


# Example usage patterns:
#
# # Basic logging setup
# setup_logging(log_level="INFO", log_file=Path("training.log"))
# logger = get_logger(__name__)
#
# # Structured logging
# structured_logger = StructuredLogger("training")
# structured_logger.log_task_proposal("reach_goal_001", "Navigate to the goal", 8.5)
# structured_logger.log_episode_end("reach_goal_001", True, 9.2, 45)
#
# # Performance tracking
# perf_tracker = PerformanceTracker(logger)
# perf_tracker.record_metric("success_rate", 0.75, generation=10)
# perf_tracker.log_metric_summary("success_rate")
