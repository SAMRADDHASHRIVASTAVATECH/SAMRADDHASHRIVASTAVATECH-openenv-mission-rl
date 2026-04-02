"""
inference.py
------------
OpenEnv Hackathon Inference Script
"""

import os
import sys

# Environment Variables (Required by OpenEnv)
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Import existing system components
from task_builder import build_tasks
from env import SpaceMissionEnv


def run_inference():
    """
    Run a complete episode using existing system.
    Output ONLY structured logs in OpenEnv format.
    """
    
    # Generate tasks using existing fallback pipeline
    tasks = build_tasks([], min_tasks=20, max_tasks=50)
    
    # Initialize environment
    env = SpaceMissionEnv(tasks, render_mode=None, scale_by_difficulty=True)
    
    # Reset and start episode
    obs, info = env.reset()
    
    # Print start marker (STRICT FORMAT)
    print("[START]")
    
    cumulative_reward = 0.0
    step_count = 0
    done = False
    
    # Run episode
    while not done:
        # Sample random action (simple agent)
        action = tuple(env.action_space.sample())
        
        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        
        # Update counters
        step_count += 1
        cumulative_reward += reward
        
        # Print step log (STRICT FORMAT - NO DEVIATIONS)
        print(f"[STEP] step={step_count} action={action} reward={reward:.2f} cumulative={cumulative_reward:.2f}")
    
    # Print end marker (STRICT FORMAT)
    print("[END]")
    
    # Validation checks (silent - no output)
    assert step_count >= 20, "Episode must have at least 20 steps"
    assert isinstance(cumulative_reward, float), "Cumulative reward must be numeric"
    
    return step_count, cumulative_reward


if __name__ == "__main__":
    try:
        run_inference()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)