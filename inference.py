"""
inference.py
------------
OpenEnv Hackathon Inference Script
Strict output format — no extra prints.
Runs ONCE then holds to prevent container restart.
"""

import os
import sys
import contextlib
import io
import time

# Environment Variables (Required by OpenEnv)
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Prevent double execution
_EXECUTED = False


def main():
    """
    Run one complete episode.
    Output ONLY [START], [STEP], [END] — nothing else.
    """
    global _EXECUTED
    if _EXECUTED:
        return
    _EXECUTED = True

    # Import inside main to prevent any module-level side effects
    from task_builder import build_tasks
    from env import SpaceMissionEnv

    # Silence ALL prints from task_builder and env init
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        tasks = build_tasks([], min_tasks=50, max_tasks=50)
        env = SpaceMissionEnv(tasks, render_mode=None, scale_by_difficulty=True)
        obs, info = env.reset()

    # Start marker
    print("[START]")
    sys.stdout.flush()

    cumulative = 0.0
    step = 0
    done = False

    while not done:
        # Sample action
        raw_action = env.action_space.sample()

        # Execute step (silence any internal prints)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            obs, reward, done, truncated, info = env.step(raw_action)

        # Update counters
        step += 1
        cumulative += float(reward)

        # Convert numpy int64 → clean Python int
        action_tuple = tuple(int(x) for x in raw_action)

        # Strict log format
        print(f"[STEP] step={step} action={action_tuple} reward={reward:.2f} cumulative={cumulative:.2f}")
        sys.stdout.flush()

    # End marker
    print("[END]")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    
    # Keep container alive to prevent restart
    # This is required for Hugging Face Spaces
    while True:
        time.sleep(3600)
