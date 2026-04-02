"""
inference.py
"""

import os
import io
import contextlib

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

from task_builder import build_tasks
from env import SpaceMissionEnv

silent = io.StringIO()
with contextlib.redirect_stdout(silent):
    tasks = build_tasks([], min_tasks=50, max_tasks=50)
    env = SpaceMissionEnv(tasks)
    obs, info = env.reset()

print("[START]")

cumulative = 0.0
step = 0
done = False

while not done:
    action = env.action_space.sample()
    
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obs, reward, done, truncated, info = env.step(action)
    
    step += 1
    cumulative += float(reward)
    action_tuple = tuple(int(x) for x in action)
    
    print(f"[STEP] step={step} action={action_tuple} reward={reward:.2f} cumulative={cumulative:.2f}")

print("[END]")

# Keep process alive (prevents restart loop)
import time
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
