"""
inference.py - OpenEnv Web API
"""

import os
import json
from flask import Flask, jsonify

from task_builder import build_tasks
from env import SpaceMissionEnv

app = Flask(__name__)

# Global environment (persists across requests)
_ENV = None
_TASKS = None


def init_env():
    """Initialize environment once."""
    global _ENV, _TASKS
    if _ENV is None:
        _TASKS = build_tasks([], min_tasks=50, max_tasks=50)
        _ENV = SpaceMissionEnv(_TASKS)
    return _ENV


@app.route("/reset", methods=["POST"])
def reset():
    """OpenEnv reset endpoint."""
    try:
        env = init_env()
        obs, info = env.reset()
        
        # Convert obs to JSON-serializable format
        obs_dict = {
            "difficulty": int(obs["difficulty"][0]),
            "task_index": int(obs["task_index"][0])
        }
        
        return jsonify({
            "status": "success",
            "observation": obs_dict,
            "info": info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/step", methods=["POST"])
def step():
    """OpenEnv step endpoint."""
    try:
        data = request.get_json()
        action = tuple(data.get("action", (0, 0, 0)))
        
        env = init_env()
        obs, reward, done, truncated, info = env.step(action)
        
        obs_dict = {
            "difficulty": int(obs["difficulty"][0]),
            "task_index": int(obs["task_index"][0])
        }
        
        return jsonify({
            "status": "success",
            "observation": obs_dict,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
