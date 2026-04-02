"""
env.py
------
Gymnasium-compatible reinforcement learning environment for
Space Mission Control decisions.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from evaluator import evaluate_action
from rewards import compute_reward, reward_summary


class SpaceMissionEnv(gym.Env):
    """
    Observation : dict  {"description": str, "difficulty": int}
    Action      : Tuple[int, int, int]
                  (category ∈ {0,1,2}, priority ∈ {0,1,2}, decision ∈ {0,1,2})
    """

    metadata = {"render_modes": ["human"]}

    # ---------------------------------------------------------------- #
    #  Construction
    # ---------------------------------------------------------------- #

    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        render_mode: Optional[str] = None,
        scale_by_difficulty: bool = True,
    ):
        super().__init__()

        if not tasks:
            raise ValueError("Task list must not be empty.")

        self.tasks = tasks
        self.render_mode = render_mode
        self.scale_by_difficulty = scale_by_difficulty

        # Spaces
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        # Observation space — we expose difficulty as a Box
        # and the description through info; gymnasium prefers numeric obs.
        self.observation_space = spaces.Dict({
            "difficulty": spaces.Box(low=1, high=5, shape=(1,), dtype=np.int32),
            "task_index": spaces.Box(
                low=0, high=len(self.tasks) - 1, shape=(1,), dtype=np.int32
            ),
        })

        # Internal state
        self._current_index: int = 0
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._episode_log: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- #
    #  Gym API
    # ---------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        self._current_index = 0
        self._total_reward = 0.0
        self._step_count = 0
        self._done = False
        self._episode_log = []

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_task(info)

        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Parameters
        ----------
        action : array-like of 3 ints  (category, priority, decision)

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping."
            )

        # Normalise action
        if hasattr(action, "tolist"):
            action = tuple(action.tolist())
        else:
            action = tuple(int(a) for a in action)

        assert len(action) == 3, "Action must have exactly 3 components."

        task = self.tasks[self._current_index]
        expected = task["expected"]
        difficulty = task.get("difficulty", 1)

        # Evaluate
        eval_result = evaluate_action(action, expected)
        score = eval_result["score"]
        reward = compute_reward(
            score, difficulty, self.scale_by_difficulty
        )

        # Logging
        step_log = {
            "step": self._step_count,
            "task_index": self._current_index,
            "action": action,
            "expected": (
                expected["category"],
                expected["priority"],
                expected["decision"],
            ),
            "score": score,
            "reward": reward,
            "difficulty": difficulty,
            "details": eval_result["details"],
        }
        self._episode_log.append(step_log)

        self._total_reward += reward
        self._step_count += 1
        self._current_index += 1

        terminated = self._current_index >= len(self.tasks)
        truncated = False
        self._done = terminated

        # Next observation (or final)
        if not terminated:
            obs = self._get_observation()
            info = self._get_info()
        else:
            obs = self._make_terminal_obs()
            info = self._get_terminal_info()

        # Attach step-specific data to info
        info["eval"] = eval_result
        info["reward_detail"] = reward_summary(score, difficulty, reward)
        info["step_log"] = step_log

        if self.render_mode == "human":
            self._render_step(step_log)
            if terminated:
                self._render_episode_summary()

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------- #

    def _get_observation(self) -> Dict[str, Any]:
        task = self.tasks[self._current_index]
        return {
            "difficulty": np.array(
                [task.get("difficulty", 1)], dtype=np.int32
            ),
            "task_index": np.array(
                [self._current_index], dtype=np.int32
            ),
        }

    def _get_info(self) -> Dict[str, Any]:
        task = self.tasks[self._current_index]
        return {
            "description": task["input"]["description"],
            "difficulty": task.get("difficulty", 1),
            "task_number": self._current_index + 1,
            "total_tasks": len(self.tasks),
            "total_reward_so_far": self._total_reward,
        }

    def _make_terminal_obs(self) -> Dict[str, Any]:
        return {
            "difficulty": np.array([0], dtype=np.int32),
            "task_index": np.array(
                [len(self.tasks) - 1], dtype=np.int32
            ),
        }

    def _get_terminal_info(self) -> Dict[str, Any]:
        return {
            "description": "EPISODE COMPLETE",
            "difficulty": 0,
            "task_number": len(self.tasks),
            "total_tasks": len(self.tasks),
            "total_reward_so_far": self._total_reward,
            "episode_log": self._episode_log,
        }

    # ---------------------------------------------------------------- #
    #  Render helpers
    # ---------------------------------------------------------------- #

    def _render_task(self, info: Dict[str, Any]):
        print("\n" + "=" * 70)
        print(
            f"  TASK {info['task_number']}/{info['total_tasks']}  "
            f"(difficulty {info['difficulty']})"
        )
        print("=" * 70)
        print(f"  {info['description']}")
        print("-" * 70)

    def _render_step(self, log: Dict[str, Any]):
        print(
            f"  Action: {log['action']}  |  Expected: {log['expected']}"
        )
        print(f"  {log['details']}")
        print(
            f"  Score: {log['score']}/3  |  "
            f"Reward: {log['reward']:+.2f}  |  "
            f"Cumulative: {self._total_reward:+.2f}"
        )

        # Show next task header if not done
        if not self._done:
            info = self._get_info()
            self._render_task(info)

    def _render_episode_summary(self):
        print("\n" + "#" * 70)
        print("  EPISODE SUMMARY")
        print("#" * 70)
        print(f"  Total steps  : {self._step_count}")
        print(f"  Total reward : {self._total_reward:+.2f}")
        perfect = sum(
            1 for lg in self._episode_log if lg["score"] == 3
        )
        print(f"  Perfect (3/3): {perfect}/{self._step_count}")
        avg_score = (
            sum(lg["score"] for lg in self._episode_log)
            / max(self._step_count, 1)
        )
        print(f"  Avg score    : {avg_score:.2f}/3")
        print("#" * 70 + "\n")

    # ---------------------------------------------------------------- #
    #  Utility
    # ---------------------------------------------------------------- #

    def get_episode_log(self) -> List[Dict[str, Any]]:
        return list(self._episode_log)

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        if self._done or self._current_index >= len(self.tasks):
            return None
        return self.tasks[self._current_index]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)